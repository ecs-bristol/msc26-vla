from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_smolvla_int4 import SmolVLAInt4Config


class SmolVLAInt4Policy(PreTrainedPolicy):
    """SmolVLA with a 4-bit weight-quantized VLM backbone.

    The inner policy is the unmodified LeRobot SmolVLA policy loaded from the
    checkpoint, so `select_action` semantics (action queue, seeded noise path,
    flow-matching denoising) are identical to the official `--policy.path`
    baseline. Only the VLM backbone weights are quantized.
    """

    config_class = SmolVLAInt4Config
    name = "smolvla_int4"

    def __init__(
        self,
        config: SmolVLAInt4Config,
        *,
        inner_policy: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.inner = inner_policy if inner_policy is not None else self._load_inner_policy(config)
        # Mirror the checkpoint's AMP setting so the official evaluator wraps the
        # run in the same autocast context as the fp16 baseline.
        config.use_amp = bool(getattr(getattr(self.inner, "config", None), "use_amp", False))
        self._quantize_vlm(config)

    def _load_inner_policy(self, config: SmolVLAInt4Config):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        inner_cfg = PreTrainedConfig.from_pretrained(config.checkpoint, revision=config.revision)
        if config.num_steps is not None:
            inner_cfg.num_steps = config.num_steps
        if config.n_action_steps is not None:
            inner_cfg.n_action_steps = config.n_action_steps
        # SmolVLA's nested VLM loader materializes cached safetensors on CPU
        # first; keep the inner load on CPU and move the assembled policy to the
        # target device afterwards (quantized weights are packed on device).
        inner_cfg.device = "cpu"
        policy_cls = get_policy_class(inner_cfg.type)
        return policy_cls.from_pretrained(
            config.checkpoint, config=inner_cfg, revision=config.revision
        )

    def _quantize_vlm(self, config: SmolVLAInt4Config) -> None:
        if config.quant_method == "none":
            return
        targets = self._vlm_quantization_targets()
        if config.quant_method == "int4_groupwise":
            self._quantize_groupwise(targets)
        elif config.quant_method == "int8_groupwise":
            self._quantize_int8(targets)
        elif config.quant_method == "mixed":
            self._quantize_mixed()
        elif config.quant_method == "bnb_nf4":
            compute_dtype = torch.float16 if config.use_amp else torch.float32
            self._quantize_bnb(targets, compute_dtype=compute_dtype)
        else:
            raise ValueError(f"unsupported quant_method: {config.quant_method}")

    def _find_vlm_module(self) -> nn.Module:
        model = getattr(self.inner, "model", None)
        vlm_with_expert = getattr(model, "vlm_with_expert", None) if model is not None else None
        vlm = getattr(vlm_with_expert, "vlm", None) if vlm_with_expert is not None else None
        if vlm is None:
            raise RuntimeError(
                "could not locate SmolVLA VLM backbone; expected policy.model.vlm_with_expert.vlm"
            )
        return vlm

    def _vlm_quantization_targets(self) -> list[nn.Module]:
        """Return the VLM submodules holding the bulk of the Linear weights.

        Embeddings and output heads are excluded to avoid tied-weight issues
        with weight-only quantizers; the action expert (Gemma) stays in fp16.
        """
        vlm = self._find_vlm_module()
        model = getattr(vlm, "model", None)
        if model is None:
            return [vlm]
        if self.config.quant_scope == "language":
            text_model = getattr(model, "text_model", None)
            if text_model is not None and hasattr(text_model, "layers"):
                return [text_model.layers]
            return [model]
        targets: list[nn.Module] = []
        for name in ("vision_model", "connector", "text_model"):
            part = getattr(model, name, None)
            if part is None:
                continue
            if name == "text_model" and hasattr(part, "layers"):
                targets.append(part.layers)
            else:
                targets.append(part)
        if not targets:
            targets.append(model)
        return targets

    def _vlm_part_map(self) -> dict[str, nn.Module]:
        """Map the VLM components to the names used by the mixed config."""
        vlm = self._find_vlm_module()
        model = getattr(vlm, "model", None)
        if model is None:
            return {"vlm": vlm}
        parts: dict[str, nn.Module] = {}
        for key, attr in (("vision", "vision_model"), ("connector", "connector"), ("text", "text_model")):
            part = getattr(model, attr, None)
            if part is None:
                continue
            if key == "text" and hasattr(part, "layers"):
                parts[key] = part.layers
            else:
                parts[key] = part
        return parts

    def _quantize_mixed(self) -> None:
        bit_map = {
            "vision": self.config.vision_bits,
            "connector": self.config.connector_bits,
            "text": self.config.text_bits,
        }
        for name, module in self._vlm_part_map().items():
            bits = bit_map.get(name, 16)
            if bits == 16:
                continue
            if bits == 8:
                self._quantize_int8([module])
            elif bits == 4:
                self._quantize_groupwise([module])

    def _quantize_groupwise(self, targets: list[nn.Module]) -> None:
        from .int4_linear import Int4WeightOnlyLinear

        group_size = self.config.group_size

        def replace(linear: nn.Linear) -> nn.Module:
            return Int4WeightOnlyLinear.from_float(linear, group_size=group_size)

        for target in targets:
            self._swap_linear_children(target, replace)

    def _quantize_int8(self, targets: list[nn.Module]) -> None:
        from .int8_linear import Int8WeightOnlyLinear

        group_size = self.config.group_size

        def replace(linear: nn.Linear) -> nn.Module:
            return Int8WeightOnlyLinear.from_float(linear, group_size=group_size)

        for target in targets:
            self._swap_linear_children(target, replace)

    def _swap_linear_children(
        self, module: nn.Module, replacer: Any
    ) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, replacer(child))
            else:
                self._swap_linear_children(child, replacer)

    def _quantize_bnb(self, targets: list[nn.Module], *, compute_dtype: torch.dtype) -> None:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError(
                "bitsandbytes is required for quant_method=bnb_nf4; run: pip install bitsandbytes"
            ) from exc

        def swap(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    quantized = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=compute_dtype,
                        compress_statistics=True,
                        quant_type="nf4",
                    )
                    quantized.weight = bnb.nn.Params4bit(
                        child.weight.detach().to(compute_dtype),
                        requires_grad=False,
                        compress_statistics=True,
                        quant_type="nf4",
                    )
                    if child.bias is not None:
                        quantized.bias = nn.Parameter(child.bias.detach(), requires_grad=False)
                    setattr(module, name, quantized)
                else:
                    swap(child)

        for target in targets:
            swap(target)

    def reset(self) -> None:
        self.inner.reset()

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return self.inner.select_action(batch, **kwargs)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return self.inner.predict_action_chunk(batch, **kwargs)

    def get_optim_params(self):
        raise RuntimeError("smolvla_int4 is an inference-only policy")

    def forward(self, *args: Any, **kwargs: Any):
        raise RuntimeError("smolvla_int4 is an inference-only policy")
