from __future__ import annotations

from torch import nn

from models.temporal_model import FourDSAVStage3
from models.qwen35_lora_reason_binary import LANGUAGE_LORA_RE


class FourDSAVStage3LoRA(FourDSAVStage3):
    """Fixed Stage-1 custom weights, differentiable inputs, trainable LM LoRA."""

    def __init__(self, stage1, config):
        super().__init__(stage1, config)
        # Fine-tune the language adapters alongside the temporal verifier.
        selected = []
        for name, parameter in self.baseline.backbone.named_parameters():
            if LANGUAGE_LORA_RE.search(name):
                parameter.requires_grad_(True)
                selected.append(name)
        if not selected:
            raise RuntimeError("No approved Qwen3.5 language LoRA parameters found")
        self.stage1_frozen = False
        self.train(True)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.stage1.eval()
        # Track gradients through the frozen evidence modules to the language adapters.
        self.baseline.train(mode)
        self.baseline.head.eval()
        self.baseline.lm_head.eval()
        self.baseline.multimodal_base.visual.eval()
        return self

    def forward_from_frozen_features(self, *args, **kwargs):
        raise RuntimeError("Trainable LoRA invalidates cached S_A/E_A/o; use live language forward")
