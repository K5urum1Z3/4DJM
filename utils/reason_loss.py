from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ReasonTokenLayout:
    classification_token_index: int
    classification_token_id: int
    classification_token_string: str
    user_end_index: int
    reason_start_index: int | None
    reason_end_index: int | None
    reason_token_count: int
    decoded_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_last_subsequence(sequence: list[int], pattern: list[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    for start in range(len(sequence) - len(pattern), -1, -1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return -1


def build_reason_token_layout(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    expect_reason: bool,
) -> ReasonTokenLayout:
    """Locate assertion and reason spans without reading annotation metadata.

    The classification token is selected strictly before the first user
    <|im_end|>.  The reason span starts after the assistant role prefix and
    ends before the final <|im_end|>.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("The auditable reason baseline currently requires batch_size=1")
    ids = input_ids[0]
    attention = attention_mask[0].bool()
    positions = torch.arange(ids.numel(), device=ids.device)
    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    im_end_positions = torch.nonzero(ids.eq(im_end_id) & attention, as_tuple=False).flatten()
    if im_end_positions.numel() < 1:
        raise RuntimeError("No <|im_end|> token was found")
    user_end = int(im_end_positions[0].item())
    special_ids = torch.tensor(sorted(set(int(v) for v in tokenizer.all_special_ids)), device=ids.device)
    valid = attention & positions.lt(user_end)
    if special_ids.numel():
        valid = valid & ~(ids.unsqueeze(-1) == special_ids).any(dim=-1)
    selected = int(positions.masked_fill(~valid, -1).max().item())
    if selected < 0:
        raise RuntimeError("No normal assertion token was found before the user boundary")

    reason_start: int | None = None
    reason_end: int | None = None
    decoded_reason: str | None = None
    reason_count = 0
    if expect_reason:
        if im_end_positions.numel() < 2:
            raise RuntimeError("A reason target was requested but no assistant end token was found")
        assistant_prefix = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        start = _find_last_subsequence(ids.tolist(), [int(v) for v in assistant_prefix])
        if start < 0:
            raise RuntimeError("Assistant role prefix was not found in the tokenized sequence")
        reason_start = start + len(assistant_prefix)
        reason_end = int(im_end_positions[-1].item())
        if not (selected < reason_start < reason_end):
            raise RuntimeError(
                f"Invalid causal layout: classification={selected}, reason=[{reason_start}, {reason_end})"
            )
        reason_count = reason_end - reason_start
        decoded_reason = tokenizer.decode(ids[reason_start:reason_end].tolist(), skip_special_tokens=False)

    return ReasonTokenLayout(
        classification_token_index=selected,
        classification_token_id=int(ids[selected].item()),
        classification_token_string=tokenizer.decode([int(ids[selected].item())], skip_special_tokens=False),
        user_end_index=user_end,
        reason_start_index=reason_start,
        reason_end_index=reason_end,
        reason_token_count=reason_count,
        decoded_reason=decoded_reason,
    )


def reason_cross_entropy(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    lm_head: torch.nn.Module,
    layouts: list[ReasonTokenLayout],
    position_maps: list[torch.Tensor] | None = None,
    token_chunk_size: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Mean causal LM CE over reason content tokens only.

    Only reason-position logits are materialized.  A target token at position
    j is predicted by the hidden state at j-1, matching standard causal LM
    label shifting while excluding assistant-role and special tokens.

    ``position_maps`` is needed by multimodal backbones that remove visual
    tokens before the language transformer.  Each map converts a position in
    the processor's pre-compression ``input_ids`` sequence to the position of
    the corresponding hidden state after visual compression.  The default
    keeps the original Qwen3-VL path unchanged.
    """
    if position_maps is not None and len(position_maps) != len(layouts):
        raise ValueError("position_maps must have one map per reason layout")
    loss_sum = hidden_states.sum() * 0.0
    token_count = 0
    for batch_index, layout in enumerate(layouts):
        start, end = layout.reason_start_index, layout.reason_end_index
        if start is None or end is None or end <= start:
            continue
        predictor_positions = torch.arange(
            start - 1, end - 1, device=hidden_states.device, dtype=torch.long
        )
        if position_maps is None:
            post_positions = predictor_positions
        else:
            position_map = position_maps[batch_index].to(hidden_states.device)
            if predictor_positions.numel() and int(predictor_positions.max()) >= position_map.numel():
                raise ValueError("Reason predictor position exceeds the compression position map")
            post_positions = position_map[predictor_positions]
            if (post_positions < 0).any():
                raise ValueError("Reason predictor position was removed by visual compression")
        predictor_hidden = hidden_states[batch_index, post_positions]
        targets = input_ids[batch_index, start:end]
        chunk_size = int(token_chunk_size or targets.numel())
        if chunk_size < 1:
            raise ValueError("token_chunk_size must be positive")
        # Qwen3.5 has a large vocabulary.  Chunking keeps the temporary
        # [reason_tokens, vocab_size] logits bounded without changing the
        # summed causal LM objective or its gradients.
        for chunk_start in range(0, int(targets.numel()), chunk_size):
            chunk_end = min(chunk_start + chunk_size, int(targets.numel()))
            logits = lm_head(predictor_hidden[chunk_start:chunk_end])
            loss_sum = loss_sum + F.cross_entropy(
                logits.float(), targets[chunk_start:chunk_end], reduction="sum"
            )
        token_count += int(targets.numel())
    if token_count == 0:
        return loss_sum, 0
    return loss_sum / token_count, token_count


def reason_mask_rows(
    input_ids: torch.Tensor,
    tokenizer: Any,
    layout: ReasonTokenLayout,
) -> list[dict[str, Any]]:
    rows = []
    for index, token_id in enumerate(input_ids[0].tolist()):
        supervised = (
            layout.reason_start_index is not None
            and layout.reason_end_index is not None
            and layout.reason_start_index <= index < layout.reason_end_index
        )
        rows.append(
            {
                "index": index,
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "reason_mask": bool(supervised),
                "label": int(token_id) if supervised else -100,
            }
        )
    return rows
