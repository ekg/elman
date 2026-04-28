"""Synthetic task data generators for testing model expressivity.

Each task implements:
  generate_batch(B, T, rng) -> (input_tokens [B,T], target_tokens [B,T], loss_mask [B,T])

  - input_tokens: int64 token ids
  - target_tokens: int64 token ids (next-token prediction usually shifted)
  - loss_mask: bool, True where we want loss (often only the answer positions)

Vocabulary is small per-task (e.g. 4 for parity). Models can use vocab=256
(byte-level) by mapping task tokens into the byte range.
"""

from .parity import ParityTask
from .modular_counter import ModularCounterTask
from .dyck import DyckTask
from .fsm_tracking import FSMTrackingTask
from .selective_copy import SelectiveCopyTask
from .assoc_recall import AssocRecallTask

ALL_TASKS = {
    'parity': ParityTask,
    'modular_counter': ModularCounterTask,
    'dyck': DyckTask,
    'fsm_tracking': FSMTrackingTask,
    'selective_copy': SelectiveCopyTask,
    'assoc_recall': AssocRecallTask,
}
