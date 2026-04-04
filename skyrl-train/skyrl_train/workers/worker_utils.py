import math
from skyrl_train.dataset.replay_buffer import Experience
from typing import List, Dict
from skyrl_train.training_batch import TrainingInputBatch


def reduce_metrics(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Reduce metrics from a list of entries per key.
    """
    reduced_metrics = dict()
    for k, v in metrics.items():
        assert len(v) > 0, f"No metrics for key {k}"
        assert all(isinstance(x, (int, float)) for x in v), f"Metrics for key {k} are not all numbers"
        reduced_metrics[k] = sum(v) / len(v)
    return reduced_metrics


class BatchIterator:
    """A simple iterator to yield micro batches of data from the training batch."""

    def __init__(self, data: TrainingInputBatch, sample_batch_size: int, drop_last: bool = False):
        self.data = data
        self.sample_batch_size = sample_batch_size
        self.total_batch_size = data.batch_size
        self.drop_last = drop_last
        assert not drop_last, "drop_last is not supported yet"
        num_micro_batches = self.total_batch_size / self.sample_batch_size
        self.num_micro_batches = int(num_micro_batches) if drop_last else math.ceil(num_micro_batches)
        # TODO: switch to tensordict.map_iter if possible
        self._chunks = self.data.chunk(self.sample_batch_size)
        self._iter = iter(self._chunks)

    def __len__(self):
        return self.num_micro_batches

    def __iter__(self):
        return self

    def __next__(self) -> Experience:
        try:
            batch = next(self._iter)
            exp = self.batch_to_experience(batch)
            return exp
        except StopIteration:
            self._iter = iter(self._chunks)
            raise StopIteration

    @staticmethod
    def batch_to_experience(batch: TrainingInputBatch):
        # TODO (sumanthrh): other keys are not permitted right now, can go into info
        # TODO: this conversion is hidden right now, might need to be surfaced in worker explicitly.
        exp = Experience(
            sequences=batch["sequences"],
            action_log_probs=batch["action_log_probs"],
            base_action_log_probs=batch["base_action_log_probs"],
            values=batch["values"],
            returns=batch["returns"],
            advantages=batch["advantages"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
            action_mask=batch["response_mask"],
            num_actions=batch.metadata["response_length"],  # int
            rollout_logprobs=batch["rollout_logprobs"] if "rollout_logprobs" in batch else None,
            step_reward=batch["step_reward"] if "step_reward" in batch else None,
            done=batch["done"] if "done" in batch else None,
            bootstrap_mask=batch["bootstrap_mask"] if "bootstrap_mask" in batch else None,
            state_index=batch["state_index"] if "state_index" in batch else None,
            action_end_index=batch["action_end_index"] if "action_end_index" in batch else None,
            next_state_index=batch["next_state_index"] if "next_state_index" in batch else None,
            parsed_action_id=batch["parsed_action_id"] if "parsed_action_id" in batch else None,
            action_valid=batch["action_valid"] if "action_valid" in batch else None,
            q_values=batch["q_values"] if "q_values" in batch else None,
            v_values=batch["v_values"] if "v_values" in batch else None,
            next_v_values=batch["next_v_values"] if "next_v_values" in batch else None,
            q_targets=batch["q_targets"] if "q_targets" in batch else None,
            v_targets=batch["v_targets"] if "v_targets" in batch else None,
            # additional info
            # can be used to log metrics etc for micro-batches in the worker
            info={},
            # propagate metadata as is
            metadata=batch.metadata,
        )
        return exp
