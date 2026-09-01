"""Training metric records published for the loss monitor.

Used by trainers and, in trainerless mode, by head workers originating batches:
both publish the same ``LocalMetrics`` records and log lines, so the monitor and
benchmark tooling are agnostic to who injected the batch.
"""

import uuid

from typing import Literal

from pydantic import BaseModel, Field, StrictFloat, conint

from agora_server.hivemind.dht import DHT
from agora_server.hivemind.utils import get_dht_time, get_logger


logger = get_logger(__name__)


class LocalMetrics(BaseModel):
    id: str = Field(default_factory=lambda _: str(uuid.uuid4()))
    mode: Literal["train/loss", "eval/loss"]
    step: conint(ge=0, strict=True)
    loss: StrictFloat
    tps: StrictFloat


class TrainerMetricsReporter:
    """Emits training/eval metrics for a batch origin (trainer or trainerless head).

    Holds the per-origin counters and produces the log lines and DHT metric stores that
    downstream monitors scrape (``TrainerRegex`` / ``TrainerPromMonitor`` and the log-health
    verifier). The driver calls :meth:`report_train_step` once per training step and
    :meth:`report_eval` once per evaluation.
    """

    def __init__(
        self,
        dht: DHT,
        local_public_key: bytes,
        experiment_prefix: str,
        statistics_expiration: float,
        monitor_key_idx: int,
        per_device_train_batch_size: int,
        sequence_length: int | None = None,
        resume_from_token: int = 0,
    ):
        self.dht = dht
        self.local_public_key = local_public_key
        self.experiment_prefix = experiment_prefix
        self.statistics_expiration = statistics_expiration
        self.monitor_key_idx = monitor_key_idx
        self.per_device_train_batch_size = per_device_train_batch_size
        self.sequence_length = sequence_length
        self.last_update_time = get_dht_time()
        self.last_train_loss = None
        self.num_batches_processed = 0
        self.total_tokens_processed = resume_from_token
        self.logger = logger

    def report_train_step(self, loss: float | None, global_step: int) -> None:
        self.num_batches_processed += self.per_device_train_batch_size
        if self.sequence_length:
            self.total_tokens_processed += self.per_device_train_batch_size * self.sequence_length
        # A missing or zero loss (a completed batch whose loss report was lost) still counts
        # the batch and its tokens but never becomes a loss sample.
        if loss:
            self.last_train_loss = loss

        # Send training loss statistics periodically
        curr_time = get_dht_time()
        if (curr_time - self.last_update_time >= self.statistics_expiration) and (self.last_train_loss is not None):
            tps = (
                self.num_batches_processed * self.sequence_length / self.statistics_expiration
                if self.sequence_length
                else 0.0
            )
            self.logger.info(
                f"train loss = {str(self.last_train_loss)}, num batches process = {str(self.num_batches_processed)}, internal trainer step = {str(global_step)}"
            )
            self.logger.info(f"tokens per second = {tps:.2f}")
            self.logger.info(f"total_tokens_processed = {self.total_tokens_processed}")
            statistics = LocalMetrics(
                mode="train/loss", step=self.num_batches_processed, loss=self.last_train_loss, tps=tps
            )

            self.dht.store(
                key=f"{self.experiment_prefix}_{self.monitor_key_idx}_metrics_train",
                subkey=self.local_public_key,
                value=statistics.model_dump(),
                expiration_time=curr_time + self.statistics_expiration,
                return_future=True,
            )

            # Reset counters
            self.last_update_time = get_dht_time()
            self.num_batches_processed = 0

    def report_eval(self, eval_loss: float, global_step: int) -> None:
        self.logger.info(f"eval loss = {str(eval_loss)}, internal trainer step = {str(global_step)}")
        statistics = LocalMetrics(
            mode="eval/loss",
            step=0,  # Step not checked for validation
            loss=eval_loss,
            tps=0.0,
        )

        self.dht.store(
            key=f"{self.experiment_prefix}_{self.monitor_key_idx}_metrics_eval",
            subkey=self.local_public_key,
            value=statistics.model_dump(),
            expiration_time=get_dht_time() + self.statistics_expiration,
            return_future=True,
        )
