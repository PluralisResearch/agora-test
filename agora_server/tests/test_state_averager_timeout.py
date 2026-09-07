"""Contain unfinished ARs through the configured log monitor termination rule."""

import logging
import os
import signal
import threading

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from agora_server.core.averaging import state_averager
from agora_server.core.averaging.state_averager import TrainingStateAverager
from agora_server.logging.log_monitor import LogMonitor, MonitorRule


class AveragingControl(Future):
    def __init__(self, initial=None, after_timeout=None):
        super().__init__()
        self.triggered = False
        self.initial = initial
        self.after_timeout = after_timeout
        self.waits = []

    def allow_allreduce(self):
        self.triggered = True
        self.complete(self.initial)

    def complete(self, outcome):
        if isinstance(outcome, BaseException):
            self.set_exception(outcome)
        elif outcome is not None:
            self.set_result(outcome)

    def result(self, timeout=None):
        self.waits.append(timeout)
        if len(self.waits) == 2:
            self.complete(self.after_timeout)
        return super().result(timeout=timeout)


@pytest.fixture
def parent(monkeypatch):
    monkeypatch.setattr(state_averager, "_AR_TIMEOUT_GRACE", 0.01)
    # Catch regressions without signalling or exiting the test runner.
    killpg = Mock()
    exit_process = Mock(side_effect=AssertionError("The averager must not exit the process"))
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(os, "getpgrp", lambda: 12345)
    monkeypatch.setattr(os, "_exit", exit_process)
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    state_averager.logger.addHandler(handler)
    parent = SimpleNamespace(
        lock_optimizer=threading.Lock(),
        lock_averaging=threading.Lock(),
        lock_averaged_tensors=threading.Lock(),
        reuse_tensors=False,
        delta_rule_averaging=False,
        _averaging_failed=False,
        _update_scheduler=lambda: None,
        _load_local_tensors_into_averager_=Mock(),
        _save_pre_averaging_state=lambda: None,
        _apply_averaging_results_=Mock(),
        _apply_optimizer_parameters_=Mock(),
        finished_optimizer_step=threading.Event(),
        finished_averaging_round=threading.Event(),
        pending_updates=set(),
        delay_before_averaging=SimpleNamespace(update=lambda **kwargs: None),
        local_epoch=7026,
        status_loglevel=logging.INFO,
        sync_epoch_when_averaging=True,
        killpg=killpg,
        exit_process=exit_process,
        records=records,
    )
    parent._wait_for_averaging = lambda control, timeout: TrainingStateAverager._wait_for_averaging(
        parent, control, timeout
    )
    try:
        yield parent
    finally:
        state_averager.logger.removeHandler(handler)
        handler.close()


@pytest.fixture
def monitor(parent):
    config_path = Path(__file__).resolve().parents[2] / "agora/src/agora/configs/default.yaml"
    rules = yaml.safe_load(config_path.read_text())["log_monitor_rules"]
    monitor = LogMonitor(rules=[MonitorRule.from_dict(rule) for rule in rules])
    try:
        yield monitor
    finally:
        monitor.stop()
        monitor._log_queue.close()
        monitor._log_queue.join_thread()


def run_round(parent, control):
    return TrainingStateAverager._do(
        parent,
        wait_for_trigger=None,
        optimizer_step=False,
        zero_grad=False,
        averaging_round=True,
        averaging_control=control,
        grad_scaler=None,
        set_to_none=True,
        timeout=0,
    )


@pytest.mark.parametrize("outcome", [{"peer": 7027}, TimeoutError("child unwound"), RuntimeError("AR failed")])
def test_completed_round_keeps_existing_behavior(parent, outcome):
    control = AveragingControl(initial=outcome)
    run_round(parent, control)
    assert parent.finished_averaging_round.is_set()
    assert parent.local_epoch == (7027 if isinstance(outcome, dict) else 7026)
    assert control.waits == [0]
    assert not parent._averaging_failed
    assert not any(record.getMessage() == state_averager._AR_TIMEOUT_MESSAGE for record in parent.records)
    parent.killpg.assert_not_called()
    parent.exit_process.assert_not_called()


@pytest.mark.parametrize("outcome", [{"peer": 7027}, TimeoutError("child unwound"), RuntimeError("AR failed")])
def test_child_completes_during_grace(parent, outcome):
    control = AveragingControl(after_timeout=outcome)
    run_round(parent, control)
    assert control.done()
    assert parent.finished_averaging_round.is_set()
    assert parent.local_epoch == (7027 if isinstance(outcome, dict) else 7026)
    assert control.waits == [0, 0.01]
    assert not parent._averaging_failed
    assert not any(record.getMessage() == state_averager._AR_TIMEOUT_MESSAGE for record in parent.records)
    parent.killpg.assert_not_called()
    parent.exit_process.assert_not_called()


def test_pending_child_logs_fatal_error_without_publishing_locked_tensors(parent):
    control = AveragingControl()
    with parent.lock_averaged_tensors:
        with pytest.raises(state_averager._UnfinishedAllreduceError):
            run_round(parent, control)
    assert control.waits == [0, 0.01]
    assert not control.done()
    assert parent._averaging_failed
    assert not parent.finished_averaging_round.is_set()
    assert sum(record.getMessage() == state_averager._AR_TIMEOUT_MESSAGE for record in parent.records) == 1
    parent.killpg.assert_not_called()
    parent.exit_process.assert_not_called()


def test_failed_averager_rejects_result_application_and_further_work(parent):
    with pytest.raises(state_averager._UnfinishedAllreduceError):
        run_round(parent, AveragingControl())
    parent._load_local_tensors_into_averager_.reset_mock()
    with parent.lock_averaged_tensors:
        with pytest.raises(state_averager._UnfinishedAllreduceError):
            TrainingStateAverager.step(parent, apply_delayed_updates=True, increment_epoch=True)
        next_control = AveragingControl()
        with pytest.raises(state_averager._UnfinishedAllreduceError):
            run_round(parent, next_control)
    assert not next_control.triggered
    assert parent.local_epoch == 7026
    parent._apply_averaging_results_.assert_not_called()
    parent._load_local_tensors_into_averager_.assert_not_called()


def test_failure_while_waiting_for_updates_cannot_fall_through_to_apply(parent):
    class BackgroundUpdate(Future):
        def result(self, timeout=None):
            parent._averaging_failed = True
            self.set_exception(state_averager._UnfinishedAllreduceError(state_averager._AR_TIMEOUT_MESSAGE))
            return super().result(timeout)

    parent.pending_updates.add(BackgroundUpdate())
    parent._allreduce_timeout = 0.01
    with pytest.raises(state_averager._UnfinishedAllreduceError):
        TrainingStateAverager.step(parent, wait_for_delayed_updates=True)
    parent._apply_averaging_results_.assert_not_called()
    parent._apply_optimizer_parameters_.assert_not_called()


def test_other_parent_error_preserves_existing_cancellation(parent):
    control = AveragingControl()
    control.result = Mock(side_effect=RuntimeError("failed to wait on control"))
    run_round(parent, control)
    assert control.cancelled()
    assert parent.finished_averaging_round.is_set()
    assert not parent._averaging_failed
    parent.killpg.assert_not_called()
    parent.exit_process.assert_not_called()


def test_actual_fatal_log_triggers_configured_monitor_termination(parent, monitor):
    with pytest.raises(state_averager._UnfinishedAllreduceError):
        run_round(parent, AveragingControl())
    parent.killpg.assert_not_called()
    for record in parent.records:
        monitor.queue_handler.handle(record)
    monitor.start()
    monitor.join(timeout=2)
    assert not monitor.is_alive(), "The monitor did not recognize the averager's fatal log"
    parent.killpg.assert_called_once_with(12345, signal.SIGTERM)
    parent.exit_process.assert_not_called()


@pytest.mark.parametrize("failures, should_terminate", [(1, False), (2, True)])
def test_ordinary_timeout_still_uses_existing_failure_threshold(parent, monitor, failures, should_terminate):
    for _ in range(failures):
        monitor.queue_handler.handle(
            logging.makeLogRecord(
                {
                    "msg": "Averaging parameters failed with <class 'TimeoutError'>",
                }
            )
        )
    # Stop after processing the queued messages, without depending on thread timing.
    original_get = monitor._log_queue.get
    remaining = failures

    def get_and_stop(**kwargs):
        nonlocal remaining
        record = original_get(**kwargs)
        remaining -= 1
        if remaining == 0:
            monitor._stop_event.set()
        return record

    monitor._log_queue.get = get_and_stop
    monitor.run()
    assert parent.killpg.called == should_terminate
    parent.exit_process.assert_not_called()
