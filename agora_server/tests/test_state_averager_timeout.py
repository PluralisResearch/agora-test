"""Contain unfinished ARs without publishing tensors still owned by the child."""

import logging
import signal
import subprocess
import sys
import threading

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agora_server.core.averaging import state_averager
from agora_server.core.averaging.state_averager import TrainingStateAverager


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
            # Complete only once the parent has entered its grace-period wait.
            self.complete(self.after_timeout)
        return super().result(timeout=timeout)


@pytest.fixture
def parent(monkeypatch):
    monkeypatch.setattr(state_averager, "_AR_TIMEOUT_GRACE", 0.01)
    # Never signal or exit the test runner itself.
    killpg = Mock()
    exit_process = Mock(side_effect=SystemExit("Unfinished all-reduce"))
    monkeypatch.setattr(state_averager.os, "killpg", killpg)
    monkeypatch.setattr(state_averager.os, "getpgrp", lambda: 12345)
    monkeypatch.setattr(state_averager.os, "_exit", exit_process)
    parent = SimpleNamespace(
        lock_optimizer=threading.Lock(),
        lock_averaging=threading.Lock(),
        lock_averaged_tensors=threading.Lock(),
        reuse_tensors=False,
        _update_scheduler=lambda: None,
        _load_local_tensors_into_averager_=lambda: None,
        _save_pre_averaging_state=lambda: None,
        finished_optimizer_step=threading.Event(),
        finished_averaging_round=threading.Event(),
        delay_before_averaging=SimpleNamespace(update=lambda **kwargs: None),
        local_epoch=7026,
        status_loglevel=logging.INFO,
        sync_epoch_when_averaging=True,
        killpg=killpg,
        exit_process=exit_process,
    )
    parent._wait_for_averaging = lambda control, timeout: TrainingStateAverager._wait_for_averaging(
        parent, control, timeout
    )
    return parent


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
    parent.killpg.assert_not_called()
    parent.exit_process.assert_not_called()


def test_pending_child_terminates_without_publishing_locked_tensors(parent):
    control = AveragingControl()
    with parent.lock_averaged_tensors:
        with pytest.raises(SystemExit, match="Unfinished all-reduce"):
            run_round(parent, control)
    parent.killpg.assert_called_once_with(12345, signal.SIGTERM)
    parent.exit_process.assert_called_once_with(1)
    assert control.waits == [0, 0.01]
    assert not control.done()  # Cancelling this future would hide unfinished work.
    assert not parent.finished_averaging_round.is_set()


def test_signal_error_cannot_fall_through_to_finished_round(parent):
    parent.killpg.side_effect = ProcessLookupError("process group disappeared")
    with pytest.raises(SystemExit, match="Unfinished all-reduce"):
        run_round(parent, AveragingControl())
    parent.exit_process.assert_called_once_with(1)
    assert not parent.finished_averaging_round.is_set()


def test_other_parent_error_preserves_existing_cancellation(parent):
    control = AveragingControl()
    control.result = Mock(side_effect=RuntimeError("failed to wait on control"))
    run_round(parent, control)
    assert control.cancelled()
    assert parent.finished_averaging_round.is_set()
    parent.killpg.assert_not_called()
    parent.exit_process.assert_not_called()


@pytest.mark.skipif(sys.platform == "win32", reason="Worker termination uses POSIX process groups")
@pytest.mark.parametrize("ignore_sigterm", [False, True])
def test_pending_child_really_terminates_isolated_process_group(ignore_sigterm):
    # A new session makes this subprocess its own group; the test runner is outside it.
    script = """
import signal
import sys
from concurrent.futures import Future
from agora_server.core.averaging import state_averager
if sys.argv[1] == "True":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
state_averager._AR_TIMEOUT_GRACE = 0.01
state_averager.TrainingStateAverager._wait_for_averaging(None, Future(), timeout=0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ignore_sigterm)],
        start_new_session=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (1 if ignore_sigterm else -signal.SIGTERM), result.stderr
    assert "terminating worker process group" in result.stderr
