"""Wall-clock watchdog around ``scipy.optimize.milp`` (HiGHS).

HiGHS normally honours ``time_limit``, but the original full comparison saw
three solves that never returned (hours of CPU, no further solves).  When a
hard timeout is configured, the solve runs in a forked child; if it exceeds
the hard limit the child is killed and a *no-incumbent* result is returned,
which the existing MPC path already turns into the hold-previous fallback.

Off by default.  Enabled only through the environment variable
``MPC_SOLVER_HARD_TIMEOUT_SECONDS`` (seconds, > 0) so frozen case configs and
their MPC fingerprints stay unchanged.  A solve that finishes before the hard
limit returns exactly what ``milp`` returned.
"""

from __future__ import annotations

import multiprocessing
import os
import time

import numpy as np
from scipy.optimize import OptimizeResult, milp

ENV_VARIABLE = 'MPC_SOLVER_HARD_TIMEOUT_SECONDS'
HARD_TIMEOUT_STATUS = 99


def hard_timeout_seconds() -> float | None:
    value = os.environ.get(ENV_VARIABLE, '').strip()
    if not value:
        return None
    seconds = float(value)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f'{ENV_VARIABLE} must be a positive number of seconds')
    return seconds


def _child(connection, args, kwargs):
    try:
        connection.send(('ok', milp(*args, **kwargs)))
    except BaseException as error:  # pragma: no cover - reported to the parent
        connection.send(('error', repr(error)))
    finally:
        connection.close()


def _failure(message: str) -> OptimizeResult:
    return OptimizeResult(x=None, fun=float('nan'), success=False,
                          status=HARD_TIMEOUT_STATUS, message=message, mip_gap=None)


def guarded_milp(*args, timeout_seconds: float | None = None, **kwargs) -> OptimizeResult:
    """``milp`` with an optional hard wall-clock limit (see module docstring)."""
    limit = hard_timeout_seconds() if timeout_seconds is None else float(timeout_seconds)
    if not limit:
        return milp(*args, **kwargs)
    context = multiprocessing.get_context('fork')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(sender, args, kwargs), daemon=True)
    started = time.perf_counter()
    process.start()
    sender.close()
    try:
        if receiver.poll(limit):
            kind, payload = receiver.recv()
            process.join(5)
            if kind == 'ok':
                return payload
            raise RuntimeError(f'milp failed in watchdog child: {payload}')
        return _failure(
            f'hard_timeout_killed_after_{time.perf_counter()-started:.1f}s '
            f'(limit {limit:.0f}s); treated as no incumbent'
        )
    except EOFError:
        return _failure('milp watchdog child exited without a result; treated as no incumbent')
    finally:
        receiver.close()
        if process.is_alive():
            process.kill()
        process.join(5)
