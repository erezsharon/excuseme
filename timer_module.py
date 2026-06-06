"""
timer_module.py
===============

Session timing for NeuralFM. Three independent modes:

* ``INFINITE``  -- counts up forever (no end).
* ``TIMER``     -- counts down from a user-set number of minutes.
* ``POMODORO``  -- alternates 25 min work / 5 min break cycles.

The controller is *tick driven*: the UI calls :meth:`SessionTimer.tick` once a
second (from a tkinter ``after`` loop). The controller never touches the clock
itself, which keeps it trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TimerMode(Enum):
    INFINITE = "Infinite"
    TIMER = "Timer"
    POMODORO = "Pomodoro"


class Phase(Enum):
    WORK = "Work"
    BREAK = "Break"
    DONE = "Done"
    RUNNING = "Running"


POMODORO_WORK_MIN = 25
POMODORO_BREAK_MIN = 5


@dataclass
class TimerState:
    mode: TimerMode
    phase: Phase
    display: str          # "MM:SS"
    running: bool
    cycle: int            # completed pomodoro work cycles
    fraction: float       # 0..1 progress through current phase (0 for infinite)


def _fmt(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class SessionTimer:
    """Drive with :meth:`tick`; read state from the returned :class:`TimerState`.

    Optional callbacks (set as attributes):
      * ``on_phase_change(phase)`` -- pomodoro work<->break or timer finished.
      * ``on_finish()``            -- a TIMER countdown hit zero.
    """

    def __init__(self) -> None:
        self.mode = TimerMode.INFINITE
        self.running = False
        self._elapsed = 0.0          # seconds elapsed in current phase
        self._duration = 0.0         # target seconds for TIMER / pomodoro phase
        self._timer_minutes = 25     # configured TIMER length
        self.phase = Phase.RUNNING
        self.cycle = 0
        self.on_phase_change = None
        self.on_finish = None

    # -- configuration -----------------------------------------------------
    def set_mode(self, mode: TimerMode, timer_minutes: int | None = None) -> None:
        self.mode = mode
        if timer_minutes is not None:
            self._timer_minutes = max(1, int(timer_minutes))
        self.reset()

    def set_timer_minutes(self, minutes: int) -> None:
        self._timer_minutes = max(1, int(minutes))
        if self.mode is TimerMode.TIMER:
            self.reset()

    # -- transport ---------------------------------------------------------
    def start(self) -> None:
        self.running = True

    def pause(self) -> None:
        self.running = False

    def toggle(self) -> None:
        self.running = not self.running

    def reset(self) -> None:
        self.running = False
        self._elapsed = 0.0
        self.cycle = 0
        if self.mode is TimerMode.INFINITE:
            self.phase = Phase.RUNNING
            self._duration = 0.0
        elif self.mode is TimerMode.TIMER:
            self.phase = Phase.RUNNING
            self._duration = self._timer_minutes * 60
        else:  # POMODORO
            self.phase = Phase.WORK
            self._duration = POMODORO_WORK_MIN * 60

    # -- tick --------------------------------------------------------------
    def tick(self, dt: float = 1.0) -> TimerState:
        if self.running:
            self._elapsed += dt
            self._maybe_advance()
        return self.state()

    def _maybe_advance(self) -> None:
        if self.mode is TimerMode.INFINITE:
            return

        if self._elapsed < self._duration:
            return

        if self.mode is TimerMode.TIMER:
            self.running = False
            self.phase = Phase.DONE
            self._elapsed = self._duration
            self._emit(self.on_finish)
            self._emit(self.on_phase_change, self.phase)
            return

        # POMODORO: flip between work and break, looping forever.
        if self.phase is Phase.WORK:
            self.cycle += 1
            self.phase = Phase.BREAK
            self._duration = POMODORO_BREAK_MIN * 60
        else:
            self.phase = Phase.WORK
            self._duration = POMODORO_WORK_MIN * 60
        self._elapsed = 0.0
        self._emit(self.on_phase_change, self.phase)

    # -- state -------------------------------------------------------------
    def state(self) -> TimerState:
        if self.mode is TimerMode.INFINITE:
            display = _fmt(self._elapsed)
            fraction = 0.0
        else:
            remaining = max(0.0, self._duration - self._elapsed)
            display = _fmt(remaining)
            fraction = (self._elapsed / self._duration) if self._duration else 0.0
        return TimerState(
            mode=self.mode,
            phase=self.phase,
            display=display,
            running=self.running,
            cycle=self.cycle,
            fraction=min(1.0, fraction),
        )

    @staticmethod
    def _emit(cb, *args) -> None:
        if cb is not None:
            try:
                cb(*args)
            except Exception:
                pass
