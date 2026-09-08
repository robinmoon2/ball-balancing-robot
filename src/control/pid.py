"""Pure PID controller. No I/O, no time.time(). dt is passed explicitly.

PID itself now lives in utils.py with the project's other shared
dataclasses; this module re-exports it so existing imports
(`from .pid import PID`, `from src.control.pid import PID`) keep working.
"""

from __future__ import annotations

from utils import PID

__all__ = ["PID"]
