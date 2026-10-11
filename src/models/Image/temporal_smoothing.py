from collections import deque
from typing import Deque, Optional
import time

import numpy as np


class MajorityVoteSmoother:
    """
    Use this when the model output is already a hard label:
    0 = open, 1 = closed.
    """

    def __init__(self, window_size: int = 5, closed_votes_required: int = 3):
        self.window_size = window_size
        self.closed_votes_required = closed_votes_required
        self.buffer: Deque[int] = deque(maxlen=window_size)

    def update(self, label: int) -> int:
        self.buffer.append(int(label))
        closed_votes = sum(self.buffer)

        return 1 if closed_votes >= self.closed_votes_required else 0

    def reset(self) -> None:
        self.buffer.clear()


class MovingAverageSmoother:
    """
    Use this when the model output is a probability of closed eye.
    Example: CNN sigmoid output.
    """

    def __init__(self, window_size: int = 5, threshold: float = 0.5):
        self.window_size = window_size
        self.threshold = threshold
        self.buffer: Deque[float] = deque(maxlen=window_size)

    def update(self, closed_probability: float) -> int:
        self.buffer.append(float(closed_probability))

        avg_prob = self.get_average_probability()

        return 1 if avg_prob > self.threshold else 0

    def get_average_probability(self) -> float:
        if not self.buffer:
            return 0.0

        return float(np.mean(self.buffer))

    def reset(self) -> None:
        self.buffer.clear()


class HysteresisMovingAverageSmoother:
    """
    Moving average smoother with two thresholds to reduce flickering.

    State:
        0 = open
        1 = closed

    Logic:
        - If current state is OPEN, switch to CLOSED only when avg_prob >= close_threshold.
        - If current state is CLOSED, switch back to OPEN only when avg_prob <= open_threshold.
    """

    def __init__(
        self,
        window_size: int = 7,
        close_threshold: float = 0.55,
        open_threshold: float = 0.35,
    ):
        if open_threshold >= close_threshold:
            raise ValueError("open_threshold must be smaller than close_threshold.")

        self.window_size = window_size
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.buffer: Deque[float] = deque(maxlen=window_size)
        self.current_state = 0

    def update(self, closed_probability: float) -> int:
        self.buffer.append(float(closed_probability))

        avg_prob = self.get_average_probability()

        # Currently OPEN -> only switch to CLOSED when probability is high enough
        if self.current_state == 0 and avg_prob >= self.close_threshold:
            self.current_state = 1

        # Currently CLOSED -> only switch back to OPEN when probability is low enough
        elif self.current_state == 1 and avg_prob <= self.open_threshold:
            self.current_state = 0

        return self.current_state

    def get_average_probability(self) -> float:
        if not self.buffer:
            return 0.0

        return float(np.mean(self.buffer))

    def reset(self) -> None:
        self.buffer.clear()
        self.current_state = 0


class ConsecutiveClosedAlert:
    """
    Trigger alert only when the smoothed state is closed for N consecutive frames.
    Example: at 15 FPS, 15 consecutive closed frames ≈ 1 second.

    This class is still useful if you want frame-based alerting.
    For webcam with variable FPS, TimeBasedClosedAlert is recommended.
    """

    def __init__(self, frames_required: int = 15):
        self.frames_required = frames_required
        self.closed_count = 0

    def update(self, smoothed_label: int) -> bool:
        if int(smoothed_label) == 1:
            self.closed_count += 1
        else:
            self.closed_count = 0

        return self.closed_count >= self.frames_required

    def reset(self) -> None:
        self.closed_count = 0


class TimeBasedClosedAlert:
    """
    Trigger alert when the smoothed state is closed for a real duration.
    This is more stable than counting frames when FPS changes.
    """

    def __init__(self, seconds_required: float = 1.0):
        self.seconds_required = seconds_required
        self.closed_start_time: Optional[float] = None
        self.closed_duration = 0.0

    def update(self, smoothed_label: int) -> bool:
        now = time.time()

        if int(smoothed_label) == 1:
            if self.closed_start_time is None:
                self.closed_start_time = now

            self.closed_duration = now - self.closed_start_time

            return self.closed_duration >= self.seconds_required

        self.closed_start_time = None
        self.closed_duration = 0.0

        return False

    def reset(self) -> None:
        self.closed_start_time = None
        self.closed_duration = 0.0