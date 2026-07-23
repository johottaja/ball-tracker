from __future__ import annotations

from .types import BallSample, BounceInterval


def estimate_bounce_subframe_index(
    samples: list[BallSample],
    bounce: BounceInterval,
) -> float | None:
    """
    Estimate the capture time where the ball reversed vertical direction.

    Uses linear velocity zero-crossing between the bounce frame pair; falls back
    to a quadratic fit around the lowest (maximum y) sample when needed.
    """
    by_frame = {sample.frame_index: sample for sample in samples}
    prev_sample = by_frame.get(bounce.frame_prev)
    curr_sample = by_frame.get(bounce.frame_curr)
    if prev_sample is None or curr_sample is None:
        return None

    before_prev = _sample_before(samples, bounce.frame_prev)
    if before_prev is None:
        return None

    vy_prev = prev_sample.bottom_y - before_prev.bottom_y
    vy_curr = curr_sample.bottom_y - prev_sample.bottom_y
    if vy_prev == 0 and vy_curr == 0:
        return _quadratic_vertex_time(samples, bounce)
    if vy_prev == vy_curr:
        return prev_sample.capture_time_s

    fraction = vy_prev / (vy_prev - vy_curr)
    fraction = max(0.0, min(1.0, fraction))
    return prev_sample.capture_time_s + fraction * (
        curr_sample.capture_time_s - prev_sample.capture_time_s
    )


def _sample_before(samples: list[BallSample], frame_index: int) -> BallSample | None:
    prior = [sample for sample in samples if sample.frame_index < frame_index]
    if not prior:
        return None
    return prior[-1]


def _quadratic_vertex_time(
    samples: list[BallSample],
    bounce: BounceInterval,
) -> float | None:
    window = [
        sample
        for sample in samples
        if bounce.frame_prev - 2 <= sample.frame_index <= bounce.frame_curr + 2
    ]
    if len(window) < 3:
        return next(
            sample.capture_time_s
            for sample in samples
            if sample.frame_index == bounce.frame_prev
        )

    peak = max(window, key=lambda sample: sample.bottom_y)
    neighbors = sorted(
        [
            sample
            for sample in window
            if abs(sample.frame_index - peak.frame_index) <= 2
        ],
        key=lambda sample: sample.frame_index,
    )
    if len(neighbors) < 3:
        return peak.capture_time_s

    xs = [sample.capture_time_s for sample in neighbors]
    ys = [float(sample.bottom_y) for sample in neighbors]
    # Fit y = a*t^2 + b*t + c; vertex at t = -b / (2a).
    n = len(xs)
    sum_x = sum(xs)
    sum_x2 = sum(x * x for x in xs)
    sum_x3 = sum(x * x * x for x in xs)
    sum_x4 = sum(x * x * x * x for x in xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2y = sum(x * x * y for x, y in zip(xs, ys))

    det = (
        n * (sum_x2 * sum_x4 - sum_x3 * sum_x3)
        - sum_x * (sum_x * sum_x4 - sum_x3 * sum_x2)
        + sum_x2 * (sum_x * sum_x3 - sum_x2 * sum_x2)
    )
    if abs(det) < 1e-9:
        return float(peak.frame_index)

    a = (
        n * (sum_x2 * sum_x2y - sum_x3 * sum_xy)
        - sum_x * (sum_x * sum_x2y - sum_x3 * sum_y)
        + sum_y * (sum_x * sum_x3 - sum_x2 * sum_x2)
    ) / det
    b = (
        n * (sum_x3 * sum_xy - sum_x4 * sum_y)
        - sum_x2 * (sum_x2 * sum_xy - sum_x4 * sum_y)
        + sum_x2y * (sum_x * sum_x3 - sum_x2 * sum_x2)
    ) / det

    if abs(a) < 1e-9:
        return float(peak.frame_index)

    return -b / (2.0 * a)
