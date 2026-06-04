from __future__ import annotations

import math

import numpy as np


def _scan_bounds(
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    """Return valid top-left coordinate bounds for probe placement."""
    obj_h, obj_w = object_shape
    probe_h, probe_w = probe_shape

    y_min = margin
    x_min = margin
    y_max = obj_h - probe_h - margin
    x_max = obj_w - probe_w - margin

    if y_max < y_min or x_max < x_min:
        raise ValueError("Probe shape is too large for the object shape and margin.")

    return y_min, y_max, x_min, x_max


def _center_position(
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    margin: int,
) -> tuple[int, int]:
    """Return the valid top-left position nearest to the object center."""
    y_min, y_max, x_min, x_max = _scan_bounds(
        object_shape=object_shape,
        probe_shape=probe_shape,
        margin=margin,
    )

    obj_h, obj_w = object_shape
    probe_h, probe_w = probe_shape

    cy = int(round((obj_h - probe_h) / 2))
    cx = int(round((obj_w - probe_w) / 2))

    cy = int(np.clip(cy, y_min, y_max))
    cx = int(np.clip(cx, x_min, x_max))

    return cy, cx


def _apply_integer_jitter(
    y: int,
    x: int,
    *,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    jitter: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Apply integer Gaussian jitter and clip to valid bounds."""
    if jitter > 0:
        dy = int(np.rint(rng.normal(0.0, jitter)))
        dx = int(np.rint(rng.normal(0.0, jitter)))
    else:
        dy = 0
        dx = 0

    yy = int(np.clip(y + dy, y_min, y_max))
    xx = int(np.clip(x + dx, x_min, x_max))

    return yy, xx


def _append_unique_position(
    positions: list[tuple[int, int]],
    seen: set[tuple[int, int]],
    position: tuple[int, int],
) -> None:
    """Append a position only if it has not appeared before."""
    if position in seen:
        return

    positions.append(position)
    seen.add(position)


def make_raster_scan_positions(
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    step: int = 16,
    margin: int = 0,
    jitter: int = 0,
    rng: np.random.Generator | None = None,
    n_scans: int | None = None,
) -> np.ndarray:
    """Generate center-out Manhattan spiral raster scan positions."""
    if step <= 0:
        raise ValueError("step must be positive.")

    if n_scans is not None and n_scans <= 0:
        raise ValueError("n_scans must be positive or None.")

    rng = np.random.default_rng() if rng is None else rng

    y_min, y_max, x_min, x_max = _scan_bounds(
        object_shape=object_shape,
        probe_shape=probe_shape,
        margin=margin,
    )
    cy, cx = _center_position(
        object_shape=object_shape,
        probe_shape=probe_shape,
        margin=margin,
    )

    positions: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add_position(y: int, x: int) -> None:
        yy, xx = _apply_integer_jitter(
            y,
            x,
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            jitter=jitter,
            rng=rng,
        )
        _append_unique_position(positions, seen, (yy, xx))

    add_position(cy, cx)

    if n_scans is not None and len(positions) >= n_scans:
        return np.asarray(positions[:n_scans], dtype=np.int64)

    max_radius_y = max(abs(cy - y_min), abs(y_max - cy))
    max_radius_x = max(abs(cx - x_min), abs(x_max - cx))
    max_ring = int(math.ceil(max(max_radius_y, max_radius_x) / step))

    for ring in range(1, max_ring + 1):
        y = cy - ring * step
        x = cx - ring * step
        side_len = 2 * ring

        # Move right along the top edge.
        for _ in range(side_len):
            add_position(y, x)
            x += step

            if n_scans is not None and len(positions) >= n_scans:
                return np.asarray(positions[:n_scans], dtype=np.int64)

        # Move down along the right edge.
        for _ in range(side_len):
            add_position(y, x)
            y += step

            if n_scans is not None and len(positions) >= n_scans:
                return np.asarray(positions[:n_scans], dtype=np.int64)

        # Move left along the bottom edge.
        for _ in range(side_len):
            add_position(y, x)
            x -= step

            if n_scans is not None and len(positions) >= n_scans:
                return np.asarray(positions[:n_scans], dtype=np.int64)

        # Move up along the left edge.
        for _ in range(side_len):
            add_position(y, x)
            y -= step

            if n_scans is not None and len(positions) >= n_scans:
                return np.asarray(positions[:n_scans], dtype=np.int64)

    return np.asarray(positions, dtype=np.int64)


def make_fermat_scan_positions(
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    step: float = 16.0,
    margin: int = 0,
    jitter: int = 0,
    rng: np.random.Generator | None = None,
    n_scans: int | None = None,
    golden_angle_rad: float = math.pi * (3.0 - math.sqrt(5.0)),
    max_candidates: int | None = None,
) -> np.ndarray:
    """Generate center-out Fermat spiral scan positions."""
    if step <= 0:
        raise ValueError("step must be positive.")

    if n_scans is not None and n_scans <= 0:
        raise ValueError("n_scans must be positive or None.")

    rng = np.random.default_rng() if rng is None else rng

    y_min, y_max, x_min, x_max = _scan_bounds(
        object_shape=object_shape,
        probe_shape=probe_shape,
        margin=margin,
    )
    cy, cx = _center_position(
        object_shape=object_shape,
        probe_shape=probe_shape,
        margin=margin,
    )

    max_radius_y = max(abs(cy - y_min), abs(y_max - cy))
    max_radius_x = max(abs(cx - x_min), abs(x_max - cx))
    max_radius = math.sqrt(max_radius_y**2 + max_radius_x**2)

    if max_candidates is None:
        # Oversample candidates to compensate for clipping, rounding, and duplicates.
        if n_scans is None:
            max_candidates = int(math.ceil((max_radius / step) ** 2 * 8.0)) + 1024
        else:
            max_candidates = max(1024, int(n_scans * 20))

    positions: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for i in range(max_candidates):
        radius = step * math.sqrt(i)
        theta = i * golden_angle_rad

        y = int(round(cy + radius * math.sin(theta)))
        x = int(round(cx + radius * math.cos(theta)))

        if y < y_min or y > y_max or x < x_min or x > x_max:
            continue

        yy, xx = _apply_integer_jitter(
            y,
            x,
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            jitter=jitter,
            rng=rng,
        )

        _append_unique_position(positions, seen, (yy, xx))

        if n_scans is not None and len(positions) >= n_scans:
            return np.asarray(positions[:n_scans], dtype=np.int64)

    if n_scans is not None and len(positions) < n_scans:
        raise RuntimeError(
            f"Could only generate {len(positions)} unique Fermat positions, "
            f"but n_scans={n_scans} was requested. "
            "Try increasing max_candidates, reducing jitter, reducing margin, "
            "or using a smaller step."
        )

    return np.asarray(positions, dtype=np.int64)