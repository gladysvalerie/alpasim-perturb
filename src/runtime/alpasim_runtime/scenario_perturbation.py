# SPDX-License-Identifier: Apache-2.0
"""Scenario perturbation: offset non-ego actor tracks to create scene variants.

Local addition to AlpaSim (not upstream).

Driven by ``simulation_config.perturbation`` so it is settable from the wizard
command line via Hydra dotlist overrides, e.g.

    uv run alpasim_wizard <...> \\
        runtime.simulation_config.perturbation.actor_id=17 \\
        runtime.simulation_config.perturbation.lateral_offset_m=3.0

Defaults are inert: actor_id="" means no-op, so an unconfigured run behaves
exactly like stock AlpaSim.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
from alpasim_utils.geometry import Pose, Trajectory

logger = logging.getLogger(__name__)

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


# --------------------------------------------------------------------------
# perturbation
# --------------------------------------------------------------------------


def _time_shift(traj: Trajectory, shift_us: int) -> Trajectory:
    """Resample so the actor sits where it would have been ``shift_us`` later.

    Timestamps are preserved; only poses move. Queries past either end are
    clamped, so the actor holds its first/last pose instead of vanishing --
    skip-mode replay drops any actor whose trajectory does not cover the
    queried timestamp, which would otherwise look like a rendering bug.
    """
    ts = traj.timestamps_us.astype(np.int64)
    if len(ts) < 2:
        return traj
    query = np.clip(ts + shift_us, ts[0], ts[-1]).astype(np.uint64)
    resampled = traj.interpolate(query)
    return Trajectory(ts.astype(np.uint64), resampled.positions, resampled.quaternions)


def _spatial_offset(
    traj: Trajectory,
    lateral_m: float,
    longitudinal_m: float,
    vertical_m: float = 0.0,
) -> Trajectory:
    """Offset in the actor's own frame at every sample (x fwd, y left, z up).

    Same mechanism as the NRE hidden-actor hack in unbound_rollout.py, which
    sinks dropped actors 100 m with transform(..., is_relative=True).
    """
    offset = Pose(
        np.array([longitudinal_m, lateral_m, vertical_m], dtype=np.float32),
        _IDENTITY_QUAT,
    )
    return traj.transform(offset, is_relative=True)


def perturb_traffic_objects(traffic_objects, cfg, scene_id: str):
    """Return a new TrafficObjects with ``cfg`` applied. No-op when unset."""
    if cfg is None or not cfg.actor_id:
        return traffic_objects

    target = str(cfg.actor_id)
    targets = {t.strip() for t in target.split(",") if t.strip()}
    shift_us = int(float(cfg.time_shift_s) * 1e6)
    lateral_m = float(cfg.lateral_offset_m)
    longitudinal_m = float(cfg.longitudinal_offset_m)
    vertical_m = float(getattr(cfg, "vertical_offset_m", 0.0))

    if shift_us == 0 and lateral_m == 0.0 and longitudinal_m == 0.0 \
            and vertical_m == 0.0:
        logger.warning("[PERTURB] actor_id=%s set but all offsets zero; no-op", target)
        return traffic_objects

    out: dict = {}
    n_hit = 0
    for obj_id, obj in traffic_objects.items():
        matched = (
            obj_id in targets
            or (target == "*" and not obj.is_static)
            or target == "**"
        )
        if not matched:
            out[obj_id] = obj
            continue

        traj = obj.trajectory
        before = traj.positions[0].copy()
        if shift_us:
            traj = _time_shift(traj, shift_us)
        if lateral_m or longitudinal_m or vertical_m:
            traj = _spatial_offset(traj, lateral_m, longitudinal_m, vertical_m)

        out[obj_id] = replace(obj, trajectory=traj)
        n_hit += 1
        if target in ("*", "**") and n_hit > 5:
            continue
        logger.warning(
            "[PERTURB] scene=%s actor=%s class=%s static=%s "
            "dt=%+.2fs lat=%+.2fm lon=%+.2fm vert=%+.2fm start %s -> %s",
            scene_id, obj_id, obj.label_class, obj.is_static,
            shift_us / 1e6, lateral_m, longitudinal_m,
            np.round(before, 2), np.round(traj.positions[0], 2),
        )

    if n_hit == 0:
        raise ValueError(
            f"perturbation.actor_id={target!r} matched no actor in scene {scene_id}. "
            f"Present: {sorted(traffic_objects.keys())}"
        )

    return type(traffic_objects)(**out)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def _yaw_from_quat(q: np.ndarray) -> np.ndarray:
    """Yaw about z from xyzw quaternions, shape [N, 4] -> [N]."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _ego_frame(actor_xy: np.ndarray, ego_xy: np.ndarray, ego_yaw: np.ndarray):
    """World -> ego frame. Returns (longitudinal, lateral); +lat is ego's left."""
    d = actor_xy - ego_xy
    c, s = np.cos(ego_yaw), np.sin(ego_yaw)
    return c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]


def log_lead_vehicle(
    traffic_objects,
    gt_ego_trajectory,
    scene_id: str,
    tag: str = "",
    max_range_m: float = 60.0,
    half_width_m: float = 1.75,
    stride: int = 10,
) -> None:
    """Log the nearest actor in a forward corridor ahead of the ego, per step.

    Defines "in front" geometrically: longitudinal in (0, max_range_m] and
    |lateral| <= half_width_m in the ego frame -- i.e. roughly the ego's own
    lane. Call this before and after perturbation to see, objectively, whether
    the perturbation removed or introduced a lead vehicle.

    Uses the GT ego trajectory, so it describes the scene as authored. That is
    exact for force-GT runs and approximate once a closed-loop ego has drifted
    from GT.
    """
    ego_ts = gt_ego_trajectory.timestamps_us.astype(np.int64)
    ego_xy = gt_ego_trajectory.positions[:, :2]
    ego_yaw = _yaw_from_quat(gt_ego_trajectory.quaternions)
    n = len(ego_ts)

    best_lon = np.full(n, np.inf)
    best_id = np.array([""] * n, dtype=object)
    best_lat = np.full(n, np.nan)
    best_cls = np.array([""] * n, dtype=object)

    for obj_id, obj in traffic_objects.items():
        a_ts = obj.trajectory.timestamps_us.astype(np.int64)
        if len(a_ts) < 2:
            continue
        alive = (ego_ts >= a_ts[0]) & (ego_ts <= a_ts[-1])
        if not alive.any():
            continue

        idx = np.flatnonzero(alive)
        pos = obj.trajectory.interpolate(ego_ts[idx].astype(np.uint64)).positions
        lon, lat = _ego_frame(pos[:, :2], ego_xy[idx], ego_yaw[idx])

        hit = (lon > 0.0) & (lon <= max_range_m) & (np.abs(lat) <= half_width_m)
        for k in np.flatnonzero(hit):
            j = idx[k]
            if lon[k] < best_lon[j]:
                best_lon[j] = lon[k]
                best_lat[j] = lat[k]
                best_id[j] = obj_id
                best_cls[j] = obj.label_class

    label = f"LEAD{'/' + tag if tag else ''}"
    n_with = int(np.isfinite(best_lon).sum())
    logger.warning(
        "%s scene=%s corridor=+/-%.2fm range=%.0fm steps_with_lead=%d/%d",
        label, scene_id, half_width_m, max_range_m, n_with, n,
    )

    t0 = ego_ts[0]
    for j in range(0, n, stride):
        t_s = (ego_ts[j] - t0) / 1e6
        if np.isfinite(best_lon[j]):
            logger.warning(
                "%s t=%6.2fs  lead=%-6s %-12s  lon=%6.2fm lat=%+5.2fm",
                label, t_s, best_id[j], best_cls[j], best_lon[j], best_lat[j],
            )
        else:
            logger.warning("%s t=%6.2fs  lead=NONE", label, t_s)


def log_actor_inventory(traffic_objects, gt_ego_trajectory, scene_id: str) -> None:
    """Print every actor with its closest approach to the ego GT path.

    Note this is path-to-path ignoring time, not same-time clearance: a small
    value means the actor drove the same strip of road as the ego at some
    point, not that it nearly hit it.
    """
    ego_xy = gt_ego_trajectory.positions[:, :2]
    rows = []
    for obj_id, obj in traffic_objects.items():
        pos = obj.trajectory.positions
        if len(pos) == 0:
            continue
        d = np.linalg.norm(pos[:, None, :2] - ego_xy[None, :, :2], axis=-1)
        i_actor, _ = np.unravel_index(np.argmin(d), d.shape)
        rows.append((
            float(d.min()), obj_id, obj.label_class, obj.is_static, len(pos),
            int(obj.trajectory.timestamps_us[i_actor]),
        ))

    logger.warning("[ACTORS] scene=%s n=%d", scene_id, len(rows))
    for dist, obj_id, label, static, n, t_us in sorted(rows):
        logger.warning(
            "[ACTORS] id=%-14s class=%-14s static=%-5s n=%-4d "
            "min_dist=%7.2fm at t=%6.2fs",
            obj_id, label, static, n, dist, t_us / 1e6,
        )