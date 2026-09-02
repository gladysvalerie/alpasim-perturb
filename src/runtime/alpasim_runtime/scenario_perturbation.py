# SPDX-License-Identifier: Apache-2.0
"""Scenario perturbation: offset non-ego actor tracks to create scene variants.

Local addition to AlpaSim (not upstream).

Driven by ``simulation_config.perturbation`` so it is settable from the wizard
command line via Hydra dotlist overrides, e.g.

    alpasim-wizard <...> \\
        runtime.simulation_config.perturbation.actor_id=1234 \\
        runtime.simulation_config.perturbation.time_shift_s=-2.0

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
    traj: Trajectory, lateral_m: float, longitudinal_m: float
) -> Trajectory:
    """Offset in the actor's own frame at every sample (x fwd, y left, z up).

    Same mechanism as the NRE hidden-actor hack in unbound_rollout.py, which
    sinks dropped actors 100 m with transform(..., is_relative=True).
    """
    offset = Pose(
        np.array([longitudinal_m, lateral_m, 0.0], dtype=np.float32), _IDENTITY_QUAT
    )
    return traj.transform(offset, is_relative=True)


def perturb_traffic_objects(traffic_objects, cfg, scene_id: str):
    """Return a new TrafficObjects with ``cfg`` applied. No-op when unset."""
    if cfg is None or not cfg.actor_id:
        return traffic_objects

    target = str(cfg.actor_id)
    shift_us = int(float(cfg.time_shift_s) * 1e6)
    lateral_m = float(cfg.lateral_offset_m)
    longitudinal_m = float(cfg.longitudinal_offset_m)

    if shift_us == 0 and lateral_m == 0.0 and longitudinal_m == 0.0:
        logger.warning("[PERTURB] actor_id=%s set but all offsets zero; no-op", target)
        return traffic_objects

    out: dict = {}
    n_hit = 0
    for obj_id, obj in traffic_objects.items():
        matched = obj_id == target or (target == "*" and not obj.is_static)
        if not matched:
            out[obj_id] = obj
            continue

        traj = obj.trajectory
        before = traj.positions[0].copy()
        if shift_us:
            traj = _time_shift(traj, shift_us)
        if lateral_m or longitudinal_m:
            traj = _spatial_offset(traj, lateral_m, longitudinal_m)

        out[obj_id] = replace(obj, trajectory=traj)
        n_hit += 1
        logger.warning(
            "[PERTURB] scene=%s actor=%s class=%s static=%s "
            "dt=%+.2fs lat=%+.2fm lon=%+.2fm start %s -> %s",
            scene_id,
            obj_id,
            obj.label_class,
            obj.is_static,
            shift_us / 1e6,
            lateral_m,
            longitudinal_m,
            np.round(before, 2),
            np.round(traj.positions[0], 2),
        )

    if n_hit == 0:
        raise ValueError(
            f"perturbation.actor_id={target!r} matched no actor in scene {scene_id}. "
            f"Present: {sorted(traffic_objects.keys())}"
        )

    return type(traffic_objects)(**out)


def log_actor_inventory(traffic_objects, gt_ego_trajectory, scene_id: str) -> None:
    """Print every actor with its closest approach to the ego GT path.

    Read this before choosing actor_id -- perturbing a parked car 80 m away
    proves nothing.
    """
    ego_xy = gt_ego_trajectory.positions[:, :2]
    rows = []
    for obj_id, obj in traffic_objects.items():
        pos = obj.trajectory.positions
        if len(pos) == 0:
            continue
        d = np.linalg.norm(pos[:, None, :2] - ego_xy[None, :, :2], axis=-1)
        i_actor, _ = np.unravel_index(np.argmin(d), d.shape)
        rows.append(
            (
                float(d.min()),
                obj_id,
                obj.label_class,
                obj.is_static,
                len(pos),
                int(obj.trajectory.timestamps_us[i_actor]),
            )
        )

    logger.warning("[ACTORS] scene=%s n=%d", scene_id, len(rows))
    for dist, obj_id, label, static, n, t_us in sorted(rows):
        logger.warning(
            "[ACTORS] id=%-14s class=%-14s static=%-5s n=%-4d "
            "min_dist=%7.2fm at t=%6.2fs",
            obj_id,
            label,
            static,
            n,
            dist,
            t_us / 1e6,
        )