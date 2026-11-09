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


def _speed_change(traj: Trajectory, event_us: int, accel_mps2: float) -> Trajectory:
    """Brake (accel < 0) or speed up (accel > 0) from ``event_us`` onward.

    The actor keeps its own recorded path (lane, curvature, heading); only
    *where along that path* it is at each timestamp changes. Before the event
    nothing changes. After it, speed = v(event) + accel * dt, floored at 0, so
    a braking actor stops and stays stopped. Past the end of the recorded path
    (speed-up case) the actor continues straight along its last heading.
    Timestamps are preserved, so the actor's existence window is unchanged.
    """
    ts = traj.timestamps_us.astype(np.int64)
    if len(ts) < 2 or not (ts[0] < event_us < ts[-1]):
        logger.warning(
            "[PERTURB] speed event at %d us outside actor track [%d, %d]; skipped",
            event_us, ts[0], ts[-1],
        )
        return traj

    pos = traj.positions.astype(np.float64)
    t = (ts - ts[0]) / 1e6
    te = (event_us - ts[0]) / 1e6
    # Distance travelled along the recorded path (xy), made strictly increasing
    # so it can be inverted even where the actor was standing still.
    seg = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)]) + np.arange(len(ts)) * 1e-9

    s_e = float(np.interp(te, t, s))
    h = 0.25  # s, half-window for a smooth speed estimate at the event
    v_e = max(0.0, (np.interp(te + h, t, s) - np.interp(te - h, t, s)) / (2 * h))

    dt = np.clip(t - te, 0.0, None)
    if accel_mps2 < 0.0:
        dt = np.minimum(dt, v_e / -accel_mps2)  # stop, then hold
    s_new = np.where(t > te, s_e + v_e * dt + 0.5 * accel_mps2 * dt**2, s)

    # Pose at each new arc length = recorded pose where the actor was at that
    # arc length (inverse lookup s -> original time).
    on_path = s_new <= s[-1]
    t_src = np.interp(np.minimum(s_new, s[-1]), s, t)
    src = traj.interpolate((ts[0] + np.round(t_src * 1e6)).astype(np.uint64))
    new_pos = src.positions.astype(np.float64)
    new_quat = src.quaternions.astype(np.float64)

    if not on_path.all():  # extrapolate straight past the end of the path
        last_xyz, last_q = pos[-1], traj.quaternions[-1].astype(np.float64)
        x, y, z, w = last_q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        extra = (s_new[~on_path] - s[-1])[:, None]
        new_pos[~on_path] = last_xyz + extra * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        new_quat[~on_path] = last_q

    return Trajectory(ts.astype(np.uint64), new_pos, new_quat)


def perturb_traffic_objects(
    traffic_objects, cfg, scene_id: str, scene_start_us: int | None = None
):
    """Return a new TrafficObjects with ``cfg`` applied. No-op when unset.

    ``scene_start_us`` is the time base for ``speed_event_s`` (the first GT
    ego timestamp, same base as the actor inventory's ``t``).
    """
    if cfg is None or not cfg.actor_id:
        return traffic_objects

    target = str(cfg.actor_id)
    targets = {t.strip() for t in target.split(",") if t.strip()}
    shift_us = int(float(cfg.time_shift_s) * 1e6)
    lateral_m = float(cfg.lateral_offset_m)
    longitudinal_m = float(cfg.longitudinal_offset_m)
    vertical_m = float(getattr(cfg, "vertical_offset_m", 0.0))
    speed_event_s = float(getattr(cfg, "speed_event_s", -1.0))
    accel_mps2 = float(getattr(cfg, "speed_accel_mps2", 0.0))
    do_speed = speed_event_s >= 0.0 and accel_mps2 != 0.0
    if do_speed and scene_start_us is None:
        raise ValueError("speed change needs scene_start_us")
    event_us = int(scene_start_us + speed_event_s * 1e6) if do_speed else 0

    if shift_us == 0 and lateral_m == 0.0 and longitudinal_m == 0.0 \
            and vertical_m == 0.0 and not do_speed:
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
        if do_speed:
            traj = _speed_change(traj, event_us, accel_mps2)
        if lateral_m or longitudinal_m or vertical_m:
            traj = _spatial_offset(traj, lateral_m, longitudinal_m, vertical_m)

        out[obj_id] = replace(obj, trajectory=traj)
        n_hit += 1
        if target in ("*", "**") and n_hit > 5:
            continue
        logger.warning(
            "[PERTURB] scene=%s actor=%s class=%s static=%s "
            "dt=%+.2fs lat=%+.2fm lon=%+.2fm vert=%+.2fm "
            "speed_event=%.2fs accel=%+.2fm/s2 start %s -> %s",
            scene_id, obj_id, obj.label_class, obj.is_static,
            shift_us / 1e6, lateral_m, longitudinal_m, vertical_m,
            speed_event_s, accel_mps2,
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

# --------------------------------------------------------------------------
# box actors: new vehicles that exist in the simulation AND in the images
# --------------------------------------------------------------------------
#
# NuRec can only re-pose actors it reconstructed, so a new vehicle has no
# appearance. Box actors are added to ``traffic_objects`` (so trafficsim
# replay, collision checks and metrics see them) under a ``box_`` track id,
# are excluded from the NuRec render request, and are painted into every
# rendered camera frame as a shaded cuboid (see ``draw_box_actors``).

BOX_ID_PREFIX = "box_"


def is_box_actor(track_id: str) -> bool:
    return str(track_id).startswith(BOX_ID_PREFIX)


def _arc_length(pos: np.ndarray) -> np.ndarray:
    """Distance along an (N,>=2) path, strictly increasing (invertible)."""
    seg = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]) + np.arange(len(pos)) * 1e-9


def _box_arc(t: np.ndarray, s0: float, v0: float, te: float, accel: float) -> np.ndarray:
    """Distance along the path at times ``t`` (s): constant ``v0``, then from
    ``te`` constant ``accel`` (braking stops at 0 m/s and holds). te < 0 or
    accel == 0 means constant speed throughout."""
    v0 = max(0.0, v0)
    if te < 0.0 or accel == 0.0:
        return s0 + v0 * t
    dt = np.clip(t - te, 0.0, None)
    if accel < 0.0:
        dt = np.minimum(dt, v0 / -accel)
    return np.where(t <= te, s0 + v0 * t, s0 + v0 * te + v0 * dt + 0.5 * accel * dt**2)


def _yaw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _box_trajectory(
    ego_traj: Trajectory,
    start_ahead_m: float,
    v0_mps: float,
    event_s: float,
    accel_mps2: float,
    lateral_m: float,
    center_z_m: float,
) -> Trajectory:
    """Box-center trajectory that drives along the GT ego path.

    At each timestamp the box sits on the ego's recorded path at its own
    distance ``s_box(t)``, with the full recorded rig pose there (so it inherits
    the road's height, slope and heading), then shifted ``lateral_m`` to its
    left and ``center_z_m`` up from the rig origin (rig origin = ground under
    the rear axle). Past either end of the path it continues straight.
    """
    ts = ego_traj.timestamps_us.astype(np.int64)
    t = (ts - ts[0]) / 1e6
    pos = ego_traj.positions.astype(np.float64)
    s = _arc_length(pos)
    s_box = _box_arc(t, s[0] + start_ahead_m, v0_mps, event_s, accel_mps2)

    t_src = np.interp(np.clip(s_box, s[0], s[-1]), s, t)
    src = ego_traj.interpolate((ts[0] + np.round(t_src * 1e6)).astype(np.uint64))
    new_pos = src.positions.astype(np.float64)
    new_quat = src.quaternions.astype(np.float64)
    q_all = ego_traj.quaternions.astype(np.float64)
    for mask, idx, extra in (
        (s_box > s[-1], -1, s_box - s[-1]),
        (s_box < s[0], 0, s_box - s[0]),
    ):
        if mask.any():
            yaw = _yaw(q_all[idx])
            direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            new_pos[mask] = pos[idx] + extra[mask][:, None] * direction
            new_quat[mask] = q_all[idx]

    path = Trajectory(ts.astype(np.uint64), new_pos, new_quat)
    offset = Pose(np.array([0.0, lateral_m, center_z_m], dtype=np.float32), _IDENTITY_QUAT)
    return path.transform(offset, is_relative=True)


def add_box_actors(
    traffic_objects,
    cfg,
    gt_ego_trajectory: Trajectory,
    ego_center_ahead_m: float,
    ego_bottom_z_m: float,
    scene_id: str,
):
    """Return ``traffic_objects`` plus one ``TrafficObject`` per box config.

    ``ego_center_ahead_m``: rig origin -> ego bbox center, along x, so that
    ``gap_m`` is measured center-to-center like the collision checker sees it.
    ``ego_bottom_z_m``: rig origin -> bottom of the ego bbox, i.e. the ground.
    """
    boxes = list(getattr(cfg, "box_actors", None) or []) if cfg is not None else []
    if not boxes:
        return traffic_objects

    from alpasim_utils.scenario import AABB, TrafficObject  # local: avoid cycles

    out = dict(traffic_objects)
    for i, b in enumerate(boxes):
        box_id = str(getattr(b, "id", "") or f"{BOX_ID_PREFIX}{i}")
        if not is_box_actor(box_id):
            box_id = BOX_ID_PREFIX + box_id
        if box_id in out:
            raise ValueError(f"box actor id {box_id!r} already exists in scene {scene_id}")
        length, width, height = float(b.length_m), float(b.width_m), float(b.height_m)
        traj = _box_trajectory(
            gt_ego_trajectory,
            start_ahead_m=ego_center_ahead_m + float(b.gap_m),
            v0_mps=float(b.v0_mps),
            event_s=float(b.speed_event_s),
            accel_mps2=float(b.accel_mps2),
            lateral_m=float(b.lateral_m),
            center_z_m=ego_bottom_z_m + height / 2.0,
        )
        out[box_id] = TrafficObject(
            track_id=box_id,
            aabb=AABB(length, width, height),
            trajectory=traj,
            is_static=False,
            label_class="automobile",
        )
        logger.warning(
            "[BOX] scene=%s id=%s size=%.2fx%.2fx%.2fm gap=%+.1fm lat=%+.2fm "
            "v0=%.1fm/s speed_event=%.2fs accel=%+.2fm/s2 start %s",
            scene_id, box_id, length, width, height, float(b.gap_m), float(b.lateral_m),
            float(b.v0_mps), float(b.speed_event_s), float(b.accel_mps2),
            np.round(traj.positions[0], 2),
        )
    return type(traffic_objects)(**out)


# --------------------------------------------------------------------------
# drawing boxes into rendered frames
# --------------------------------------------------------------------------

# corner order: index bits = (x>0, y>0, z>0) for x fwd, y left, z up
_CORNER_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64
)
# faces as corner-index quads, with a flat-shading gray level per face
_FACES = (
    ((5, 7, 6, 4), 165),  # front  (x+)
    ((0, 2, 3, 1), 150),  # rear   (x-)
    ((2, 6, 7, 3), 120),  # left   (y+)
    ((0, 1, 5, 4), 120),  # right  (y-)
    ((1, 3, 7, 5), 200),  # top    (z+)
    ((0, 4, 6, 2), 60),   # bottom (z-)
)
_MIN_DEPTH_M = 0.3


class _FthetaProjector:
    """f-theta camera projection, same math as AlpaSim's eval CameraProjector
    (eval/data.py) and the driver's _FthetaCamera (rectification.py)."""

    def __init__(self, intrinsics, image_wh: tuple[int, int]):
        kind = intrinsics.WhichOneof("camera_param")
        if kind != "ftheta_param":
            raise ValueError(f"box drawing supports ftheta cameras only, got {kind}")
        p = intrinsics.ftheta_param
        sx = image_wh[0] / float(intrinsics.resolution_w)
        sy = image_wh[1] / float(intrinsics.resolution_h)
        self.poly = np.asarray(p.angle_to_pixeldist_poly, dtype=np.float64) * 0.5 * (sx + sy)
        self.c = np.array([p.principal_point_x * sx, p.principal_point_y * sy])
        self.max_angle = p.max_angle if p.max_angle > 0 else np.pi / 2
        lc = p.linear_cde if p.HasField("linear_cde") else None
        self.lin = np.array(
            [[lc.linear_c, lc.linear_d], [lc.linear_e, 1.0]] if lc else [[1.0, 0.0], [0.0, 1.0]]
        )

    def project(self, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        r_xy = np.hypot(x, y)
        theta = np.arctan2(r_xy, z)
        ok = (z > _MIN_DEPTH_M) & (theta <= self.max_angle + 1e-6)
        radii = np.polynomial.polynomial.polyval(theta, self.poly)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(r_xy > 1e-9, radii / np.maximum(r_xy, 1e-9), 0.0)
        uv = (np.stack([x, y], axis=-1) * scale[:, None]) @ self.lin.T + self.c
        return uv, ok


def _box_corners(aabb, pose_se3: np.ndarray) -> np.ndarray:
    half = np.array([aabb.x, aabb.y, aabb.z], dtype=np.float64) / 2.0
    local = _CORNER_SIGNS * half
    return local @ pose_se3[:3, :3].T + pose_se3[:3, 3]


def _edge_points(a: np.ndarray, b: np.ndarray, n: int = 12) -> np.ndarray:
    w = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    return a + (b - a) * w


def draw_box_actors(
    image_bytes: bytes,
    boxes: list[tuple[object, np.ndarray, bool]],
    camera_definition,
    ego_pose_world: Pose,
) -> bytes:
    """Paint cuboids into one encoded camera frame and re-encode it.

    ``boxes``: (aabb, world-from-box SE3, solid). Solid boxes are flat-shaded
    gray with dark edges (box actors); non-solid are thin colored outlines
    (debug overlay of the real actors, used to validate the projection).
    Faces are drawn far-to-near; edges are sampled so the fisheye curvature is
    followed. Real scene geometry does not occlude the box.
    """
    if not boxes:
        return image_bytes
    import io

    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(image_bytes))
    fmt = img.format or "JPEG"
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    proj = _FthetaProjector(camera_definition.intrinsics, img.size)
    world_from_cam = (ego_pose_world @ camera_definition.rig_to_camera).as_se3()
    cam_from_world = np.linalg.inv(world_from_cam.astype(np.float64))
    cam_pos = world_from_cam[:3, 3]

    # far boxes first
    order = sorted(
        range(len(boxes)),
        key=lambda i: -np.linalg.norm(boxes[i][1][:3, 3] - cam_pos),
    )
    for i in order:
        aabb, pose_se3, solid = boxes[i]
        corners_w = _box_corners(aabb, pose_se3.astype(np.float64))
        corners_c = corners_w @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
        faces = []
        for quad, gray in _FACES:
            ring = np.concatenate(
                [_edge_points(corners_c[quad[k]], corners_c[quad[(k + 1) % 4]]) for k in range(4)]
            )
            uv, ok = proj.project(ring)
            if not ok.all():
                continue  # face crosses the camera plane / leaves the lens FOV
            depth = float(np.linalg.norm(corners_c[list(quad)].mean(axis=0)))
            faces.append((depth, [tuple(p) for p in uv], gray))
        for _, poly, gray in sorted(faces, key=lambda f: -f[0]):
            if solid:
                draw.polygon(poly, fill=(gray, gray, gray), outline=(25, 25, 25), width=2)
            else:
                draw.line(poly + [poly[0]], fill=(0, 255, 255), width=2)

    out = io.BytesIO()
    if fmt.upper() == "PNG":
        img.save(out, format="PNG")
    else:
        img.save(out, format="JPEG", quality=95)
    return out.getvalue()
