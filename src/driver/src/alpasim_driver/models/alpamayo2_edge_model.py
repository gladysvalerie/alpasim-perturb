# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Alpamayo 2 Edge wrapper implementing the common trajectory-model interface.

Based on :mod:`alpamayo2_model` (Alpamayo 2 Super). Differences, all taken from
the released ``alpamayo2_edge`` inference path (commit 02f55a7):

* Six cameras with a different number of frames per camera: front wide and
  front tele get 5 frames at -2.0/-1.0/-0.5/-0.1/0 s, the other four get 3
  frames at -1.0/-0.1/0 s (``load_physical_aiavdataset.py`` L21-30,
  ``sample.py`` L13). The driver keeps ``context_length`` frames per camera
  (2.1 s at 10 Hz) and this wrapper picks the frames Edge wants by timestamp.
* 20 ego history poses (-1.9 .. 0 s) instead of 16.
* A navigation text on every call (``sample.py`` L118). It comes from the route
  through :func:`route_to_nav_text`, like Alpamayo 1.5; without a route the
  wrapper falls back to :attr:`FALLBACK_NAV_TEXT`.
* Edge needs ``transformers==5.17.0`` while the other Alpamayo drivers pin
  4.57.1, so it runs in its own Python environment.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
from alpamayo2_edge.checkpoint import load_model
from alpamayo2_edge.helper import prepare_model_inputs, to_device

from ..schema import ModelConfig
from .alpamayo_base import (
    CAMERA_NAME_TO_INDEX,
    _validate_camera_frame_at_or_before_t0,
    _validate_ego_history_span,
    build_ego_history,
    configure_deterministic_runtime,
    route_in_rig_at_t0,
)
from .base import (
    BaseTrajectoryModel,
    CameraImages,
    DriveCommand,
    ModelInputValidationError,
    ModelPrediction,
    PredictionInput,
)
from .nav_text import route_to_nav_text
from .trajectory_selection import (
    TrajectorySelectionStrategy,
    plan_in_local_frame,
    select_trajectory,
)

logger = logging.getLogger(__name__)

# Public six-camera rig profile [0, 1, 2, 3, 5, 6] (no rear tele).
EDGE_CAMERA_IDS = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_front_tele_30fov",
)
# Rig indices of the cameras that get the long (2 s, 5 frame) history.
_FRONT_CAMERA_INDICES = (1, 6)
_FRONT_OFFSETS_US = (-2_000_000, -1_000_000, -500_000, -100_000, 0)
_OTHER_OFFSETS_US = (-1_000_000, -100_000, 0)


class Alpamayo2EdgeModel(BaseTrajectoryModel):
    """Alpamayo 2 Edge wrapper implementing the common interface."""

    DTYPE: torch.dtype = torch.bfloat16
    NUM_HISTORY_STEPS: int = 20
    HISTORY_TIME_STEP: float = 0.1
    # Frames kept per camera: 2.1 s at 10 Hz, one frame more than the 2 s the
    # front cameras need, so the oldest request is covered even when the newest
    # frame is slightly after t0.
    DEFAULT_CONTEXT_LENGTH: int = 22
    MIN_CONTEXT_LENGTH: int = 21
    OUTPUT_FREQUENCY_HZ: int = 10
    MAX_GENERATION_LENGTH: int = 256
    # Edge refuses to run without a navigation text. Used when the scenario
    # submits no route (route_generator_type: NONE). Same wording as the
    # Alpamayo 1.5 route classifier.
    FALLBACK_NAV_TEXT: str = "Continue straight"

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> Alpamayo2EdgeModel:
        """Create an Alpamayo2EdgeModel from driver configuration."""
        if output_frequency_hz != cls.OUTPUT_FREQUENCY_HZ:
            raise ValueError(
                f"{cls.__name__} predicts waypoints at {cls.OUTPUT_FREQUENCY_HZ}Hz "
                f"and cannot resample, but output_frequency_hz is "
                f"{output_frequency_hz}."
            )
        if model_cfg.cfg_guidance_weight is not None:
            raise ValueError(f"{cls.__name__} does not support cfg_guidance_weight.")
        if model_cfg.trajectory_candidate_microbatch_size is not None:
            raise ValueError(
                f"{cls.__name__} does not support trajectory_candidate_microbatch_size; "
                "sample all candidates in one call."
            )
        selection = model_cfg.trajectory_selection
        return cls(
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
            context_length=context_length or cls.DEFAULT_CONTEXT_LENGTH,
            force_determinism=model_cfg.force_determinism,
            num_traj_samples=model_cfg.num_trajectory_samples,
            selection_strategy=TrajectorySelectionStrategy(selection.strategy),
            max_num_distance_points=selection.max_num_distance_points,
            skip_first_n_distance_points=selection.skip_first_n_distance_points,
        )

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        num_traj_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
        diffusion_steps: int = 10,
        force_determinism: bool = False,
        selection_strategy: TrajectorySelectionStrategy = (
            TrajectorySelectionStrategy.ALWAYS_FIRST
        ),
        max_num_distance_points: int = 64,
        skip_first_n_distance_points: int = 0,
    ):
        """Initialize Alpamayo 2 Edge.

        Args:
            checkpoint_path: HuggingFace model ID (``nvidia/Alpamayo2-Edge``) or
                a local checkpoint directory.
            device: Torch device for inference.
            camera_ids: Camera logical IDs; must be exactly the six Edge cameras.
            context_length: Frames kept per camera (at least 21 = 2 s at 10 Hz).
            num_traj_samples: Trajectory samples to draw before selection.
            top_p: Top-p sampling parameter for CoT generation.
            temperature: Temperature for CoT sampling.
            diffusion_steps: Flow-matching steps of the action expert.
            force_determinism: Make stochastic inference repeatable from each
                prediction's inference seed.
            selection_strategy: How to select one sampled trajectory.
            max_num_distance_points: Waypoints entering the selection distance
                average.
            skip_first_n_distance_points: Leading waypoints excluded from the
                selection distance average.
        """
        if sorted(camera_ids) != sorted(EDGE_CAMERA_IDS):
            raise ValueError(
                f"{self.__class__.__name__} needs exactly the cameras "
                f"{list(EDGE_CAMERA_IDS)}, got {camera_ids}."
            )
        if context_length < self.MIN_CONTEXT_LENGTH:
            raise ValueError(
                f"{self.__class__.__name__} needs at least {self.MIN_CONTEXT_LENGTH} "
                f"frames per camera (2 s at 10 Hz), got context_length={context_length}."
            )
        if num_traj_samples < 1:
            raise ValueError(
                f"num_traj_samples must be at least 1, got {num_traj_samples}."
            )
        if (
            selection_strategy is not TrajectorySelectionStrategy.ALWAYS_FIRST
            and num_traj_samples < 2
        ):
            raise ValueError(
                f"Trajectory selection strategy {selection_strategy} needs at least "
                f"2 samples to choose from, got num_traj_samples={num_traj_samples}."
            )
        if diffusion_steps < 1:
            raise ValueError(
                f"diffusion_steps must be at least 1, got {diffusion_steps}."
            )

        if force_determinism:
            configure_deterministic_runtime()

        logger.info("Loading Alpamayo 2 Edge checkpoint from %s", checkpoint_path)
        self._model, _loading_info = load_model(
            checkpoint_path,
            device=str(device),
            dtype="bfloat16",
            attn_implementation="sdpa",
        )

        action_space = self._model.expert.action_space
        reported_time_step = 1.0 / self.OUTPUT_FREQUENCY_HZ
        if not math.isclose(
            action_space.dt, reported_time_step, rel_tol=1e-5, abs_tol=1e-6
        ):
            raise ValueError(
                f"{self.__class__.__name__} checkpoint predicts a waypoint every "
                f"{action_space.dt}s, but the driver reports waypoints every "
                f"{reported_time_step}s (OUTPUT_FREQUENCY_HZ="
                f"{self.OUTPUT_FREQUENCY_HZ})."
            )

        self._device = device
        self._camera_ids = list(camera_ids)
        self._context_length = context_length
        self._num_traj_samples = num_traj_samples
        self._top_p = top_p
        self._temperature = temperature
        self._diffusion_steps = diffusion_steps
        self._force_determinism = force_determinism
        self._selection_strategy = selection_strategy
        self._max_num_distance_points = max_num_distance_points
        self._skip_first_n_distance_points = skip_first_n_distance_points

        logger.info(
            "Initialised Alpamayo2EdgeModel with %d cameras, context_length=%d, "
            "num_traj_samples=%d, selection_strategy=%s",
            len(self._camera_ids),
            context_length,
            num_traj_samples,
            selection_strategy,
        )

    # ------------------------------------------------------------------
    # BaseTrajectoryModel interface
    # ------------------------------------------------------------------

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self.OUTPUT_FREQUENCY_HZ

    def _encode_command(self, command: DriveCommand) -> Any:
        """Edge takes navigation as text (see :meth:`_nav_text`), not a command."""
        return None

    # ------------------------------------------------------------------
    # Input assembly
    # ------------------------------------------------------------------

    def _select_frames(
        self, camera_images: CameraImages, t0_us: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick Edge's frames per camera by timestamp.

        For each requested time ``t0 + offset`` the newest frame at or before it
        is used, as in Edge's PhysicalAI-AV loader. If no frame is that old, the
        oldest kept frame is used and a warning is logged.

        Returns:
            ``image_frames`` uint8 CPU ``[22, 1, 3, H, W]``, ``camera_indices``
            int64 ``[22]`` and ``absolute_timestamps`` int64 ``[22]``.
        """
        images: list[torch.Tensor] = []
        cameras: list[int] = []
        timestamps: list[int] = []
        for cam_id in sorted(self._camera_ids, key=CAMERA_NAME_TO_INDEX.__getitem__):
            cam_index = CAMERA_NAME_TO_INDEX[cam_id]
            offsets = (
                _FRONT_OFFSETS_US
                if cam_index in _FRONT_CAMERA_INDICES
                else _OTHER_OFFSETS_US
            )
            frames = sorted(camera_images[cam_id], key=lambda frame: frame[0])
            if not frames:
                raise ModelInputValidationError(f"No frames for camera {cam_id}.")
            frame_times = np.array([int(ts) for ts, _ in frames], dtype=np.int64)
            for offset in offsets:
                target_us = t0_us + offset
                idx = int(np.searchsorted(frame_times, target_us, side="right")) - 1
                if idx < 0:
                    logger.warning(
                        "%s: no frame at or before %d us (t0%+d ms); using the oldest "
                        "kept frame at %d us. Increase context_length or the warm-up.",
                        cam_id,
                        target_us,
                        offset // 1000,
                        frame_times[0],
                    )
                    idx = 0
                ts, img = frames[idx]
                # HWC uint8 -> 1x3xHxW uint8 on the CPU (Edge resizes on the CPU).
                images.append(
                    torch.as_tensor(img).to("cpu").permute(2, 0, 1).unsqueeze(0)
                )
                cameras.append(cam_index)
                timestamps.append(int(ts))
        return (
            torch.stack(images, dim=0).contiguous(),
            torch.tensor(cameras, dtype=torch.int64),
            torch.tensor(timestamps, dtype=torch.int64),
        )

    def _nav_text(
        self, prediction_input: PredictionInput, pose_local_to_rig_t0: Any
    ) -> str:
        """Navigation instruction from the route, or the fallback text."""
        if prediction_input.route is not None:
            nav_text = route_to_nav_text(
                route_in_rig_at_t0(
                    prediction_input.route,
                    prediction_input.ego_pose_history,
                    pose_local_to_rig_t0,
                )
            )
            if nav_text:
                return nav_text
        return self.FALLBACK_NAV_TEXT

    # ------------------------------------------------------------------
    # Main prediction
    # ------------------------------------------------------------------

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        """Generate a single trajectory prediction from cameras and ego history."""
        if self._force_determinism:
            torch.manual_seed(prediction_input.inference_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(prediction_input.inference_seed)

        self._validate_cameras(prediction_input.camera_images)

        if not prediction_input.ego_pose_history:
            raise ModelInputValidationError(
                "Alpamayo2EdgeModel needs ego pose history to select planning t0."
            )
        # Same planning t0 as the Alpamayo 2 Super wrapper: the latest ego pose.
        planning_t0_us = int(prediction_input.ego_pose_history[-1].timestamp_us)
        _validate_camera_frame_at_or_before_t0(
            prediction_input.camera_images,
            planning_t0_us=planning_t0_us,
            model_name=self.__class__.__name__,
        )
        _validate_ego_history_span(
            prediction_input.ego_pose_history,
            planning_t0_us=planning_t0_us,
            num_history_steps=self.NUM_HISTORY_STEPS,
            history_time_step=self.HISTORY_TIME_STEP,
            model_name=self.__class__.__name__,
        )
        ego_history_xyz, ego_history_rot, pose_local_to_rig_t0 = build_ego_history(
            prediction_input.ego_pose_history,
            planning_t0_us,
            self.NUM_HISTORY_STEPS,
            self.HISTORY_TIME_STEP,
        )

        nav_text = self._nav_text(prediction_input, pose_local_to_rig_t0)
        image_frames, camera_indices, frame_timestamps = self._select_frames(
            prediction_input.camera_images, planning_t0_us
        )
        data = {
            "image_frames": image_frames,
            "camera_indices": camera_indices,
            "absolute_timestamps": frame_timestamps,
            "ego_history_xyz": ego_history_xyz.float().cpu(),
            "ego_history_rot": ego_history_rot.float().cpu(),
            "nav_text": [nav_text],
        }
        model_inputs = prepare_model_inputs(
            data, self._model.config, self._model.tokenizer
        )
        model_inputs = to_device(model_inputs, device=self._device)

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self._device.type,
                dtype=self.DTYPE,
                enabled=self._device.type == "cuda",
            ),
        ):
            pred_xyz, pred_rot, _logprob, extra = (
                self._model.sample_trajectories_from_data(
                    model_inputs,
                    top_p=self._top_p,
                    temperature=self._temperature,
                    num_traj_sets=1,
                    num_traj_samples=self._num_traj_samples,
                    max_generation_length=self.MAX_GENERATION_LENGTH,
                    diffusion_kwargs={"inference_step": self._diffusion_steps},
                    return_extra=True,
                )
            )

        # Candidates are in the rig frame at t0, shaped
        # [batch=1, num_traj_sets=1, K, T, ...] -> (K, T, ...).
        candidate_positions = pred_xyz[0, 0].float().cpu().numpy()
        candidate_rotations = pred_rot[0, 0].float().cpu().numpy()
        waypoint_timestamps_us = self._waypoint_timestamps_us(
            planning_t0_us, num_waypoints=candidate_positions.shape[1]
        )
        select_ix = select_trajectory(
            candidate_positions=candidate_positions,
            candidate_timestamps_us=waypoint_timestamps_us,
            previous_plan_in_local=prediction_input.previous_plan,
            pose_local_to_rig_t0=pose_local_to_rig_t0,
            strategy=self._selection_strategy,
            max_num_distance_points=self._max_num_distance_points,
            skip_first_n_distance_points=self._skip_first_n_distance_points,
        )
        selected_positions = candidate_positions[select_ix]
        selected_rotations = candidate_rotations[select_ix]

        reasoning_text = None
        if "cot" in extra:
            reasoning_text = str(extra["cot"][0, 0, select_ix])
        logger.info(
            "Alpamayo2EdgeModel nav=%r Chain-of-Causation: %s", nav_text, reasoning_text
        )

        return ModelPrediction(
            candidate_positions=candidate_positions,
            candidate_rotations=candidate_rotations,
            selected_index=select_ix,
            reasoning_text=reasoning_text,
            model_t0_us=planning_t0_us,
            pose_local_to_rig_t0=pose_local_to_rig_t0,
            waypoint_timestamps_us=waypoint_timestamps_us,
            selected_plan=plan_in_local_frame(
                selected_positions,
                selected_rotations,
                waypoint_timestamps_us,
                pose_local_to_rig_t0,
            ),
        )

    def _waypoint_timestamps_us(self, t0_us: int, num_waypoints: int) -> np.ndarray:
        """Timestamps of predicted waypoints, starting one step after t0."""
        step_us = 1_000_000 // self.output_frequency_hz
        return t0_us + np.arange(1, num_waypoints + 1, dtype=np.uint64) * step_us
