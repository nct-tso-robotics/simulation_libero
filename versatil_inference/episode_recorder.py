"""Video and trajectory recording for evaluation episodes."""

import csv
from enum import Enum
from pathlib import Path

import cv2
import numpy as np


class VideoCodec(str, Enum):
    """Supported video codec fourcc codes."""

    MJPG = "MJPG"


class EpisodeRecorder:
    """Records video frames and trajectory data for a single environment.

    Buffers frames and flushes them periodically to a video writer.
    On save, renames the video file to include the success/failure result
    and writes the trajectory as a CSV with named columns.
    """

    BUFFER_SIZE = 10
    VIDEO_FPS = 10

    def __init__(
        self,
        environment_id: str,
        language_instruction: str,
        trajectory_columns: list[str],
        frame_skip: int = 3,
    ):
        self.safe_instruction = (
            language_instruction.replace(" ", "_").replace("/", "-")
        )
        self.trajectory_columns = trajectory_columns
        self.frame_skip = frame_skip
        self.step_counter = 0
        self.frames_buffer: list[np.ndarray] = []
        self.trajectory_rows: list[dict[str, float]] = []
        self.writer: cv2.VideoWriter | None = None
        self.filepath: Path | None = None
        self.environment_id = environment_id
        self.num_saves = 0

    def _init_writer(self, output_directory: Path) -> None:
        """Initialize the video writer from the first buffered frame.

        Args:
            output_directory: Directory to write the video file to.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{self.environment_id}_unknown_"
            f"{self.safe_instruction}_{self.num_saves}.avi"
        )
        self.filepath = output_directory / filename
        height, width = self.frames_buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*VideoCodec.MJPG.value)
        self.writer = cv2.VideoWriter(
            str(self.filepath), fourcc, self.VIDEO_FPS, (width, height)
        )

    def _flush_buffer(self, output_directory: Path) -> None:
        """Write buffered frames to the video file.

        Args:
            output_directory: Directory to write the video file to.
        """
        if not self.frames_buffer:
            return
        if self.writer is None:
            self._init_writer(output_directory)
        for frame in self.frames_buffer:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(bgr)
        self.frames_buffer = []

    def add_observation(
        self,
        frame: np.ndarray,
        trajectory_row: dict[str, float],
        reward: float,
        output_directory: Path,
    ) -> None:
        """Buffer a frame and trajectory row. Flushes video when buffer is full.

        Args:
            frame: Image frame to record.
            trajectory_row: Dict of trajectory column values.
            reward: Reward for this step.
            output_directory: Directory to write the video file to on flush.
        """
        trajectory_row["reward"] = reward
        self.trajectory_rows.append(trajectory_row)
        if self.step_counter % self.frame_skip == 0:
            self.frames_buffer.append(frame)
            if len(self.frames_buffer) >= self.BUFFER_SIZE:
                self._flush_buffer(output_directory)
        self.step_counter += 1

    def reset(self) -> None:
        """Release the writer and clear all buffers for a new episode."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.frames_buffer = []
        self.trajectory_rows = []
        self.filepath = None
        self.step_counter = 0

    def save(self, was_success: bool, output_directory: Path) -> None:
        """Flush remaining frames, rename video with result, write trajectory CSV.

        Args:
            was_success: Whether the episode was successful.
            output_directory: Directory to save final video and trajectory CSV.
        """
        if not self.frames_buffer and self.writer is None:
            return
        self._flush_buffer(output_directory)
        self.writer.release()
        self.writer = None
        output_directory.mkdir(parents=True, exist_ok=True)
        result = "success" if was_success else "fail"
        file_prefix = (
            f"{self.environment_id}_{result}_"
            f"{self.safe_instruction}_{self.num_saves}"
        )
        new_video_path = output_directory / f"{file_prefix}.avi"
        self.filepath.rename(new_video_path)
        csv_columns = self.trajectory_columns + ["reward"]
        csv_path = output_directory / f"{file_prefix}_trajectory.csv"
        with open(csv_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(self.trajectory_rows)
        self.filepath = None
        self.num_saves += 1