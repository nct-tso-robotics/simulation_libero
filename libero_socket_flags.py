"""LIBERO-specific socket communication flags and keys."""
from enum import Enum


class LiberoRequestKeys(str, Enum):
    """JSON keys that may appear in a client request to the LIBERO server."""
    ROUTE_NAME = "route_name"
    TASK_ID = "task_id"
    ROBOT_ACTION = "robot_action"  # 7D: position delta (3) + orientation delta (3) + gripper (1)
    REQUEST_AGENTVIEW = "request_agentview_rgb"
    REQUEST_EYE_IN_HAND = "request_eye_in_hand_rgb"
    REQUEST_EE_POS = "request_ee_pos"
    REQUEST_EE_ORI = "request_ee_ori"
    REQUEST_EE_STATES = "request_ee_states"
    REQUEST_GRIPPER_STATES = "request_gripper_states"
    REQUEST_JOINT_STATES = "request_joint_states"
    REQUEST_LANGUAGE_INSTRUCTION = "request_language_instruction"
    COMPRESSION_TYPE = "compression_type"
    EPISODE_IDX = "episode_idx"
    TASK_IDX = "task_idx"


class LiberoResponseKeys(str, Enum):
    """JSON keys that may appear in a server response from the LIBERO server."""
    STATUS = "status"
    ERROR_MSG = "error_msg"
    RESULT = "result"
    TASK_ID = "task_id"
    # Image observations (LIBERO camera naming convention)
    AGENTVIEW_RGB = "agentview_rgb"
    EYE_IN_HAND_RGB = "eye_in_hand_rgb"
    # Proprioceptive observations (LIBERO naming convention)
    EE_POS = "ee_pos"  # (3,) end-effector position
    EE_ORI = "ee_ori"  # (3,) end-effector orientation (euler)
    EE_STATES = "ee_states"  # (6,) concatenation of ee_pos and ee_ori
    GRIPPER_STATES = "gripper_states"  # (2,) gripper qpos
    JOINT_STATES = "joint_states"  # (7,) joint positions
    LANGUAGE_INSTRUCTION = "language_instruction"
    DONE = "done"
    SUCCESS = "success"
    REWARD = "reward"
    TIMESTEP = "timestep"
    MAX_TIMESTEPS = "max_timesteps"
    COMPRESSION_TYPE = "compression_type"
    IMAGE_HEIGHT = "image_height"
    IMAGE_WIDTH = "image_width"


class LiberoRoutes(str, Enum):
    """Legal route names for the LIBERO server."""
    TASK_STATUS = "task_status"
    GET_OBSERVATION = "get_observation"
    SEND_ACTION = "send_action"
    RESET_EPISODE = "reset_episode"


class LiberoStatus(str, Enum):
    """Possible values of the status field in a server response."""
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    PROCESSING = "PROCESSING"
    EPISODE_DONE = "EPISODE_DONE"
    EPISODE_SUCCESS = "EPISODE_SUCCESS"
    WAITING_FOR_RESET = "WAITING_FOR_RESET"