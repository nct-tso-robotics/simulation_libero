"""Socket communication flags for Libero simulation server."""

from enum import Enum


DEFAULT_CLIENT_NAME = "unknown"


class LiberoRoute(str, Enum):
    """Route names for the Libero server."""

    GET_OBSERVATION = "get_observation"
    SEND_ACTION = "send_action"
    REGISTER_CLIENT = "register_client"


class LiberoStatus(str, Enum):
    """Status values in server responses."""

    FINISHED = "FINISHED"
    ERROR = "ERROR"
    WAITING_ACTION = "WAITING_ACTION"
    CREATING_ENV = "CREATING_ENV"
    PROCESSING = "PROCESSING"


class LiberoResponseKey(str, Enum):
    """Response keys in server responses."""

    STATUS = "status"
    ERROR_MSG = "error_msg"
    RESET_ENVIRONMENT_INDICES = "reset_environment_indices"
    TIMESTEP = "timestep"
    IMAGE_HEIGHT = "image_height"
    IMAGE_WIDTH = "image_width"


class LiberoRequestKey(str, Enum):
    """Request keys in client requests."""

    ROUTE_NAME = "route_name"
    REQUESTED_KEYS = "requested_keys"
    ACTIONS = "actions"
    CLIENT_NAME = "client_name"
    COMPRESSION_TYPE = "compression_type"


class LiberoObservationKey(str, Enum):
    """Observation keys matching VersatIL metadata."""

    AGENTVIEW = "agentview_rgb"
    EYE_IN_HAND = "eye_in_hand_rgb"
    EE_POS_ACTION = "ee_pos_action"
    EE_ORI_ACTION = "ee_ori_action"
    GRIPPER_STATE_ACTION = "gripper_state_action"
    LANGUAGE_INSTRUCTION = "language_instruction"


class LiberoGymKey(str, Enum):
    """Raw Libero environment observation keys."""

    AGENTVIEW_IMAGE = "agentview_image"
    EYE_IN_HAND_IMAGE = "robot0_eye_in_hand_image"
    EE_POS = "robot0_eef_pos"
    EE_QUAT = "robot0_eef_quat"
    GRIPPER_QPOS = "robot0_gripper_qpos"
    JOINT_POS = "robot0_joint_pos"


class LiberoTrajectoryColumn(str, Enum):
    """Column names for trajectory CSV recording."""

    EE_POS_X = "ee_pos_x"
    EE_POS_Y = "ee_pos_y"
    EE_POS_Z = "ee_pos_z"
    EE_QUAT_X = "ee_quat_x"
    EE_QUAT_Y = "ee_quat_y"
    EE_QUAT_Z = "ee_quat_z"
    EE_QUAT_W = "ee_quat_w"
    GRIPPER_QPOS_0 = "gripper_qpos_0"
    GRIPPER_QPOS_1 = "gripper_qpos_1"


class TaskSuiteName(str, Enum):
    """Libero task suite names."""

    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"
    LIBERO_ALL = "libero_all" 


LIBERO_ALL_SUITES: list[str] = [
    TaskSuiteName.LIBERO_SPATIAL.value,
    TaskSuiteName.LIBERO_OBJECT.value,
    TaskSuiteName.LIBERO_GOAL.value,
    TaskSuiteName.LIBERO_10.value,
]


TASK_SUITE_MAX_STEPS: dict[str, int] = {
    TaskSuiteName.LIBERO_SPATIAL.value: 220,
    TaskSuiteName.LIBERO_OBJECT.value: 280,
    TaskSuiteName.LIBERO_GOAL.value: 300,
    TaskSuiteName.LIBERO_10.value: 520,
    TaskSuiteName.LIBERO_90.value: 400,
}

DEFAULT_MAX_TIMESTEPS = 300