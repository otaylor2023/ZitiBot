"""OpenSai / Franka Redis key names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpenSaiRedisKeys:
    cartesian_task_goal_position: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_position"
    )
    cartesian_task_goal_orientation: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_orientation"
    )
    cartesian_task_current_position: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position"
    )
    cartesian_task_current_orientation: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_orientation"
    )
    # Live cartesian-task gains. OpenSai polls these every controller
    # loop (registered with ``addToReceiveGroup`` in
    # ``RobotControllerRedisInterface.cpp``), so writing a new value to
    # Redis updates the PD gains at ~1 kHz — no controller restart
    # needed. Stored as a JSON 1-element list (e.g. ``[300.0]``) which
    # ``MotionForceTask::setPosControlGains`` wraps to ``kp * I3``.
    # See ``zitibot_core.gains`` for the read/write/context-manager API.
    cartesian_task_position_kp: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::position_kp"
    )
    cartesian_task_position_kv: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::position_kv"
    )
    cartesian_task_position_ki: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::position_ki"
    )
    cartesian_task_orientation_kp: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::orientation_kp"
    )
    cartesian_task_orientation_kv: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::orientation_kv"
    )
    # Live OTG linear-velocity cap. Like the gains above this is in the
    # controller's ``addToReceiveGroup`` so OpenSai re-reads it every loop —
    # writing it slows / speeds the internal trajectory generator without a
    # restart. UNLIKE the gains, the C++ side binds this to a plain ``double``
    # parsed with ``std::stod`` (see RedisClient.cpp), so it must be written as
    # a BARE scalar string (e.g. ``"0.03"``), NOT a JSON 1-element list.
    cartesian_task_otg_max_linear_velocity: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::otg_max_linear_velocity"
    )
    # Live OTG angular-velocity cap (rad/s). Same bare-scalar ``std::stod``
    # format as the linear cap above; also in ``addToReceiveGroup`` so writes
    # take effect every loop without a restart.
    cartesian_task_otg_max_angular_velocity: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::otg_max_angular_velocity"
    )
    # Live OTG enable flag. Boolean written as a bare ``"0"`` / ``"1"`` string
    # (parsed on the C++ side like the other receive-group scalars). Toggling
    # it disables the internal online trajectory generator so streamed goal
    # poses are tracked directly instead of being re-planned every write —
    # useful when publishing a dense trajectory (e.g. the scramble stir).
    cartesian_task_otg_enabled: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::otg_enabled"
    )
    active_controller: str = "opensai::controllers::FrankaRobot::active_controller_name"
    config_file_name: str = "::sai-interfaces-webui::config_file_name"
    endeffector_transform: str = "opensai::redis_driver::FrankaRobot::T_end_effector"
    # Joint velocities published by the Franka redis_driver at 1 kHz
    # (``robot_state.dq`` straight from libfranka). Used as a clean
    # "is the arm still moving?" signal for the convergence-wait loop —
    # JSON-roundtripping EE positions then numerically differentiating
    # them is much noisier than reading dq directly.
    joint_velocities: str = "opensai::sensors::FrankaRobot::joint_velocities"
    joint_positions: str = "opensai::sensors::FrankaRobot::joint_positions"
    # Joint controller goal keys. The joint_controller defined in
    # zitibot_panda.xml expects all three to be in radians (size 7 for
    # the Franka). ``goal_velocity`` / ``goal_acceleration`` are usually
    # left at zero unless we're streaming a trajectory.
    joint_task_goal_position: str = (
        "opensai::controllers::FrankaRobot::joint_controller::joint_task::goal_position"
    )
    joint_task_goal_velocity: str = (
        "opensai::controllers::FrankaRobot::joint_controller::joint_task::goal_velocity"
    )
    joint_task_goal_acceleration: str = (
        "opensai::controllers::FrankaRobot::joint_controller::joint_task::goal_acceleration"
    )
    # Live joint-task gains. Same naming as the sai-interfaces joint-task UI
    # sliders (<joint_task prefix>::kp/kv/ki).
    joint_task_kp: str = (
        "opensai::controllers::FrankaRobot::joint_controller::joint_task::kp"
    )
    joint_task_kv: str = (
        "opensai::controllers::FrankaRobot::joint_controller::joint_task::kv"
    )
    joint_task_ki: str = (
        "opensai::controllers::FrankaRobot::joint_controller::joint_task::ki"
    )


@dataclass(frozen=True)
class RedisKeys(OpenSaiRedisKeys):
    gripper_mode: str = "opensai::FrankaRobot::gripper::mode"
    gripper_desired_width: str = "opensai::FrankaRobot::gripper::desired_width"
    gripper_desired_speed: str = "opensai::FrankaRobot::gripper::desired_speed"
    gripper_desired_force: str = "opensai::FrankaRobot::gripper::desired_force"
    gripper_max_width: str = "opensai::FrankaRobot::gripper::max_width"
    gripper_current_width: str = "opensai::FrankaRobot::gripper::current_width"
    # Grasp result published by the gripper driver after each force-close
    # (GRASP mode): "1" if an object is held (libfranka grasp() succeeded
    # AND the fingers ended up held apart), "0" if the close found nothing.
    # The Python ``grasp`` helper resets this to ``GRIPPER_GRASP_PENDING``
    # before issuing a new grasp so callers can distinguish a fresh result
    # from a stale one. See drivers/FrankaPanda/redis_driver/gripper.cpp.
    gripper_grasp_success: str = "opensai::FrankaRobot::gripper::grasp_success"
    # Driver readiness flag. "0" while the gripper driver is homing + running
    # its startup init (which resets the command keys), "1" once it is in its
    # main loop and honoring client commands. Clients must wait for "1" before
    # issuing gripper commands, else their commands get wiped by the init.
    gripper_ready: str = "opensai::FrankaRobot::gripper::ready"


KEYS = RedisKeys()

# Franka gripper driver modes (see drivers/FrankaPanda/redis_driver/gripper.cpp).
GRIPPER_MODE_MOVE = "m"
GRIPPER_MODE_GRASP = "g"
GRIPPER_MODE_OPEN_MAX = "o"

# Sentinel written to ``gripper_grasp_success`` while a grasp is in flight,
# before the driver publishes the "1"/"0" result.
GRIPPER_GRASP_PENDING = "pending"

CONTROLLER_WAIT_DT_S = 0.05
CONTROLLER_WAIT_TIMEOUT_S = 30.0
