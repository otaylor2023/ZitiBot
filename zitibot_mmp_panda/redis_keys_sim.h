/**
 * @file redis_keys_sim.h
 * @author William Chong (williamchong@stanford.edu)
 * @brief Redis keys for Sai simulation + simviz (mmp_panda mobile manipulator).
 * @version 0.1
 * @date 2022-04-30
 *
 * @copyright Copyright (c) 2022
 *
 * Hardware / Franka uses ``redis_keys.h``.
 */

const std::string JOINT_ANGLES_KEY = "sai::sim::mmp_panda::sensors::q";
const std::string JOINT_VELOCITIES_KEY = "sai::sim::mmp_panda::sensors::dq";
const std::string JOINT_TORQUES_COMMANDED_KEY = "sai::sim::mmp_panda::actuators::fgc";
const std::string CONTROLLER_RUNNING_KEY = "sai::sim::mmp_panda::controller";
// World-frame translation of dynamic object "empty_bowl" from sim (Vector3, meters).
const std::string BOWL_POSITION_KEY =
	"sai::sim::mmp_panda::sensors::empty_bowl::position";
// Pose-task goal for link7 + control_point (world frame). Rotation = 9 scalars, column-major.
const std::string EE_GOAL_POSITION_KEY =
	"sai::sim::mmp_panda::desire::ee_goal_position";
const std::string EE_GOAL_ROTATION_KEY =
	"sai::sim::mmp_panda::desire::ee_goal_rotation";
// 1.0 while controller commands the arm pose task; 0.0 during MOVE_BASE only.
const std::string EE_GOAL_ACTIVE_KEY =
	"sai::sim::mmp_panda::desire::ee_goal_active";
// World-frame point (Vector3, m); optional for external tools / future use. Bowl FSM
// pick reference is computed in the controller from BOWL_POSITION_KEY + placeholder.
const std::string TOUCH_POINT_TARGET_KEY =
	"sai::sim::mmp_panda::desire::touch_point_target";
// Gemini keypoint target in EE / link7 frame (m). Same JSON encoding as setEigen
// elsewhere (SaiCommon). Rotation is 9 scalars column-major; vision demo publishes
// identity until you wire a real orientation. Active 1.0 = last SPACE had valid 3D.
const std::string GEMINI_TARGET_EE_POSITION_KEY =
	"sai::sim::mmp_panda::desire::gemini_target_ee_position";
const std::string GEMINI_TARGET_EE_ROTATION_KEY =
	"sai::sim::mmp_panda::desire::gemini_target_ee_rotation";
const std::string GEMINI_TARGET_EE_ACTIVE_KEY =
	"sai::sim::mmp_panda::desire::gemini_target_ee_active";
