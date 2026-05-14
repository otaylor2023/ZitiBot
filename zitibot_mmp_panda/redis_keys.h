/**
 * @file redis_keys.h
 * @brief Redis keys for the physical robot / Franka driver stack (not simulation).
 *
 * Simulation and the main pick FSM still use ``redis_keys_sim.h``.
 *
 * Reference keys from the driver (see comments in git history / lab docs):
 * - sai::sensors::FrankaRobot::joint_positions / joint_velocities / joint_torques
 * - sai::commands::FrankaRobot::control_torques
 */

// Franka (SAI driver): measured joint state in; commanded torques out.
const std::string JOINT_POSITIONS_KEY = "sai::sensors::FrankaRobot::joint_positions";
const std::string JOINT_VELOCITIES_KEY = "sai::sensors::FrankaRobot::joint_velocities";
const std::string JOINT_TORQUES_KEY = "sai::sensors::FrankaRobot::joint_torques";
const std::string CONTROL_TORQUES_KEY = "sai::commands::FrankaRobot::control_torques";

// tidybot base pose (optional for future controllers).
const std::string TIDYBOT_POSITION_KEY = "tidybot01::pos";
const std::string TIDYBOT_ORIENTATION_KEY = "tidybot01::ori";

// Touch controller publishes world-frame EE goal for link7 control point (debug / viz).
const std::string EE_GOAL_POSITION_KEY = "tidybot01::ee_goal_position";
const std::string EE_GOAL_ROTATION_KEY = "tidybot01::ee_goal_rotation";
const std::string EE_GOAL_ACTIVE_KEY = "tidybot01::ee_goal_active";

// Gemini keypoint in EE / link7 frame (m). Same JSON encoding as SaiCommon setEigen.
const std::string GEMINI_TARGET_EE_POSITION_KEY =
	"tidybot01::gemini_target_ee_position";
const std::string GEMINI_TARGET_EE_ROTATION_KEY =
	"tidybot01::gemini_target_ee_rotation";
const std::string GEMINI_TARGET_EE_ACTIVE_KEY = "tidybot01::gemini_target_ee_active";
