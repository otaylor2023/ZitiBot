/**
 * @file controller_touch.cpp
 * @brief Standalone touch demo: move link7 control point to a fixed world goal.
 *
 * Latches **one** world goal = first synced EE (from Redis, after expand) +
 * ``kTouchOffsetWorld`` (world axes). Stored in ``std::optional`` and never
 * recomputed from the moving tip.
 *
 * Redis: ``redis_keys.h`` — Franka joint_positions / joint_velocities in;
 * EE goal keys out. Torques optional (dry run: pose task only, no publish).
 */

#include <SaiModel.h>
#include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

using namespace std;
using namespace Eigen;
using namespace SaiPrimitives;

#include <signal.h>
static bool runloop = false;
static void sighandler(int) { runloop = false; }

#include "redis_keys.h"

/** Franka driver publishes 7 arm joints; mmp_panda URDF is 3 base + 7 arm + 2 fingers. */
static constexpr int kFrankaArmDof = 7;
static constexpr int kFrankaArmStartIdx = 3;

/** If @p v_redis has length ``dof``, return it. If length 7, copy arm segment into
 *  ``template_full`` (keeps base + fingers from the template). */
static VectorXd expand_franka_redis_to_model(const VectorXd& v_redis, int dof,
		const VectorXd& template_full, const string& key_name) {
	if (v_redis.size() == static_cast<Index>(dof)) {
		return v_redis;
	}
	if (v_redis.size() == kFrankaArmDof &&
		dof >= kFrankaArmStartIdx + kFrankaArmDof) {
		VectorXd v = template_full;
		v.segment(kFrankaArmStartIdx, kFrankaArmDof) = v_redis;
		return v;
	}
	throw runtime_error(
		"controller_touch: Redis \"" + key_name + "\" length " +
		to_string(v_redis.size()) + " != model dof " + to_string(dof) +
		" and is not " + to_string(kFrankaArmDof) +
		" (Franka arm-only).");
}

// World goal for link7 (same convention as controller.cpp).
static Matrix3d link7_orientation_world(double roll_deg, double pitch_deg,
										double yaw_deg) {
	const double d2r = M_PI / 180.0;
	const double r = roll_deg * d2r;
	const double p = pitch_deg * d2r;
	const double y = yaw_deg * d2r;
	const Matrix3d R_down_and_rpy =
		(AngleAxisd(y, Vector3d::UnitZ()) * AngleAxisd(p, Vector3d::UnitY()) *
		 AngleAxisd(r, Vector3d::UnitX()) * AngleAxisd(M_PI, Vector3d::UnitX()))
			.toRotationMatrix();
	const Matrix3d R_finger_align =
		AngleAxisd(M_PI / 4.0, Vector3d::UnitZ()).toRotationMatrix();
	return R_down_and_rpy * R_finger_align;
}

static int redis_fatal(const string& msg) {
	cerr << msg << endl;
	return EXIT_FAILURE;
}

int main(int argc, char** argv) {
	(void)argc;
	(void)argv;

	static const string robot_file =
		string(CS225A_URDF_FOLDER) + "/mmp_panda/mmp_panda.urdf";

	static const Vector3d kTouchOffsetWorld(0.2, 0.0, 0.0);
	const double ee_pos_convergence_tol = 2.5e-2;

	SaiCommon::RedisClient redis_client;
	try {
		redis_client.connect();
	} catch (const std::exception& e) {
		return redis_fatal(string("controller_touch: Redis connect failed: ") + e.what());
	}

	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	auto robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
	const int dof = robot->dof();

	cout << "\ncontroller_touch: Redis keys (from redis_keys.h)\n"
		 << "  READ  " << JOINT_POSITIONS_KEY << "\n"
		 << "  READ  " << JOINT_VELOCITIES_KEY << "\n"
		 << "  WRITE " << EE_GOAL_POSITION_KEY << "\n"
		 << "  WRITE " << EE_GOAL_ROTATION_KEY << "\n"
		 << "  WRITE " << EE_GOAL_ACTIVE_KEY << "\n"
		 << "  WRITE " << CONTROL_TORQUES_KEY << " (vector length "
		 << (dof >= kFrankaArmStartIdx + kFrankaArmDof ? kFrankaArmDof : dof) << ")\n"
		 << "  NOTE: CONTROL_TORQUES_KEY publish is commented out (dry run).\n"
		 << "  NOTE: Only pose task torques computed (no base / gripper / posture).\n"
		 << "  model dof=" << dof << "\n"
		 << flush;

	VectorXd q_expand_template = VectorXd::Zero(dof);
	VectorXd dq_expand_template = VectorXd::Zero(dof);
	if (dof >= 2) {
		q_expand_template(dof - 2) = 0.04;
		q_expand_template(dof - 1) = -0.04;
	}

	VectorXd q0;
	VectorXd dq0;
	try {
		q0 = expand_franka_redis_to_model(redis_client.getEigen(JOINT_POSITIONS_KEY), dof,
				q_expand_template, JOINT_POSITIONS_KEY);
		dq0 = expand_franka_redis_to_model(redis_client.getEigen(JOINT_VELOCITIES_KEY), dof,
				dq_expand_template, JOINT_VELOCITIES_KEY);
	} catch (const std::exception& e) {
		return redis_fatal(string("controller_touch: initial Redis read failed: ") +
						   e.what());
	}
	robot->setQ(q0);
	robot->setDq(dq0);

	robot->updateModel();

	const string control_link = "link7";
	const Vector3d control_point(0, 0, 0.1);
	cout << "  EE task: link=\"" << control_link << "\"  control_point in link (m) "
		 << control_point.transpose() << "\n\n"
		 << flush;
	Affine3d compliant_frame = Affine3d::Identity();
	compliant_frame.translation() = control_point;
	auto pose_task =
		std::make_shared<SaiPrimitives::MotionForceTask>(robot, control_link,
														 compliant_frame);
	pose_task->setPosControlGains(400, 40, 0);
	pose_task->setOriControlGains(0.0, 0.0, 0.0);
	pose_task->enableVelocitySaturation(0.3, M_PI/3.0);

	// Base / posture / gripper tasks (disabled — pose-only dry run).
	MatrixXd base_selection_matrix = MatrixXd::Zero(3, robot->dof());
	base_selection_matrix(0, 0) = 1;
	base_selection_matrix(1, 1) = 1;
	base_selection_matrix(2, 2) = 1;
	auto base_task =
			std::make_shared<SaiPrimitives::JointTask>(robot, base_selection_matrix);
	base_task->setGains(400, 40, 0);
	//
	// auto joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
	// const double joint_kp_nullspace_ee = 250.0;
	// const double joint_kv_nullspace_ee = 60.0;
	// joint_task->setGains(100.0, 20.0, 0);
	//
	// MatrixXd gripper_selection_matrix = MatrixXd::Zero(2, dof);
	// gripper_selection_matrix(0, dof - 2) = 1;
	// gripper_selection_matrix(1, dof - 1) = 1;
	// auto gripper_task = std::make_shared<SaiPrimitives::JointTask>(
	// 		robot, gripper_selection_matrix);
	// gripper_task->setDynamicDecouplingType(
	// 		SaiPrimitives::DynamicDecouplingType::IMPEDANCE);
	// gripper_task->setGains(5e3, 1e2, 0);
	//
	// const Vector2d gripper_opening(0.04, -0.04);

	std::optional<Vector3d> latched_pose_goal_world;

	cout << "controller_touch: goal = first synced EE (Redis) + offset "
		 << kTouchOffsetWorld.transpose() << " m (world)\n";

	Matrix3d ee_touch_R = link7_orientation_world(0.0, 0.0, 0.0);
	bool touch_arrived_printed = false;

	pose_task->reInitializeTask();
	base_task->reInitializeTask();
	// joint_task->reInitializeTask();
	// gripper_task->reInitializeTask();
	// gripper_task->setGoalPosition(gripper_opening);
	base_task->setGoalPosition(robot->q().head(3));
	base_task->setGains(220, 55, 0);
	pose_task->setGoalPosition(robot->position(control_link, control_point));
	pose_task->setGoalOrientation(ee_touch_R);
	pose_task->setPosControlGains(200, 95, 0);
	pose_task->setOriControlGains(240.0, 55.0, 0.0);
	// joint_task->setGoalPosition(robot->q());
	// joint_task->setGains(joint_kp_nullspace_ee, joint_kv_nullspace_ee, 0);

	VectorXd command_torques = VectorXd::Zero(dof);
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

	runloop = true;
	const double control_freq = 1000;
	SaiCommon::LoopTimer timer(control_freq, 1e6);

	VectorXd q_prev_full = q0;
	VectorXd dq_prev_full = dq0;
	auto status_print_last =
			std::chrono::steady_clock::now() - std::chrono::seconds(1);

	while (runloop) {
		timer.waitForNextLoop();
		try {
			VectorXd q_redis = redis_client.getEigen(JOINT_POSITIONS_KEY);
			VectorXd dq_redis = redis_client.getEigen(JOINT_VELOCITIES_KEY);
			VectorXd q_in =
					expand_franka_redis_to_model(q_redis, dof, q_prev_full, JOINT_POSITIONS_KEY);
			VectorXd dq_in = expand_franka_redis_to_model(
					dq_redis, dof, dq_prev_full, JOINT_VELOCITIES_KEY);
			robot->setQ(q_in);
			robot->setDq(dq_in);
			q_prev_full = q_in;
			dq_prev_full = dq_in;
		} catch (const std::exception& e) {
			cerr << "controller_touch: Redis read failed: " << e.what() << endl;
			runloop = false;
			break;
		}
		robot->updateModel();

		if (!latched_pose_goal_world.has_value()) {
			const Vector3d cur_ee = robot->position(control_link, control_point);
			latched_pose_goal_world.emplace(cur_ee + kTouchOffsetWorld);
			cout << fixed << setprecision(4);
			cout << "controller_touch: latched goal (world m)\n"
				 << "  EE_current=" << cur_ee.transpose() << "\n"
				 << "  goal      =" << latched_pose_goal_world->transpose() << "\n"
				 << "  offset    =" << kTouchOffsetWorld.transpose() << "\n"
				 << flush;
			pose_task->reInitializeTask();
			pose_task->setGoalPosition(latched_pose_goal_world.value());
			pose_task->setGoalOrientation(ee_touch_R);
			pose_task->setPosControlGains(200, 95, 0);
			pose_task->setOriControlGains(240.0, 55.0, 0.0);
		}

		const Vector3d pose_goal_world = latched_pose_goal_world.value();

		pose_task->setGoalPosition(pose_goal_world);
		pose_task->setGoalOrientation(ee_touch_R);
		// gripper_task->setGoalPosition(gripper_opening);
		base_task->setGoalPosition(robot->q().head(3));

		N_prec.setIdentity();
		pose_task->updateTaskModel(N_prec);
		base_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
		// gripper_task->updateTaskModel(base_task->getTaskAndPreviousNullspace());
		// joint_task->updateTaskModel(gripper_task->getTaskAndPreviousNullspace());

		const VectorXd tau_pose = pose_task->computeTorques();
		const VectorXd tau_base = base_task->computeTorques();
		command_torques = tau_pose + tau_base;

		const Vector3d x_touch = robot->position(control_link, control_point);
		const double touch_err = (x_touch - pose_goal_world).norm();
		if (touch_err < ee_pos_convergence_tol) {
			if (!touch_arrived_printed) {
				cout << "controller_touch: at goal (hold)\n";
				touch_arrived_printed = true;
			}
		} else if (touch_err >= ee_pos_convergence_tol * 2.0) {
			touch_arrived_printed = false;
		}

		// Status line at 1 Hz (wall clock): q, EE, pose errors, pose torques (not sent).
		const auto now = std::chrono::steady_clock::now();
		if (now - status_print_last >= std::chrono::seconds(1)) {
			status_print_last = now;
			cout << fixed << setprecision(5) << "controller_touch: q=" << robot->q().transpose()
				 << "\n  EE_world_m=" << x_touch.transpose()
				 << "\n  pose_err_pos_m=" << pose_task->getPositionError().transpose()
				 << "\n  pose_err_ori=" << pose_task->getOrientationError().transpose()
				 << "\n  tau_pose=" << tau_pose.transpose()
				 << "\n  tau_base=" << tau_base.transpose()
				 << "\n  tau_total=" << command_torques.transpose()
				 << flush;
		}

		Matrix3d ee_goal_R_world = ee_touch_R;
		Vector3d ee_goal_pos_world = pose_goal_world;
		VectorXd ee_goal_R_flat(9);
		ee_goal_R_flat << ee_goal_R_world.col(0), ee_goal_R_world.col(1),
			ee_goal_R_world.col(2);
		VectorXd ee_goal_p(3);
		ee_goal_p << ee_goal_pos_world(0), ee_goal_pos_world(1), ee_goal_pos_world(2);
		redis_client.setEigen(EE_GOAL_POSITION_KEY, ee_goal_p);
		redis_client.setEigen(EE_GOAL_ROTATION_KEY, ee_goal_R_flat);
		VectorXd ee_goal_active_v(1);
		ee_goal_active_v(0) = 1.0;
		redis_client.setEigen(EE_GOAL_ACTIVE_KEY, ee_goal_active_v);

		// Robot torque command (commented out — uncomment to drive hardware).
		// if (dof >= kFrankaArmStartIdx + kFrankaArmDof) {
		// 	redis_client.setEigen(CONTROL_TORQUES_KEY,
		// 			command_torques.segment(kFrankaArmStartIdx, kFrankaArmDof));
		// } else {
		// 	redis_client.setEigen(CONTROL_TORQUES_KEY, command_torques);
		// }
	}

	timer.stop();
	cout << "\ncontroller_touch timer stats:\n";
	timer.printInfoPostRun();
	// if (dof >= kFrankaArmStartIdx + kFrankaArmDof) {
	// 	redis_client.setEigen(CONTROL_TORQUES_KEY, VectorXd::Zero(kFrankaArmDof));
	// } else {
	// 	redis_client.setEigen(CONTROL_TORQUES_KEY, 0 * command_torques);
	// }
	return 0;
}
