/**
 * @file controller_touch.cpp
 * @brief Standalone touch demo: move link7 control point to startup EE + world offset.
 *
 * Default: fixed world goal = initial control-point position + kTouchOffsetWorld.
 * With ``--gemini``: while ``gemini_target_ee_active`` > 0.5, goal position is
 * ``R_world_link7 * p_ee + origin_link7_world`` from Redis (p_ee from vision);
 * otherwise the same hardcoded offset goal as default mode.
 *
 * Redis: always ``redis_keys.h`` (hardware) — Franka joint_positions /
 * joint_velocities may be length 7 (Franka arm) or full ``dof``; we expand into
 * the mmp_panda model (3 base + 7 arm + 2 fingers). Command torques published to
 * Redis are the **7 arm** components (Franka ``control_torques``). Not compatible
 * with simviz’s sim key namespace. Do not run two controllers on one Redis.
 */

#include <SaiModel.h>
#include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
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

static bool parse_use_gemini(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--gemini") == 0) {
			return true;
		}
		cerr << "controller_touch: unknown argument: " << argv[i] << "\n";
		cerr << "Usage: controller_touch_zitibot_mmp_panda [--gemini]\n";
		exit(EXIT_FAILURE);
	}
	return false;
}

int main(int argc, char** argv) {
	static const string robot_file =
		string(CS225A_URDF_FOLDER) + "/mmp_panda/mmp_panda.urdf";

	const bool use_gemini_mode = parse_use_gemini(argc, argv);

	// World-frame offset (m) from the startup EE control point to the touch goal.
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

	// When Redis has 7 Franka joints only, we fill base + fingers from this template.
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
	const Vector3d control_point(0, 0, 0.07);
	Affine3d compliant_frame = Affine3d::Identity();
	compliant_frame.translation() = control_point;
	auto pose_task =
		std::make_shared<SaiPrimitives::MotionForceTask>(robot, control_link,
														 compliant_frame);
	pose_task->setPosControlGains(400, 40, 0);
	pose_task->setOriControlGains(0.0, 0.0, 0.0);

	MatrixXd base_selection_matrix = MatrixXd::Zero(3, robot->dof());
	base_selection_matrix(0, 0) = 1;
	base_selection_matrix(1, 1) = 1;
	base_selection_matrix(2, 2) = 1;
	auto base_task =
		std::make_shared<SaiPrimitives::JointTask>(robot, base_selection_matrix);
	base_task->setGains(400, 40, 0);

	auto joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
	const double joint_kp_nullspace_ee = 250.0;
	const double joint_kv_nullspace_ee = 60.0;
	joint_task->setGains(100.0, 20.0, 0);

	MatrixXd gripper_selection_matrix = MatrixXd::Zero(2, dof);
	gripper_selection_matrix(0, dof - 2) = 1;
	gripper_selection_matrix(1, dof - 1) = 1;
	auto gripper_task = std::make_shared<SaiPrimitives::JointTask>(
		robot, gripper_selection_matrix);
	gripper_task->setDynamicDecouplingType(
		SaiPrimitives::DynamicDecouplingType::IMPEDANCE);
	gripper_task->setGains(5e3, 1e2, 0);

	const Vector2d gripper_opening(0.04, -0.04);

	const Vector3d p0 = robot->position(control_link, control_point);
	const Vector3d touch_goal_hardcoded = p0 + kTouchOffsetWorld;
	Vector3d touch_goal_world = touch_goal_hardcoded;
	if (use_gemini_mode) {
		cout << "controller_touch: --gemini  goal from Redis when "
				"gemini_target_ee_active>0.5; else hardcoded offset "
			 << kTouchOffsetWorld.transpose() << " m (world)\n";
	} else {
		cout << "controller_touch: goal = EE at start + offset "
			 << kTouchOffsetWorld.transpose() << " m (world)\n";
	}

	Matrix3d ee_touch_R = link7_orientation_world(0.0, 0.0, 0.0);
	bool touch_arrived_printed = false;

	pose_task->reInitializeTask();
	base_task->reInitializeTask();
	joint_task->reInitializeTask();
	gripper_task->reInitializeTask();
	gripper_task->setGoalPosition(gripper_opening);
	base_task->setGoalPosition(robot->q().head(3));
	base_task->setGains(220, 55, 0);
	pose_task->setGoalPosition(touch_goal_hardcoded);
	pose_task->setGoalOrientation(ee_touch_R);
	pose_task->setPosControlGains(200, 95, 0);
	pose_task->setOriControlGains(240.0, 55.0, 0.0);
	joint_task->setGoalPosition(robot->q());
	joint_task->setGains(joint_kp_nullspace_ee, joint_kv_nullspace_ee, 0);

	VectorXd command_torques = VectorXd::Zero(dof);
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

	runloop = true;
	const double control_freq = 1000;
	SaiCommon::LoopTimer timer(control_freq, 1e6);

	VectorXd q_prev_full = q0;
	VectorXd dq_prev_full = dq0;

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

		if (use_gemini_mode) {
			bool gemini_live = false;
			Vector3d p_ee = Vector3d::Zero();
			try {
				VectorXd act = redis_client.getEigen(GEMINI_TARGET_EE_ACTIVE_KEY);
				if (act.size() == 1 && act(0) > 0.5) {
					VectorXd pe = redis_client.getEigen(GEMINI_TARGET_EE_POSITION_KEY);
					if (pe.size() == 3) {
						p_ee = pe;
						gemini_live = true;
					}
				}
			} catch (const std::exception&) {
				// Keys missing or malformed — fall back to hardcoded goal
			}
			if (gemini_live) {
				const Vector3d origin_w =
					robot->position(control_link, Vector3d::Zero());
				touch_goal_world = origin_w + robot->rotation(control_link) * p_ee;
			} else {
				touch_goal_world = touch_goal_hardcoded;
			}
		}

		pose_task->setGoalPosition(touch_goal_world);
		pose_task->setGoalOrientation(ee_touch_R);
		gripper_task->setGoalPosition(gripper_opening);

		N_prec.setIdentity();
		pose_task->updateTaskModel(N_prec);
		base_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
		gripper_task->updateTaskModel(base_task->getTaskAndPreviousNullspace());
		joint_task->updateTaskModel(gripper_task->getTaskAndPreviousNullspace());

		command_torques = pose_task->computeTorques() + base_task->computeTorques() +
						  joint_task->computeTorques() + gripper_task->computeTorques();

		const Vector3d x_touch = robot->position(control_link, control_point);
		const double touch_err = (x_touch - touch_goal_world).norm();
		if (touch_err < ee_pos_convergence_tol) {
			if (!touch_arrived_printed) {
				cout << "controller_touch: at goal (hold)\n";
				touch_arrived_printed = true;
			}
		} else if (touch_err >= ee_pos_convergence_tol * 2.0) {
			touch_arrived_printed = false;
		}

		Matrix3d ee_goal_R_world = ee_touch_R;
		Vector3d ee_goal_pos_world = touch_goal_world;
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

		if (dof >= kFrankaArmStartIdx + kFrankaArmDof) {
			redis_client.setEigen(CONTROL_TORQUES_KEY,
					command_torques.segment(kFrankaArmStartIdx, kFrankaArmDof));
		} else {
			redis_client.setEigen(CONTROL_TORQUES_KEY, command_torques);
		}
	}

	timer.stop();
	cout << "\ncontroller_touch timer stats:\n";
	timer.printInfoPostRun();
	if (dof >= kFrankaArmStartIdx + kFrankaArmDof) {
		redis_client.setEigen(CONTROL_TORQUES_KEY, VectorXd::Zero(kFrankaArmDof));
	} else {
		redis_client.setEigen(CONTROL_TORQUES_KEY, 0 * command_torques);
	}
	return 0;
}
