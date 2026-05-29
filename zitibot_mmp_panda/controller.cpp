/**
 * @file controller.cpp
 * @brief Controller file
 *
 * Gains: SaiPrimitives tasks expose setPosControlGains / setOriControlGains / setGains only;
 * there are no preset "modes" in this repo (see course starters for similar manual values).
 *
 */

#include <SaiModel.h>
#include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;
using namespace Eigen;
using namespace SaiPrimitives;

#include <signal.h>
bool runloop = false;
void sighandler(int) { runloop = false; }

#include "redis_keys_sim.h"

// World goal for link7: +Z along −world Z (approach down), then yaw about +world Z (degrees).
// Finger joints use rpy (0,0,−π/4) on link7 (see mmp_panda.urdf); post-multiply +π/4 about
// link +Z cancels that so the opening lines up with world X/Y when yaw is 0°/90°/…
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
	// URDF finger_joint* origin: rpy="0 0 -0.78539816339" → child is Rz(−π/4) from link7.
	// Use R_link7 = R_nom * Rz(+π/4) so R_link7 * Rz(−π/4) = R_nom (aligned opening).
	const Matrix3d R_finger_align =
		AngleAxisd(M_PI / 4.0, Vector3d::UnitZ()).toRotationMatrix();
	return R_down_and_rpy * R_finger_align;
}

static int redis_fatal(const string& msg) {
	cerr << msg << endl;
	return EXIT_FAILURE;
}

/** @return false if @p v.rows() * v.cols() != n */
static bool require_eigen_length(const VectorXd& v, int n, const string& key) {
	if (v.size() != n) {
		cerr << "Controller: Redis key \"" << key << "\" must have length " << n
			 << ", got " << v.size() << "." << endl;
		return false;
	}
	return true;
}

enum State {
	MOVE_BASE = 0,
	EE_ABOVE_BOWL = 1,
	EE_GRASP_BOWL = 2,
	EE_LIFT_BOWL = 3,
	BASE_TO_POUR = 4,
	EE_POUR_BOWL = 5,
};

int main(int, char**) {
	static const string robot_file =
		string(CS225A_URDF_FOLDER) + "/mmp_panda/mmp_panda.urdf";

	int state = MOVE_BASE;
	int grasp_phase = 0;
	double grasp_phase_t0 = 0.0;
	double tilt_t0 = 0.0;
	Matrix3d ee_tilt_R_end = Matrix3d::Identity();
	Vector3d wrist_pivot_world = Vector3d::Zero();
	bool tilt_complete_printed = false;
	Matrix3d ee_hover_R = Matrix3d::Identity();

	auto redis_client = SaiCommon::RedisClient();
	try {
		redis_client.connect();
	} catch (const std::exception& e) {
		return redis_fatal(string("Controller: Redis connect failed: ") + e.what());
	}

	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	auto robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
	const int dof = robot->dof();
	VectorXd q0;
	VectorXd dq0;
	try {
		q0 = redis_client.getEigen(JOINT_ANGLES_KEY);
		dq0 = redis_client.getEigen(JOINT_VELOCITIES_KEY);
	} catch (const std::exception& e) {
		return redis_fatal(string("Controller: initial Redis read failed: ") +
						   e.what());
	}
	if (!require_eigen_length(q0, dof, JOINT_ANGLES_KEY) ||
		!require_eigen_length(dq0, dof, JOINT_VELOCITIES_KEY)) {
		return EXIT_FAILURE;
	}
	robot->setQ(q0);
	robot->setDq(dq0);
	robot->updateModel();
	VectorXd q_hold_for_pour = VectorXd::Zero(dof);
	VectorXd command_torques = VectorXd::Zero(dof);
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

	const string control_link = "link7";
	const Vector3d control_point = Vector3d(0, 0, 0.07);
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
	const double joint_kp_move = 100.0;
	const double joint_kv_move = 20.0;
	const double joint_kp_nullspace_ee = 250.0;
	const double joint_kv_nullspace_ee = 60.0;
	joint_task->setGains(joint_kp_move, joint_kv_move, 0);

	MatrixXd gripper_selection_matrix = MatrixXd::Zero(2, dof);
	gripper_selection_matrix(0, dof - 2) = 1;
	gripper_selection_matrix(1, dof - 1) = 1;
	auto gripper_task = std::make_shared<SaiPrimitives::JointTask>(
		robot, gripper_selection_matrix);
	gripper_task->setDynamicDecouplingType(
		SaiPrimitives::DynamicDecouplingType::IMPEDANCE);
	gripper_task->setGains(5e3, 1e2, 0);

	// URDF: finger_joint1 in [0, 0.04], finger_joint2 in [-0.04, 0]. Max spread = open;
	// both 0 = closed (pinch).
	Vector2d gripper_opening(0.04, -0.04);
	Vector2d gripper_closed(0.0, 0.0);

	const Vector3d base_target(0.15, 0.18, 0.0);
	const Vector3d base_pour_target(1.35, 0.18, 0.0);

	VectorXd bowl_read = redis_client.getEigen(BOWL_POSITION_KEY);
	Vector3d bowl_pose = Vector3d::Zero();
	if (bowl_read.size() == 3) {
		bowl_pose << bowl_read(0), bowl_read(1), bowl_read(2);
	}

	// Bowl FSM: bowl origin from BOWL_POSITION_KEY; grasp = bowl + offset below (placeholder
	// in world frame — replace with R_bowl * p_local when orientation is available).
	static const Vector3d kBowlOriginToPickPointPlaceholder(0.0, -0.14, 0.2);
	const Vector3d grasp_position = bowl_pose + kBowlOriginToPickPointPlaceholder;
	// Hover / grasp / lift offsets are relative to pick_reference_world.
	const Vector3d hover_offset(0, 0, 0.15);
	// After grasp, move straight up in world +Z from the grasp goal (meters).
	const double lift_dz = 0.15;

	VectorXd q_initial = robot->q();
	VectorXd q_desired = q_initial;
	q_desired.head(3) = base_target;
	q_desired(dof - 2) = gripper_opening(0);
	q_desired(dof - 1) = gripper_opening(1);
	gripper_task->setGoalPosition(gripper_opening);
	joint_task->setGoalPosition(q_desired);

	const double base_axis_tol_xy = 0.015;  // m; both X and Y must be this close
	const double base_axis_tol_yaw = 0.04;  // rad
	const double ee_pos_convergence_tol = 2.5e-2;
	const double gripper_convergence_tol = 8e-3;
	const double grasp_open_dwell_sec = 1.0;
	const double grasp_closed_dwell_sec = 1.0;
	// After lift: slerp to +90° about world +Y (horizontal hinge in Z-up world).
	const double tilt_duration_sec = 6.0;

	runloop = true;
	int main_rc = 0;
	double control_freq = 1000;
	SaiCommon::LoopTimer timer(control_freq, 1e6);

	while (runloop) {
		timer.waitForNextLoop();
		const double sim_time = timer.elapsedSimTime();

		VectorXd q_read;
		VectorXd dq_read;
		try {
			q_read = redis_client.getEigen(JOINT_ANGLES_KEY);
			dq_read = redis_client.getEigen(JOINT_VELOCITIES_KEY);
		} catch (const std::exception& e) {
			cerr << "Controller: Redis read failed: " << e.what() << endl;
			main_rc = EXIT_FAILURE;
			runloop = false;
			break;
		}
		if (!require_eigen_length(q_read, dof, JOINT_ANGLES_KEY) ||
			!require_eigen_length(dq_read, dof, JOINT_VELOCITIES_KEY)) {
			main_rc = EXIT_FAILURE;
			runloop = false;
			break;
		}
		robot->setQ(q_read);
		robot->setDq(dq_read);
		robot->updateModel();

		const Vector3d ee_hover_world = grasp_position + hover_offset;
		Vector3d ee_lift_world = grasp_position + Vector3d(0.0, 0.0, lift_dz);

		if (state == MOVE_BASE) {
			N_prec.setIdentity();
			joint_task->updateTaskModel(N_prec);
			gripper_task->updateTaskModel(
				joint_task->getTaskAndPreviousNullspace());

			command_torques =
				joint_task->computeTorques() + gripper_task->computeTorques();

			const Vector3d base_err = robot->q().head(3) - base_target;
			if (std::abs(base_err(0)) < base_axis_tol_xy &&
				std::abs(base_err(1)) < base_axis_tol_xy &&
				std::abs(base_err(2)) < base_axis_tol_yaw) {
				cout << "MOVE_BASE -> EE_ABOVE_BOWL\n";
				pose_task->reInitializeTask();
				base_task->reInitializeTask();
				joint_task->reInitializeTask();
				gripper_task->reInitializeTask();
				gripper_task->setGoalPosition(gripper_opening);

				base_task->setGoalPosition(base_target);
				base_task->setGains(220, 55, 0);

				// RPY degrees (world-fixed X, then Y, then Z), after flipping +Z to −world Z.
				ee_hover_R = link7_orientation_world(0.0, 0.0, 0.0);
				pose_task->setGoalPosition(ee_hover_world);
				pose_task->setGoalOrientation(ee_hover_R);
				pose_task->setPosControlGains(200, 95, 0);
				pose_task->setOriControlGains(240.0, 55.0, 0.0);

				joint_task->setGoalPosition(robot->q());
				joint_task->setGains(joint_kp_nullspace_ee, joint_kv_nullspace_ee, 0);

				state = EE_ABOVE_BOWL;
			}
		} else if (state == EE_ABOVE_BOWL) {
			pose_task->setGoalPosition(ee_hover_world);
			pose_task->setGoalOrientation(ee_hover_R);
			gripper_task->setGoalPosition(gripper_opening);

			N_prec.setIdentity();
			pose_task->updateTaskModel(N_prec);
			base_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
			gripper_task->updateTaskModel(
				base_task->getTaskAndPreviousNullspace());
			joint_task->updateTaskModel(
				gripper_task->getTaskAndPreviousNullspace());

			command_torques = pose_task->computeTorques() +
							  base_task->computeTorques() +
							  joint_task->computeTorques() +
							  gripper_task->computeTorques();

			Vector3d x = robot->position(control_link, control_point);
			if ((x - ee_hover_world).norm() < ee_pos_convergence_tol) {
				cout << "EE_ABOVE_BOWL -> EE_GRASP_BOWL (gripper open, then descend)\n";
				gripper_task->reInitializeTask();
				gripper_task->setGoalPosition(gripper_opening);
				pose_task->setGoalPosition(grasp_position);
				pose_task->setGoalOrientation(ee_hover_R);
				grasp_phase = 0;
				state = EE_GRASP_BOWL;
			}
		} else if (state == EE_GRASP_BOWL) {
			// 0: descend with gripper open
			// 1: hold at grasp, open, grasp_open_dwell_sec
			// 2: command close until fingers converged
			// 3: hold closed at grasp, grasp_closed_dwell_sec, then lift
			if (grasp_phase == 0 || grasp_phase == 1) {
				pose_task->setGoalPosition(grasp_position);
				pose_task->setGoalOrientation(ee_hover_R);
				gripper_task->setGoalPosition(gripper_opening);
			} else {
				pose_task->setGoalPosition(grasp_position);
				pose_task->setGoalOrientation(ee_hover_R);
				gripper_task->setGoalPosition(gripper_closed);
			}

			N_prec.setIdentity();
			pose_task->updateTaskModel(N_prec);
			base_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
			gripper_task->updateTaskModel(
				base_task->getTaskAndPreviousNullspace());
			joint_task->updateTaskModel(
				gripper_task->getTaskAndPreviousNullspace());

			command_torques = pose_task->computeTorques() +
							  base_task->computeTorques() +
							  joint_task->computeTorques() +
							  gripper_task->computeTorques();

			if (grasp_phase == 0) {
				Vector3d x = robot->position(control_link, control_point);
				if ((x - grasp_position).norm() < ee_pos_convergence_tol) {
					cout << "EE_GRASP_BOWL: at grasp pose; dwell open "
						 << grasp_open_dwell_sec << " s\n";
					grasp_phase = 1;
					grasp_phase_t0 = sim_time;
				}
			} else if (grasp_phase == 1) {
				if (sim_time - grasp_phase_t0 >= grasp_open_dwell_sec) {
					cout << "EE_GRASP_BOWL: closing gripper\n";
					gripper_task->reInitializeTask();
					gripper_task->setGoalPosition(gripper_closed);
					grasp_phase = 2;
				}
			} else if (grasp_phase == 2) {
				Vector2d gerr =
					robot->q().segment<2>(dof - 2) - gripper_closed;
				if (gerr.norm() < gripper_convergence_tol) {
					cout << "EE_GRASP_BOWL: gripper closed; dwell "
						 << grasp_closed_dwell_sec << " s\n";
					grasp_phase = 3;
					grasp_phase_t0 = sim_time;
				}
			} else if (grasp_phase == 3) {
				if (sim_time - grasp_phase_t0 >= grasp_closed_dwell_sec) {
					cout << "EE_GRASP_BOWL -> EE_LIFT_BOWL\n";
					pose_task->reInitializeTask();
					pose_task->setGoalPosition(ee_lift_world);
					pose_task->setGoalOrientation(ee_hover_R);
					gripper_task->setGoalPosition(gripper_closed);
					grasp_phase = 0;
					state = EE_LIFT_BOWL;
				}
			}
		} else if (state == EE_LIFT_BOWL) {
			pose_task->setGoalPosition(ee_lift_world);
			pose_task->setGoalOrientation(ee_hover_R);
			gripper_task->setGoalPosition(gripper_closed);

			N_prec.setIdentity();
			pose_task->updateTaskModel(N_prec);
			base_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
			gripper_task->updateTaskModel(
				base_task->getTaskAndPreviousNullspace());
			joint_task->updateTaskModel(
				gripper_task->getTaskAndPreviousNullspace());

			command_torques = pose_task->computeTorques() +
							  base_task->computeTorques() +
							  joint_task->computeTorques() +
							  gripper_task->computeTorques();

			Vector3d x_lift = robot->position(control_link, control_point);
			if ((x_lift - ee_lift_world).norm() < ee_pos_convergence_tol) {
				cout << "EE_LIFT_BOWL -> BASE_TO_POUR (joint hold + base only)\n";
				q_hold_for_pour = robot->q();
				joint_task->reInitializeTask();
				joint_task->setGains(joint_kp_move, joint_kv_move, 0);
				VectorXd q_goal_pour = q_hold_for_pour;
				q_goal_pour.head(3) = base_pour_target;
				joint_task->setGoalPosition(q_goal_pour);
				gripper_task->reInitializeTask();
				gripper_task->setGoalPosition(gripper_closed);
				state = BASE_TO_POUR;
			}
		} else if (state == BASE_TO_POUR) {
			VectorXd q_goal_pour = q_hold_for_pour;
			q_goal_pour.head(3) = base_pour_target;
			joint_task->setGoalPosition(q_goal_pour);
			gripper_task->setGoalPosition(gripper_closed);

			N_prec.setIdentity();
			joint_task->updateTaskModel(N_prec);
			gripper_task->updateTaskModel(
				joint_task->getTaskAndPreviousNullspace());

			command_torques =
				joint_task->computeTorques() + gripper_task->computeTorques();

			const Vector3d base_err_pour =
				robot->q().head(3) - base_pour_target;
			if (std::abs(base_err_pour(0)) < base_axis_tol_xy &&
				std::abs(base_err_pour(1)) < base_axis_tol_xy &&
				std::abs(base_err_pour(2)) < base_axis_tol_yaw) {
				cout << "BASE_TO_POUR -> EE_POUR_BOWL (slow 90° about world Y)\n";
				pose_task->reInitializeTask();
				base_task->reInitializeTask();
				base_task->setGoalPosition(base_pour_target);
				base_task->setGains(220, 55, 0);
				joint_task->reInitializeTask();
				joint_task->setGoalPosition(robot->q());
				joint_task->setGains(joint_kp_nullspace_ee,
									 joint_kv_nullspace_ee, 0);
				gripper_task->reInitializeTask();
				gripper_task->setGoalPosition(gripper_closed);

				ee_hover_R = robot->rotation(control_link);
				ee_tilt_R_end =
					AngleAxisd(M_PI / 2.0, Vector3d::UnitY()).toRotationMatrix() *
					ee_hover_R;
				wrist_pivot_world =
					robot->position(control_link, Vector3d::Zero());
				tilt_t0 = sim_time;
				tilt_complete_printed = false;
				pose_task->setGoalPosition(wrist_pivot_world +
										   ee_hover_R * control_point);
				pose_task->setGoalOrientation(ee_hover_R);
				pose_task->setPosControlGains(200, 95, 0);
				pose_task->setOriControlGains(240.0, 55.0, 0.0);
				state = EE_POUR_BOWL;
			}
		} else if (state == EE_POUR_BOWL) {
			double alpha = (sim_time - tilt_t0) / tilt_duration_sec;
			if (alpha > 1.0) {
				alpha = 1.0;
			}
			Quaterniond q0(ee_hover_R);
			Quaterniond q1(ee_tilt_R_end);
			const Matrix3d R_tilt =
				q0.slerp(alpha, q1).normalized().toRotationMatrix();

			const Vector3d ee_tilt_pos_goal =
				wrist_pivot_world + R_tilt * control_point;
			pose_task->setGoalPosition(ee_tilt_pos_goal);
			pose_task->setGoalOrientation(R_tilt);
			gripper_task->setGoalPosition(gripper_closed);

			N_prec.setIdentity();
			pose_task->updateTaskModel(N_prec);
			base_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
			gripper_task->updateTaskModel(
				base_task->getTaskAndPreviousNullspace());
			joint_task->updateTaskModel(
				gripper_task->getTaskAndPreviousNullspace());

			command_torques = pose_task->computeTorques() +
							  base_task->computeTorques() +
							  joint_task->computeTorques() +
							  gripper_task->computeTorques();

			if (alpha >= 1.0 && !tilt_complete_printed) {
				cout << "EE_POUR_BOWL: 90 deg tilt complete (hold)\n";
				tilt_complete_printed = true;
			}
		}
		// Publish pose-task EE goal for simviz / debugging (control point in world).
		Vector3d ee_goal_pos_world(0.0, 0.0, 0.0);
		Matrix3d ee_goal_R_world = Matrix3d::Identity();
		double ee_goal_active = 0.0;
		if (state == EE_ABOVE_BOWL) {
			ee_goal_active = 1.0;
			ee_goal_pos_world = ee_hover_world;
			ee_goal_R_world = ee_hover_R;
		} else if (state == EE_GRASP_BOWL) {
			ee_goal_active = 1.0;
			ee_goal_pos_world = grasp_position;
			ee_goal_R_world = ee_hover_R;
		} else if (state == EE_LIFT_BOWL) {
			ee_goal_active = 1.0;
			ee_goal_pos_world = ee_lift_world;
			ee_goal_R_world = ee_hover_R;
		} else if (state == EE_POUR_BOWL) {
			ee_goal_active = 1.0;
			double alpha_pub = (sim_time - tilt_t0) / tilt_duration_sec;
			if (alpha_pub > 1.0) {
				alpha_pub = 1.0;
			}
			Quaterniond q0p(ee_hover_R);
			Quaterniond q1p(ee_tilt_R_end);
			ee_goal_R_world = q0p.slerp(alpha_pub, q1p).normalized().toRotationMatrix();
			ee_goal_pos_world =
				wrist_pivot_world + ee_goal_R_world * control_point;
		}
		VectorXd ee_goal_R_flat(9);
		ee_goal_R_flat << ee_goal_R_world.col(0), ee_goal_R_world.col(1),
			ee_goal_R_world.col(2);
		VectorXd ee_goal_p(3);
		ee_goal_p << ee_goal_pos_world(0), ee_goal_pos_world(1),
			ee_goal_pos_world(2);
		redis_client.setEigen(EE_GOAL_POSITION_KEY, ee_goal_p);
		redis_client.setEigen(EE_GOAL_ROTATION_KEY, ee_goal_R_flat);
		VectorXd ee_goal_active_v(1);
		ee_goal_active_v(0) = ee_goal_active;
		redis_client.setEigen(EE_GOAL_ACTIVE_KEY, ee_goal_active_v);

		redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, command_torques);
	}

	timer.stop();
	cout << "\nSimulation loop timer stats:\n";
	timer.printInfoPostRun();
	redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, 0 * command_torques);

	return main_rc;
}
