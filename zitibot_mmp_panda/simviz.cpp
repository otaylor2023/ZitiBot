/**
 * @file simviz.cpp
 * @brief Simulation and visualization of panda robot with 1 DOF gripper 
 * 
 */

#include <algorithm>
#include <math.h>
#include <signal.h>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <typeinfo>
#include <random>

#include "SaiGraphics.h"
#include "SaiModel.h"
#include "SaiSimulation.h"
#include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "logger/Logger.h"

bool fSimulationRunning = true;
void sighandler(int){fSimulationRunning = false;}

#include "redis_keys.h"

using namespace Eigen;
using namespace std;

// mutex and globals
VectorXd ui_torques;
mutex mutex_torques, mutex_update;

// specify urdf and robots 
static const string robot_name = "mmp_panda";
static const string camera_name = "camera_fixed";

// dynamic objects information
const vector<std::string> object_names = {"cup", "empty_bowl"};
static const int kEmptyBowlObjectIndex = 1;
vector<Affine3d> object_poses;
vector<VectorXd> object_velocities;
const int n_objects = object_names.size();

// simulation thread
void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> sim);

// Initial mobile base joints for mmp_panda: (0) prismatic X [m], (1) prismatic Y [m],
// (2) base yaw [rad]. Edit here — must stay within URDF limits (roughly ±2.9 m / rad).
static const Eigen::Vector3d kInitialBaseJoints(-1.0, 0.0, 0.0);

int main() {
	SaiModel::URDF_FOLDERS["CS225A_URDF_FOLDER"] = string(CS225A_URDF_FOLDER);
	static const string robot_file = string(CS225A_URDF_FOLDER) + "/mmp_panda/mmp_panda.urdf";
	static const string world_file = string(MMP_PANDA_FOLDER) + "/world_mmp_panda.urdf";
	std::cout << "Loading URDF world model file: " << world_file << endl;

	// start redis client
	auto redis_client = SaiCommon::RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load graphics scene
	auto graphics = std::make_shared<SaiGraphics::SaiGraphics>(world_file, camera_name, false);
	graphics->setBackgroundColor(66.0/255, 135.0/255, 245.0/255);  // set blue background 	
	// graphics->showLinkFrame(true, robot_name, "link7", 0.15);  // can add frames for different links
	// graphics->getCamera(camera_name)->setClippingPlanes(0.1, 50);  // set the near and far clipping planes 
	graphics->addUIForceInteraction(robot_name);
	cout << "Visualizer: press Q to quit; P to print robot state.\n";

	// load robots
	auto robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
	robot->updateModel();
	ui_torques = VectorXd::Zero(robot->dof());

	// Seed base pose before sim + Redis (see kInitialBaseJoints above).
	VectorXd q0 = robot->q();
	q0.head(3) = kInitialBaseJoints;
	const double pr_lo = -2.8973;
	const double pr_hi = 2.9873;
	const double yaw_lo = -2.8973;
	const double yaw_hi = 2.8973;
	q0(0) = std::clamp(q0(0), pr_lo, pr_hi);
	q0(1) = std::clamp(q0(1), pr_lo, pr_hi);
	q0(2) = std::clamp(q0(2), yaw_lo, yaw_hi);
	robot->setQ(q0);
	robot->setDq(VectorXd::Zero(robot->dof()));
	robot->updateModel();

	// load simulation world
	auto sim = std::make_shared<SaiSimulation::SaiSimulation>(world_file, false);
	sim->setJointPositions(robot_name, robot->q());
	sim->setJointVelocities(robot_name, robot->dq());

	// fill in object information 
	for (int i = 0; i < n_objects; ++i) {
		object_poses.push_back(sim->getObjectPose(object_names[i]));
		object_velocities.push_back(sim->getObjectVelocity(object_names[i]));
	}

	Vector3d bowl_t0 = object_poses[kEmptyBowlObjectIndex].translation();
	VectorXd bowl_xyz0(3);
	bowl_xyz0 << bowl_t0.x(), bowl_t0.y(), bowl_t0.z();
	redis_client.setEigen(BOWL_POSITION_KEY, bowl_xyz0);

    // set co-efficient of restition to zero for force control
    sim->setCollisionRestitution(0.0);

    // set co-efficient of friction
    sim->setCoeffFrictionStatic(0.0);
    sim->setCoeffFrictionDynamic(0.0);

	/*------- Set up visualization -------*/
	// init redis client values 
	redis_client.setEigen(JOINT_ANGLES_KEY, robot->q()); 
	redis_client.setEigen(JOINT_VELOCITIES_KEY, robot->dq()); 
	redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, 0 * robot->q());
	VectorXd ee_goal_R_I(9);
	ee_goal_R_I << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
	redis_client.setEigen(EE_GOAL_POSITION_KEY, VectorXd::Zero(3));
	redis_client.setEigen(EE_GOAL_ROTATION_KEY, ee_goal_R_I);
	VectorXd ee_goal_inactive(1);
	ee_goal_inactive(0) = 0.0;
	redis_client.setEigen(EE_GOAL_ACTIVE_KEY, ee_goal_inactive);

	// start simulation thread
	thread sim_thread(simulation, sim);

	// while window is open:
	bool p_key_was_down = false;
	while (graphics->isWindowOpen() && fSimulationRunning) {
		if (graphics->isKeyPressed(GLFW_KEY_Q)) {
			fSimulationRunning = false;
			break;
		}
		const bool p_down = graphics->isKeyPressed(GLFW_KEY_P);
		if (p_down && !p_key_was_down) {
			VectorXd q = redis_client.getEigen(JOINT_ANGLES_KEY);
			VectorXd dq = redis_client.getEigen(JOINT_VELOCITIES_KEY);
			robot->setQ(q);
			robot->setDq(dq);
			robot->updateModel();
			const Vector3d ee_in_link(0.0, 0.0, 0.07);
			const Vector3d x_ee = robot->position("link7", ee_in_link);
			const Matrix3d R_ee = robot->rotation("link7");
			cout << "\n--- robot state (P) ---\n";
			cout << "q  (" << q.size() << "): " << q.transpose() << "\n";
			cout << "dq (" << dq.size() << "): " << dq.transpose() << "\n";
			cout << "base (x_m, y_m, yaw_rad): " << q.head<3>().transpose() << "\n";
			cout << "link7 pos (world, m): " << x_ee.transpose() << "\n";
			cout << "link7 R (world<-link7):\n" << R_ee << "\n";
			VectorXd g_act = redis_client.getEigen(EE_GOAL_ACTIVE_KEY);
			VectorXd g_pos = redis_client.getEigen(EE_GOAL_POSITION_KEY);
			VectorXd g_Rflat = redis_client.getEigen(EE_GOAL_ROTATION_KEY);
			cout << "goal EE (pose task, control_point in world):\n";
			if (g_act.size() == 1 && g_act(0) > 0.5 && g_pos.size() == 3 &&
				g_Rflat.size() == 9) {
				Map<const Matrix3d> R_goal(g_Rflat.data());
				cout << "  active: yes\n";
				cout << "  pos (m): " << g_pos.transpose() << "\n";
				cout << "  R (world<-link7):\n" << R_goal << "\n";
			} else {
				cout << "  active: no (MOVE_BASE or controller not publishing)\n";
				if (g_pos.size() == 3) {
					cout << "  pos (stale, m): " << g_pos.transpose() << "\n";
				}
			}
			{
				lock_guard<mutex> lock(mutex_update);
				for (int i = 0; i < n_objects; ++i) {
					cout << "object \"" << object_names[i] << "\" T (xyz): "
						 << object_poses[i].translation().transpose() << "\n";
				}
			}
			cout << "---\n" << flush;
		}
		p_key_was_down = p_down;

        graphics->updateRobotGraphics(robot_name, redis_client.getEigen(JOINT_ANGLES_KEY));
		{
			lock_guard<mutex> lock(mutex_update);
			for (int i = 0; i < n_objects; ++i) {
				graphics->updateObjectGraphics(object_names[i], object_poses[i]);
			}
		}
		graphics->renderGraphicsWorld();
		{
			lock_guard<mutex> lock(mutex_torques);
			ui_torques = graphics->getUITorques(robot_name);
		}
	}

    // stop simulation
	fSimulationRunning = false;
	sim_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> sim) {
	// fSimulationRunning = true;

    // create redis client
    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

	// create a timer
	double sim_freq = 2000;
	SaiCommon::LoopTimer timer(sim_freq);

	sim->setTimestep(1.0 / sim_freq);
    sim->enableGravityCompensation(true);
	sim->enableJointLimits(robot_name);

	while (fSimulationRunning) {
		timer.waitForNextLoop();
		VectorXd control_torques = redis_client.getEigen(JOINT_TORQUES_COMMANDED_KEY);
		{
			lock_guard<mutex> lock(mutex_torques);
			sim->setJointTorques(robot_name, control_torques + ui_torques);
		}
		sim->integrate();
        redis_client.setEigen(JOINT_ANGLES_KEY, sim->getJointPositions(robot_name));
        redis_client.setEigen(JOINT_VELOCITIES_KEY, sim->getJointVelocities(robot_name));

		// update object information 
		{
			lock_guard<mutex> lock(mutex_update);
			for (int i = 0; i < n_objects; ++i) {
				object_poses[i] = sim->getObjectPose(object_names[i]);
				object_velocities[i] = sim->getObjectVelocity(object_names[i]);
			}
		}
		Vector3d bowl_t = object_poses[kEmptyBowlObjectIndex].translation();
		VectorXd bowl_xyz(3);
		bowl_xyz << bowl_t.x(), bowl_t.y(), bowl_t.z();
		redis_client.setEigen(BOWL_POSITION_KEY, bowl_xyz);
	}
	timer.stop();
	cout << "\nSimulation loop timer stats:\n";
	timer.printInfoPostRun();
}