import json
import time

import numpy as np
import redis
from dataclasses import dataclass


@dataclass
class RedisKeys:
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
    active_controller: str = "opensai::controllers::FrankaRobot::active_controller_name"
    config_file_name: str = "::sai-interfaces-webui::config_file_name"


redis_keys = RedisKeys()

config_file_for_this_example = "zitibot_panda.xml"
controller_to_use = "cartesian_controller"

X_OFFSET_M = 0.02

# Home pose: EE at ~[0.6, -0.2, 0.35] m, tool z down, 45° yaw.
_HOME_YAW_RAD = np.radians(45.0)
_c, _s = np.cos(_HOME_YAW_RAD), np.sin(_HOME_YAW_RAD)
HOME_GOAL_POSITION = np.array([0.6, -0.2, 0.35])
HOME_GOAL_ORIENTATION = np.array(
    [
        [_c, -_s, 0.0],
        [-_s, -_c, 0.0],
        [0.0, 0.0, -1.0],
    ]
)

redis_client = redis.Redis()

config_file_name = redis_client.get(redis_keys.config_file_name).decode("utf-8")
if config_file_name != config_file_for_this_example:
    print(
        "This controller expects config file:",
        config_file_for_this_example,
        "but got:",
        config_file_name,
    )
    raise SystemExit(1)

while redis_client.get(redis_keys.active_controller).decode("utf-8") != controller_to_use:
    redis_client.set(redis_keys.active_controller, controller_to_use)

current_position = np.array(
    json.loads(redis_client.get(redis_keys.cartesian_task_current_position))
)
current_orientation = np.array(
    json.loads(redis_client.get(redis_keys.cartesian_task_current_orientation))
)
goal_pos = HOME_GOAL_POSITION.copy()
goal_ori = HOME_GOAL_ORIENTATION.copy()
print(
    f"start pos {current_position} ori=\n{current_orientation}\n"
    f"goal pos {goal_pos} ori=\n{goal_ori}"
)

loop_time = 0.0
dt = 0.01
init_time = time.perf_counter_ns() * 1e-9

try:
    while True:
        loop_time += dt
        time.sleep(max(0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time)))

        redis_client.set(
            redis_keys.cartesian_task_goal_position, json.dumps(goal_pos.tolist())
        )
        redis_client.set(
            redis_keys.cartesian_task_goal_orientation, json.dumps(goal_ori.tolist())
        )

        current_position = np.array(
            json.loads(redis_client.get(redis_keys.cartesian_task_current_position))
        )
        current_orientation = np.array(
            json.loads(redis_client.get(redis_keys.cartesian_task_current_orientation))
        )

        print(
            f"pos={current_position}  ori={current_orientation}",
            flush=True,
        )

except KeyboardInterrupt:
    print("Keyboard interrupt")
