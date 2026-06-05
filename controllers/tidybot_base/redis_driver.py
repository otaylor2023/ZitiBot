""" 
    Redis driver for base control
"""
import redis 
import numpy as np 
import ast 
import time 
from base_controller import Vehicle

"""
    Custom redis keys 
"""
BASE_POSE_KEY = "hb1::current_pose"
BASE_VEL_KEY = "hb1::current_vel"
DESIRED_BASE_POSE_KEY = "hb1::desired_pose"
STOP_BASE_KEY = "hb1::stop"
KILL_BASE_KEY = "hb1::kill"
# Live scalar that controllers (see ``zitibot_core.base.go_to_pose``)
# write to bump or cut the OTG max_velocity without restarting this
# driver. Clamped to [MAX_VEL_SCALE_MIN, MAX_VEL_SCALE_MAX] below.
# Missing / unparseable key = 1.0 (use the CLI baseline as-is).
MAX_VEL_SCALE_KEY = "hb1::max_vel_scale"
MAX_VEL_SCALE_MIN = 0.1
MAX_VEL_SCALE_MAX = 3.0
MAX_VEL_SCALE_CHANGE_EPS = 1e-3

"""
    Functions to read and write from redis 
    Redis keys are in format "[x, y, z]" for the pose 
"""
def numpy_array_to_string(array):
    """Converts a NumPy array to a string in the format "[x, y, z]".

    Args:
        array: A NumPy array (1-dimensional).

    Returns:
        A string representation of the array in the format "[x, y, z]".
        Returns an empty string if the input is not a 1D NumPy array.
    """
    if isinstance(array, np.ndarray) and array.ndim == 1:
        elements_str = ", ".join(map(str, array))
        return f"[{elements_str}]"
    else:
        return ""
    
def string_to_numpy_array(string):
    """Converts a string in the format "[x, y, z]" to a NumPy array.

    Args:
    string: A string representing a list of numbers in the format "[x, y, z]".

    Returns:
    A 1-dimensional NumPy array containing the numbers from the string.
    Returns None if the input string is not in the correct format.
    """
    if not isinstance(string, str):
        return None

    if not (string.startswith('[') and string.endswith(']')):
        return None

    try:
        # Use ast.literal_eval to safely evaluate the string as a Python literal
        list_representation = ast.literal_eval(string)
        if isinstance(list_representation, list):
            return np.array(list_representation)
        else:
            return None
    except (SyntaxError, ValueError):
        return None

if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description="TidyBot base Redis driver")
    ap.add_argument(
        "--max-vel-xy",
        type=float,
        default=0.25,
        help="Max planar speed (m/s) for x and y",
    )
    ap.add_argument(
        "--max-vel-yaw",
        type=float,
        default=0.79,
        help="Max yaw rate (rad/s)",
    )
    ap.add_argument(
        "--max-accel-xy",
        type=float,
        default=0.1,
        help="Max planar acceleration (m/s^2)",
    )
    ap.add_argument(
        "--max-accel-yaw",
        type=float,
        default=0.5,
        help="Max yaw acceleration (rad/s^2)",
    )
    driver_args = ap.parse_args()

    max_vel = (driver_args.max_vel_xy, driver_args.max_vel_xy, driver_args.max_vel_yaw)
    max_accel = (
        driver_args.max_accel_xy,
        driver_args.max_accel_xy,
        driver_args.max_accel_yaw,
    )
    print(f"Vehicle limits: max_vel={max_vel}  max_accel={max_accel}")

    vehicle = Vehicle(max_vel=max_vel, max_accel=max_accel)
    vehicle.start_control()
    time.sleep(1)  # Wait for initialization to finish 

    # Redis server
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.set(BASE_POSE_KEY, numpy_array_to_string(vehicle.x))
    redis_client.set(BASE_VEL_KEY, numpy_array_to_string(vehicle.dx))
    redis_client.set(DESIRED_BASE_POSE_KEY, numpy_array_to_string(vehicle.x))
    redis_client.set(STOP_BASE_KEY, 'ok')
    redis_client.set(KILL_BASE_KEY, 'ok')

    # Logging
    prev_goal_pose = vehicle.x

    # OTG max_velocity scaling — see MAX_VEL_SCALE_KEY above. Cache the
    # last applied scale so we only mutate / log when it actually changes.
    baseline_max_vel = np.array(vehicle.max_vel, dtype=np.float64).copy()
    redis_client.set(MAX_VEL_SCALE_KEY, "1.0")
    current_scale = 1.0
    print(f"Vehicle max_vel scale: {current_scale:.3f} (baseline {baseline_max_vel.tolist()})")

    def _read_max_vel_scale():
        raw = redis_client.get(MAX_VEL_SCALE_KEY)
        if raw is None:
            return 1.0
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val):
            return None
        return float(np.clip(val, MAX_VEL_SCALE_MIN, MAX_VEL_SCALE_MAX))

    # Main loop for reading and writing to redis
    try:
        while True:

            # Apply any live max_vel rescale before consuming the next pose
            # goal so the OTG sees the new limit on this tick. Mutating
            # otg_inp.max_velocity in place is safe with Ruckig — limits
            # are not copied back by ``otg_out.pass_to_input``.
            new_scale = _read_max_vel_scale()
            if new_scale is not None and abs(new_scale - current_scale) > MAX_VEL_SCALE_CHANGE_EPS:
                scaled = baseline_max_vel * new_scale
                vehicle.otg_inp.max_velocity = scaled
                print(
                    f"Vehicle max_vel scale: {current_scale:.3f} -> {new_scale:.3f} "
                    f"=> max_velocity {scaled.tolist()}"
                )
                current_scale = new_scale

            # Read stop signal
            if str(redis_client.get(STOP_BASE_KEY)) == 'stop':
                # print('Vehicle stopping')
                vehicle.set_target_velocity(np.zeros(3))
                redis_client.set(DESIRED_BASE_POSE_KEY, numpy_array_to_string(vehicle.x))
                if (np.linalg.norm(vehicle.dx) < 0.001):
                    print('Vehicle stopped; proceeding to pose control')
                    redis_client.set(STOP_BASE_KEY, 'ok')
                    continue
                else:
                    print('Vehicle stopping')
                    continue
            elif str(redis_client.get(KILL_BASE_KEY)) == 'kill':
                print('Vehicle killed')
                vehicle.stop_control()
            else:
                # Read and set goal position if it's different than the previous action 
                current_goal_pose = string_to_numpy_array(str(redis_client.get(DESIRED_BASE_POSE_KEY)))
                # print(redis_client.get(DESIRED_BASE_POSE_KEY))

                if not np.array_equal(current_goal_pose, prev_goal_pose):
                    print('Moving to ', current_goal_pose)
                    prev_goal_pose = current_goal_pose
                vehicle.set_target_position(current_goal_pose)

                # Write current position
                current_pose = vehicle.x
                current_velocity = vehicle.dx 
                redis_client.set(BASE_POSE_KEY, numpy_array_to_string(current_pose))
                redis_client.set(BASE_VEL_KEY, numpy_array_to_string(current_velocity))

            time.sleep(0.01)            
    finally:
        print('Vehicle stopped')
        vehicle.stop_control()
