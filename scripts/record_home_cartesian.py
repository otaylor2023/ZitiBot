#!/usr/bin/env python3
"""
Record the current end-effector cartesian pose from the OpenSai cartesian controller.

This reads from the same Redis keys that the cartesian controller actually uses,
ensuring the recorded pose matches what controllers will command.

Usage:
    1. Place the robot in your desired "home" position using the cartesian controller
    2. Run: python scripts/record_home_cartesian.py
    3. Copy the printed ARM_HOME_POSITION and ARM_HOME_ORIENTATION into constants.py
"""

import sys
import time
import redis
import numpy as np
import json

def read_cartesian_pose_from_controller(redis_client, verbose=False):
    """
    Read current end-effector cartesian pose from the OpenSai cartesian controller.
    This uses the same Redis keys that controllers actually use.
    """
    pos_key = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position"
    ori_key = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_orientation"
    
    try:
        pos_data = redis_client.get(pos_key)
        ori_data = redis_client.get(ori_key)
        
        if pos_data is None:
            print(f"Warning: Redis key '{pos_key}' not found")
            return None, None
        if ori_data is None:
            print(f"Warning: Redis key '{ori_key}' not found")
            return None, None
        
        if isinstance(pos_data, bytes):
            pos_data = pos_data.decode("utf-8")
        if isinstance(ori_data, bytes):
            ori_data = ori_data.decode("utf-8")
        
        position = np.array(json.loads(pos_data), dtype=np.float64).reshape(3)
        orientation = np.array(json.loads(ori_data), dtype=np.float64).reshape(3, 3)
        
        if verbose:
            print(f"\n[DEBUG] Position from controller: {position}")
            print(f"[DEBUG] Orientation from controller:")
            print(orientation)
        
        return position, orientation
    except Exception as e:
        print(f"Error reading cartesian pose from controller: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    try:
        redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        redis_client.ping()
    except Exception as e:
        print(f"Error: Could not connect to Redis at localhost:6379")
        print(f"  Details: {e}")
        print(f"\nMake sure:")
        print(f"  1. Redis server is running: redis-server")
        print(f"  2. OpenSai arm controller is running: ./scripts/launch.sh config_folder/xml_config_files/picklebot.xml")
        return 1
    
    print("=" * 70)
    print("RECORDING HOME CARTESIAN POSE")
    print("=" * 70)
    print("\nPlace the robot in your desired 'home' position, then press Enter...")
    input()
    
    print("\nReading cartesian pose from the cartesian controller...")
    time.sleep(0.5)
    
    position, orientation = read_cartesian_pose_from_controller(redis_client, verbose=True)
    if position is None or orientation is None:
        print(f"Error: Could not read valid cartesian pose from controller")
        return 1
    
    print("\n" + "=" * 70)
    print("CURRENT CARTESIAN HOME POSE")
    print("=" * 70)
    
    print("\nPosition (for copy-paste into constants.py):")
    print("ARM_HOME_POSITION = np.array([")
    print(f"    {position[0]:12.8f},  # x")
    print(f"    {position[1]:12.8f},  # y")
    print(f"    {position[2]:12.8f},  # z")
    print("], dtype=np.float64)")
    
    print("\nOrientation 3×3 rotation matrix (for copy-paste into constants.py):")
    print("ARM_HOME_ORIENTATION = np.array([")
    for i, row in enumerate(orientation):
        print(f"    [{row[0]:+.8f}, {row[1]:+.8f}, {row[2]:+.8f}],")
    print("], dtype=np.float64)")
    
    print("\nMetrics for reference:")
    print(f"  Position XYZ: [{position[0]:+.4f}, {position[1]:+.4f}, {position[2]:+.4f}]")
    
    # Convert rotation matrix to Euler angles for reference
    from scipy.spatial.transform import Rotation as R
    rot = R.from_matrix(orientation)
    euler = rot.as_euler('xyz', degrees=True)
    print(f"  Orientation (Euler XYZ): [{euler[0]:+.2f}°, {euler[1]:+.2f}°, {euler[2]:+.2f}°]")
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Copy the 'Position' and 'Orientation' blocks above")
    print("  2. Open: zitibot_core/constants.py")
    print("  3. Replace ARM_HOME_POSITION and ARM_HOME_ORIENTATION with the new values")
    print("  4. Re-run your controller — it will move to this cartesian pose on startup")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
