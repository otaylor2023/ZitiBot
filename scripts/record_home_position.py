#!/usr/bin/env python3
"""
Record the current robot joint state and print it for use in constants.py

Usage:
    1. Place the robot in your desired "home" position
    2. Run: python scripts/record_home_position.py
    3. Copy the printed joint positions into zitibot_core/constants.py ARM_HOME_JOINT_POSITIONS
"""

import sys
import time
import redis
import numpy as np

def read_joint_positions(redis_client, key="opensai::sensors::FrankaRobot::joint_positions"):
    """Read current joint positions from Redis."""
    try:
        data = redis_client.get(key)
        if data is None:
            return None
        return np.array(eval(data), dtype=np.float64)
    except Exception as e:
        print(f"Error reading joint positions: {e}")
        return None

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
    print("RECORDING HOME JOINT POSITION")
    print("=" * 70)
    print("\nPlace the robot in your desired 'home' position, then press Enter...")
    input()
    
    print("\nReading joint positions...")
    time.sleep(0.5)
    
    q = read_joint_positions(redis_client)
    if q is None or len(q) != 7:
        print(f"Error: Could not read valid joint positions (got {q})")
        return 1
    
    print("\n" + "=" * 70)
    print("CURRENT JOINT CONFIGURATION")
    print("=" * 70)
    
    print("\nRadians (for copy-paste into constants.py):")
    print("ARM_HOME_JOINT_POSITIONS = np.array([")
    for i, angle in enumerate(q):
        print(f"    {angle:12.8f},  # q{i+1}")
    print("], dtype=np.float64)")
    
    print("\nDegrees (for reference):")
    q_deg = np.degrees(q)
    for i, angle in enumerate(q_deg):
        print(f"  q{i+1} = {angle:7.2f}°")
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Copy the 'Radians' block above")
    print("  2. Open: zitibot_core/constants.py")
    print("  3. Replace ARM_HOME_JOINT_POSITIONS with the new values")
    print("  4. Re-run your controller")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
