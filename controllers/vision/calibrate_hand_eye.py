#!/usr/bin/env python3
"""
Interactive eye-in-hand calibration: RealSense + ArUco + Redis T_end_effector.

Measures T_FLANGE_CAMERA (camera optical -> Franka end-effector / flange) using
cv2.calibrateHandEye(). Uses the same T_end_effector Redis key as the live
ZitiBot pipeline (libfranka kEndEffector), so the result drops into:

    p_base = T_base_flange @ T_FLANGE_CAMERA @ p_camera

without a separate FK library.

Prerequisites
-------------
1. Redis + Franka driver running (publishing T_end_effector at ~1 kHz).
2. RealSense connected (no other process holding the camera).
3. Printed ArUco marker: DICT_4X4_100, marker ID matching --marker-id (default 0).
   Measure the **black square side length** in meters for --marker-length-m.

Operator procedure
------------------
1. Tape/mount the marker rigidly in the workspace (fixed in the world).
2. Run this script; a live UI window shows the camera feed with the detected
   marker outline + pose axes drawn when the target marker is in view.
3. Jog the arm via the OpenSai web UI to varied poses. At each pose: marker
   fully visible (green "marker OK"), arm settled, then press SPACE/ENTER.
4. Collect ~15 poses with **diverse wrist orientations** (rotation diversity
   matters more than translation diversity).
5. Keys in the window: SPACE/ENTER = capture, r = redo last, q/ESC = quit.

Output
------
* ``hand_eye_T_flange_camera.npy`` — 4x4 float64 homogeneous matrix.
* ``hand_eye_T_flange_camera.json`` — same matrix + metadata + residual.

Wire-in (manual, later)
-------------------------
Replace hardcoded ``T_FLANGE_CAMERA`` in:
* ``zitibot_tasks/gemini.py``
* ``grasp_and_pour_jar_controller.py``

Example::

    T_FLANGE_CAMERA = np.load(
        Path(__file__).resolve().parent / "hand_eye_T_flange_camera.npy"
    )

Usage
-----
::

    cd /home/tidybot01/OpenSai/ZitiBot/controllers/vision

    # Defaults: DICT_4X4_100, marker id 0, 100mm marker, 15 poses.
    python calibrate_hand_eye.py

    # Override any default as needed:
    python calibrate_hand_eye.py --marker-length-m 0.1 --num-poses 15 --dict DICT_4X4_100 --marker-id 0

Dependencies: redis, numpy, pyrealsense2, opencv-contrib-python
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import redis

# Sibling import when run as ``python calibrate_hand_eye.py`` from this dir.
_VISION_DIR = Path(__file__).resolve().parent
if str(_VISION_DIR) not in sys.path:
    sys.path.insert(0, str(_VISION_DIR))

from realsense_rgbd import next_rgbd_frame, start_realsense  # noqa: E402

# Current production placeholder (for diff report only).
T_FLANGE_CAMERA_HARDCODED_OLD = np.array(
    [
        [0.0, -1.0, 0.0, 0.053401],
        [1.0, 0.0, 0.0, -0.009],
        [0.0, 0.0, 1.0, 0.018930],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

OFFSET = np.array([0.053401, -0.009, 0.018930], dtype=np.float64)

ROT_Z = np.array( 
    [ np.cos(np.rad_to_deg(45)), -np.sin(np.rad_to_deg(45)), 0.0],
    [ np.sin(np.rad_to_deg(45)), np.cos(np.rad_to_deg(45)), 0.0],
    [ 0.0, 0.0, 1.0],
    dtype=np.float64,
)

TRANSLATION_Z = ROT_Z.T @ OFFSET

T_FLANGE_CAMERA_HARDCODED = np.eye(4, dtype=np.float64)
T_FLANGE_CAMERA_HARDCODED[:3, :3] = ROT_Z
T_FLANGE_CAMERA_HARDCODED[:3, 3] = TRANSLATION_Z

DEFAULT_EE_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_OUT = _VISION_DIR / "hand_eye_T_flange_camera.npy"
DEBUG_DIR = _VISION_DIR / "hand_eye_debug"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eye-in-hand calibration (RealSense + ArUco + Redis T_end_effector)."
    )
    p.add_argument(
        "--marker-length-m",
        type=float,
        default=0.1,
        help="ArUco marker black-square side length in meters (default 0.1 = 100mm).",
    )
    p.add_argument("--num-poses", type=int, default=15, help="Number of pose pairs to collect.")
    p.add_argument("--marker-id", type=int, default=0, help="ArUco marker ID to detect.")
    p.add_argument(
        "--dict",
        dest="aruco_dict",
        default="DICT_4X4_100",
        choices=[
            "DICT_4X4_50",
            "DICT_4X4_100",
            "DICT_4X4_250",
            "DICT_5X5_100",
            "DICT_6X6_250",
            "DICT_7X7_1000",
        ],
        help="ArUco dictionary name (must match printed marker).",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output .npy path for 4x4 transform.")
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--ee-key", default=DEFAULT_EE_KEY, help="Redis 4x4 JSON: base <- end-effector.")
    p.add_argument("--ee-samples", type=int, default=5, help="T_end_effector reads to average per pose.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cam-timeout-ms", type=int, default=2000)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument(
        "--debug-dir",
        type=Path,
        default=DEBUG_DIR,
        help="Directory for annotated capture images.",
    )
    return p.parse_args()


def connect_redis(host: str, port: int) -> redis.Redis:
    client = redis.Redis(host=host, port=port, decode_responses=False)
    client.ping()
    return client


def read_T_end_effector(redis_client: redis.Redis, key: str) -> np.ndarray | None:
    """Read 4x4 base<-EE transform from Redis (same format as arm.read_T_base_flange)."""
    try:
        raw = redis_client.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        T = np.array(json.loads(raw), dtype=np.float64)
        if T.size != 16:
            return None
        return T.reshape(4, 4)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    """Average translations; rotation via quaternion mean (Markley)."""
    if len(transforms) == 1:
        return transforms[0].copy()
    from scipy.spatial.transform import Rotation as R

    ts = np.stack([T[:3, 3] for T in transforms], axis=0)
    t_mean = ts.mean(axis=0)
    rots = R.from_matrix([T[:3, :3] for T in transforms])
    # scipy Rotation.mean available in recent scipy
    try:
        r_mean = rots.mean()
    except AttributeError:
        # Fallback: use first rotation if mean unavailable
        r_mean = rots[0]
    T_out = np.eye(4, dtype=np.float64)
    T_out[:3, :3] = r_mean.as_matrix()
    T_out[:3, 3] = t_mean
    return T_out


def read_T_ee_averaged(
    redis_client: redis.Redis,
    key: str,
    n_samples: int,
) -> np.ndarray | None:
    samples: list[np.ndarray] = []
    for _ in range(n_samples):
        T = read_T_end_effector(redis_client, key)
        if T is not None:
            samples.append(T)
        time.sleep(0.002)
    if not samples:
        return None
    return average_transforms(samples)


def intrinsics_to_opencv(color_intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """RealSense intrinsics -> (cameraMatrix 3x3, distCoeffs)."""
    K = np.array(
        [
            [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
            [0.0, color_intrinsics.fy, color_intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.array(color_intrinsics.coeffs, dtype=np.float64).reshape(-1)
    return K, dist


def make_aruco_detector(dict_name: str):
    dict_id = getattr(cv2.aruco, dict_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector, dictionary, parameters, True
    return None, dictionary, parameters, False


def marker_object_points(marker_length_m: float) -> np.ndarray:
    """4 corners in marker frame (Z out of marker plane), OpenCV ArUco order."""
    h = marker_length_m / 2.0
    return np.array(
        [
            [-h, h, 0.0],
            [h, h, 0.0],
            [h, -h, 0.0],
            [-h, -h, 0.0],
        ],
        dtype=np.float32,
    )


def detect_marker_pose(
    color_bgr: np.ndarray,
    *,
    detector,
    dictionary,
    parameters,
    use_aruco_detector: bool,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_id: int,
    marker_length_m: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Always returns ``(annotated_bgr, R_target2cam, t_target2cam)``.

    ``R``/``t`` are ``None`` when the target marker is not found (so the live
    UI can still draw every frame). When found, the annotated frame also has
    the marker outline and pose axes drawn on it.
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    if use_aruco_detector:
        corners, ids, _rejected = detector.detectMarkers(gray)
    else:
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )

    annotated = color_bgr.copy()
    if ids is None or len(ids) == 0:
        return annotated, None, None

    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

    target_idx = None
    for i, mid in enumerate(ids.flatten()):
        if int(mid) == marker_id:
            target_idx = i
            break
    if target_idx is None:
        return annotated, None, None

    corner = corners[target_idx]
    obj_pts = marker_object_points(marker_length_m)
    img_pts = corner.reshape(-1, 2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        # Fallback for older OpenCV / edge cases
        if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
            rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
                corner,
                marker_length_m,
                camera_matrix,
                dist_coeffs,
            )
            rvec = rvecs[0]
            tvec = tvecs[0]
        else:
            return annotated, None, None

    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)

    cv2.drawFrameAxes(
        annotated,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        marker_length_m * 0.5,
    )
    return annotated, R, t


def split_rt(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64).reshape(3, 1)
    return R, t


def assemble_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def ax_xb_residual(
    T_gripper2base_list: list[np.ndarray],
    T_target2cam_list: list[np.ndarray],
    T_cam2gripper: np.ndarray,
) -> float:
    """
    Eye-in-hand consistency: T_g2b @ X @ T_t2c should be constant across poses.
    Returns mean translation std (m) + mean rotation deviation (rad) heuristic.
    """
    composed = [
        T_g @ T_cam2gripper @ T_tc
        for T_g, T_tc in zip(T_gripper2base_list, T_target2cam_list)
    ]
    ref = composed[0]
    trans_errs = []
    rot_errs = []
    for T in composed[1:]:
        trans_errs.append(np.linalg.norm(T[:3, 3] - ref[:3, 3]))
        R_err = ref[:3, :3].T @ T[:3, :3]
        trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rot_errs.append(float(np.arccos(trace)))
    if not trans_errs:
        return 0.0
    return float(np.mean(trans_errs) + np.mean(rot_errs))


def rotation_diversity_warning(T_list: list[np.ndarray]) -> str | None:
    """Warn if EE rotations are too similar across poses."""
    from scipy.spatial.transform import Rotation as R

    rots = [R.from_matrix(T[:3, :3]) for T in T_list]
    if len(rots) < 3:
        return None
    # Max angle between any pair of orientations
    max_angle = 0.0
    for i in range(len(rots)):
        for j in range(i + 1, len(rots)):
            rel = rots[j].inv() * rots[i]
            ang = rel.magnitude()
            max_angle = max(max_angle, ang)
    if max_angle < np.radians(30.0):
        return (
            f"Low rotation diversity (max pairwise angle {np.degrees(max_angle):.1f}°). "
            "Jog to more varied wrist orientations for better calibration."
        )
    return None


def print_matrix_diff(T_new: np.ndarray, T_ref: np.ndarray, label: str) -> None:
    dt = np.linalg.norm(T_new[:3, 3] - T_ref[:3, 3])
    R_err = T_ref[:3, :3].T @ T_new[:3, :3]
    trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(trace)))
    print(f"\n{label}")
    print(f"  translation delta: {dt*1000:.2f} mm")
    print(f"  rotation delta:    {angle_deg:.2f} deg")
    print("  T_new:\n", T_new)
    print("  T_ref (hardcoded):\n", T_ref)


def run_calibration(
    R_g2b_list: list[np.ndarray],
    t_g2b_list: list[np.ndarray],
    R_t2c_list: list[np.ndarray],
    t_t2c_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_g2b_list,
        t_g2b_list,
        R_t2c_list,
        t_t2c_list,
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    T_cam2gripper = assemble_T(R_cam2gripper, t_cam2gripper)
    return R_cam2gripper, t_cam2gripper, T_cam2gripper


def main() -> int:
    args = parse_args()

    if args.num_poses < 3:
        print("Need at least 3 poses for calibrateHandEye.", file=sys.stderr)
        return 1

    print("=" * 70)
    print("HAND-EYE CALIBRATION (eye-in-hand)")
    print("=" * 70)
    print(f"Redis: {args.redis_host}:{args.redis_port}")
    print(f"EE key: {args.ee_key}")
    print(f"Marker: {args.aruco_dict} id={args.marker_id}  length={args.marker_length_m:.4f} m")

    try:
        redis_client = connect_redis(args.redis_host, args.redis_port)
    except Exception as e:
        print(f"Redis connect failed: {e}", file=sys.stderr)
        print("Start redis-server and the Franka OpenSai driver first.", file=sys.stderr)
        return 1

    tee = read_T_end_effector(redis_client, args.ee_key)
    if tee is None:
        print(f"WARNING: {args.ee_key!r} not in Redis yet.", file=sys.stderr)
        print("Start the Franka driver before capturing poses.", file=sys.stderr)
    else:
        print(f"T_end_effector OK (sample t={tee[:3, 3].tolist()})")

    pipeline = align = depth_scale = color_intrinsics = None
    try:
        pipeline, align, depth_scale, color_intrinsics = start_realsense(
            args.width,
            args.height,
            args.fps,
            args.warmup_frames,
            args.cam_timeout_ms,
        )
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"RealSense failed: {e}", file=sys.stderr)
        return 1

    camera_matrix, dist_coeffs = intrinsics_to_opencv(color_intrinsics)
    print(
        f"Intrinsics: fx={camera_matrix[0,0]:.2f} fy={camera_matrix[1,1]:.2f} "
        f"cx={camera_matrix[0,2]:.2f} cy={camera_matrix[1,2]:.2f}"
    )

    detector, dictionary, parameters, use_aruco_detector = make_aruco_detector(args.aruco_dict)

    T_g2b_full: list[np.ndarray] = []
    T_t2c_full: list[np.ndarray] = []
    R_g2b_list: list[np.ndarray] = []
    t_g2b_list: list[np.ndarray] = []
    R_t2c_list: list[np.ndarray] = []
    t_t2c_list: list[np.ndarray] = []

    win = "hand-eye calibration"
    cv2.namedWindow(win)

    try:
        miss_counter = [0]
        args.debug_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("CAPTURE (live UI window)")
        print("=" * 70)
        print(f"Target: {args.num_poses} poses | marker id={args.marker_id} | dict={args.aruco_dict}")
        print("In the window: SPACE/ENTER = capture | r = redo last | q/ESC = quit\n")

        aborted = False
        while len(R_g2b_list) < args.num_poses:
            triple = next_rgbd_frame(
                pipeline,
                align,
                depth_scale,
                args.cam_timeout_ms,
                miss_counter,
                max_misses=10,
            )
            if triple is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    aborted = True
                    break
                continue

            color_bgr, _depth_m, _depth_vis = triple
            annotated, R_tc, t_tc = detect_marker_pose(
                color_bgr,
                detector=detector,
                dictionary=dictionary,
                parameters=parameters,
                use_aruco_detector=use_aruco_detector,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                marker_id=args.marker_id,
                marker_length_m=args.marker_length_m,
            )

            found = R_tc is not None and t_tc is not None
            n_done = len(R_g2b_list)
            hud = annotated
            count_txt = f"pose {n_done}/{args.num_poses}"
            if found:
                status_txt = "marker OK - SPACE/ENTER to capture"
                status_color = (0, 255, 0)
            else:
                status_txt = f"marker id={args.marker_id} NOT found"
                status_color = (0, 0, 255)
            cv2.putText(hud, count_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(hud, status_txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, status_color, 2, cv2.LINE_AA)
            cv2.putText(hud, "SPACE/ENTER=capture  r=redo  q/ESC=quit", (10, hud.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, hud)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                aborted = True
                break

            if key == ord("r"):
                if R_g2b_list:
                    for lst in (
                        R_g2b_list,
                        t_g2b_list,
                        R_t2c_list,
                        t_t2c_list,
                        T_g2b_full,
                        T_t2c_full,
                    ):
                        lst.pop()
                    print("  Removed last sample.")
                else:
                    print("  Nothing to redo.")
                continue

            if key in (ord(" "), ord("\r"), 13, 10):
                if not found:
                    print(f"  Skipped: marker id={args.marker_id} not detected this frame.")
                    continue
                T_ee = read_T_ee_averaged(redis_client, args.ee_key, args.ee_samples)
                if T_ee is None:
                    print(f"  ERROR: Could not read {args.ee_key!r} from Redis.")
                    continue

                T_tc = assemble_T(R_tc, t_tc)
                R_g2b, t_g2b = split_rt(T_ee)
                R_g2b_list.append(R_g2b)
                t_g2b_list.append(t_g2b)
                R_t2c_list.append(R_tc)
                t_t2c_list.append(t_tc.reshape(3, 1))
                T_g2b_full.append(T_ee.copy())
                T_t2c_full.append(T_tc.copy())

                out_img = args.debug_dir / f"pose_{len(R_g2b_list):03d}.png"
                cv2.imwrite(str(out_img), annotated)
                print(
                    f"  OK pose {len(R_g2b_list)}/{args.num_poses}: "
                    f"t_gripper2base={t_g2b.ravel().tolist()}  saved {out_img.name}"
                )

        if aborted:
            print("\nAborted.")
            if len(R_g2b_list) < 3:
                return 130

    except KeyboardInterrupt:
        print("\nAborted.")
        return 130
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()

    if len(R_g2b_list) < 3:
        print("Too few samples collected.", file=sys.stderr)
        return 1

    warn = rotation_diversity_warning(T_g2b_full)
    if warn:
        print(f"\nWARNING: {warn}")

    print("\n" + "=" * 70)
    print("CALIBRATE")
    print("=" * 70)

    _R, _t, T_FLANGE_CAMERA = run_calibration(
        R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list
    )

    residual = ax_xb_residual(T_g2b_full, T_t2c_full, T_FLANGE_CAMERA)
    print(f"AX=XB consistency residual (lower is better): {residual:.6f}")

    print_matrix_diff(T_FLANGE_CAMERA, T_FLANGE_CAMERA_HARDCODED, "vs hardcoded T_FLANGE_CAMERA")

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, T_FLANGE_CAMERA)

    json_path = out_path.with_suffix(".json")
    meta = {
        "T_FLANGE_CAMERA": T_FLANGE_CAMERA.tolist(),
        "description": "camera optical -> flange/EE (eye-in-hand T_cam2gripper)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "marker_length_m": args.marker_length_m,
        "marker_id": args.marker_id,
        "aruco_dict": args.aruco_dict,
        "num_poses": len(R_g2b_list),
        "ee_key": args.ee_key,
        "ax_xb_residual": residual,
        "hardcoded_translation_delta_m": float(
            np.linalg.norm(T_FLANGE_CAMERA[:3, 3] - T_FLANGE_CAMERA_HARDCODED[:3, 3])
        ),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"Saved: {json_path}")
    print("\nNext: load this matrix into gemini.py / grasp_and_pour_jar_controller.py")
    print("      (see module docstring).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
