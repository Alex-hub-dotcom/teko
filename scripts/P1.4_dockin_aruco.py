# SPDX-License-Identifier: BSD-3-Clause
"""
P1.4 – ArUco-based docking with comprehensive ground truth logging
-------------------------------------------------------------------

This script performs automatic docking using ArUco detection and saves
comprehensive ground truth data including:
- Final positions and orientations of both robots
- Distance metrics
- Success criteria
- Timestamp and metadata

This ground truth will be used as target poses for RL training.
"""

from isaaclab.app import AppLauncher
import time
import numpy as np
import torch
import cv2
import os
import json
from datetime import datetime

# ==== General configuration ====
HEADLESS = False
DESIRED_SECONDS = 30
STEPS = DESIRED_SECONDS * 120
PRINT_EVERY = 20

# ==== ArUco parameters ====
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 30
aruco_params.adaptiveThreshWinSizeStep = 3
aruco_params.adaptiveThreshConstant = 2.0
aruco_params.minMarkerPerimeterRate = 0.002
aruco_params.maxMarkerPerimeterRate = 4.0
aruco_params.minCornerDistanceRate = 0.01
aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.01
aruco_params.perspectiveRemovePixelPerCell = 6
aruco_params.errorCorrectionRate = 0.7
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ==== Camera calibration ====
MARKER_SIZE = 0.05  # meters
camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# ==== Docking thresholds ====
DOCKING_SUCCESS_DISTANCE = 0.50  # 1cm for success
DOCKING_ACCEPTABLE_DISTANCE = 0.02  # 2cm acceptable


def to_numpy_img(t):
    """Convert tensor/array to uint8 NumPy RGB image."""
    if t is None:
        raise RuntimeError("Camera frame returned None.")
    arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in [3, 4]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]  # Remove alpha channel
    if arr.max() <= 1.0 + 1e-6:
        arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


def quaternion_to_euler(quat):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in degrees."""
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.degrees([roll, pitch, yaw])


def get_goal_robot_pose(stage, env_idx=0):
    """Get the pose of the static goal robot from USD stage."""
    from pxr import UsdGeom  # Import after Isaac Sim initialized
    
    goal_path = f"/World/envs/env_{env_idx}/RobotGoal"
    goal_prim = stage.GetPrimAtPath(goal_path)
    
    if not goal_prim.IsValid():
        print(f"[WARN] Goal robot not found at {goal_path}")
        return None, None
    
    xformable = UsdGeom.Xformable(goal_prim)
    
    # Get local transform
    local_transform = xformable.GetLocalTransformation()
    
    # Extract translation
    translation = local_transform.ExtractTranslation()
    pos = np.array([translation[0], translation[1], translation[2]])
    
    # Extract rotation (as quaternion)
    rotation = local_transform.ExtractRotation()
    quat_gf = rotation.GetQuat()
    # USD quaternion is (real, i, j, k) = (w, x, y, z)
    quat = np.array([quat_gf.GetReal(), 
                     quat_gf.GetImaginary()[0],
                     quat_gf.GetImaginary()[1], 
                     quat_gf.GetImaginary()[2]])
    
    return pos, quat


def save_ground_truth(env, duration, success_distance):
    """Save comprehensive ground truth data."""
    from omni.usd import get_context  # Import after Isaac Sim initialized
    
    # Get active robot pose (position + quaternion)
    robot_state = env.robot.data.root_state_w[0].cpu().numpy()
    robot_pos = robot_state[:3]
    robot_quat = robot_state[3:7]  # w, x, y, z
    robot_euler = quaternion_to_euler(robot_quat)
    
    # Get goal robot pose from USD
    stage = get_context().get_stage()
    goal_pos, goal_quat = get_goal_robot_pose(stage, env_idx=0)
    
    if goal_pos is None:
        print("[ERROR] Could not retrieve goal robot pose!")
        goal_pos = np.array([np.nan, np.nan, np.nan])
        goal_quat = np.array([np.nan, np.nan, np.nan, np.nan])
        goal_euler = np.array([np.nan, np.nan, np.nan])
    else:
        goal_euler = quaternion_to_euler(goal_quat)
    
    # Calculate metrics
    distance = np.linalg.norm(robot_pos - goal_pos)
    success = distance <= success_distance
    
    # Prepare ground truth data
    ground_truth = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'success_threshold_m': success_distance,
            'docking_success': bool(success),
            'final_distance_m': float(distance)
        },
        'active_robot': {
            'position': {
                'x': float(robot_pos[0]),
                'y': float(robot_pos[1]),
                'z': float(robot_pos[2])
            },
            'orientation_quat': {
                'w': float(robot_quat[0]),
                'x': float(robot_quat[1]),
                'y': float(robot_quat[2]),
                'z': float(robot_quat[3])
            },
            'orientation_euler_deg': {
                'roll': float(robot_euler[0]),
                'pitch': float(robot_euler[1]),
                'yaw': float(robot_euler[2])
            }
        },
        'goal_robot': {
            'position': {
                'x': float(goal_pos[0]),
                'y': float(goal_pos[1]),
                'z': float(goal_pos[2])
            },
            'orientation_quat': {
                'w': float(goal_quat[0]),
                'x': float(goal_quat[1]),
                'y': float(goal_quat[2]),
                'z': float(goal_quat[3])
            },
            'orientation_euler_deg': {
                'roll': float(goal_euler[0]),
                'pitch': float(goal_euler[1]),
                'yaw': float(goal_euler[2])
            }
        }
    }
    
    # Create logs directory
    log_dir = "/workspace/teko/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save as JSON (comprehensive)
    json_path = os.path.join(log_dir, f"docking_groundtruth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    # Save as CSV (simple format for quick analysis)
    csv_path = os.path.join(log_dir, "docking_groundtruth_latest.csv")
    csv_data = np.array([
        robot_pos[0], robot_pos[1], robot_pos[2],
        goal_pos[0], goal_pos[1], goal_pos[2],
        distance, float(success)
    ])
    np.savetxt(
        csv_path,
        csv_data.reshape(1, -1),
        delimiter=",",
        header="robot_x,robot_y,robot_z,goal_x,goal_y,goal_z,distance_m,success",
        comments=""
    )
    
    # Print summary
    print("\n" + "="*70)
    print("GROUND TRUTH SUMMARY")
    print("="*70)
    print(f"Active Robot Position: [{robot_pos[0]:.4f}, {robot_pos[1]:.4f}, {robot_pos[2]:.4f}]")
    print(f"Goal Robot Position:   [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]")
    print(f"Final Distance:        {distance:.4f} m ({distance*100:.2f} cm)")
    print(f"Docking Success:       {'✓ YES' if success else '✗ NO'}")
    print(f"Duration:              {duration:.1f} seconds")
    print(f"\nSaved to:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV:  {csv_path}")
    print("="*70 + "\n")
    
    return ground_truth


def main():
    app = AppLauncher(headless=HEADLESS).app
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    cfg = TekoEnvCfg()
    env = TekoEnv(cfg=cfg)
    env.reset()

    # Get camera from multi-env setup (cameras is a list)
    cameras_list = getattr(env, "cameras", None)
    if cameras_list is None or len(cameras_list) == 0:
        raise RuntimeError("Cameras not found in environment.")
    
    # Use first camera (env_0)
    camera_obj = cameras_list[0]
    getter = getattr(camera_obj, "get_rgba", None)
    if getter is None:
        raise RuntimeError("Camera object has no 'get_rgba()' method.")

    print("[INFO] Starting ArUco-based docking with ground truth logging...")
    print(f"[INFO] Docking success threshold: {DOCKING_SUCCESS_DISTANCE*100:.1f} cm")
    t0 = time.time()
    device = getattr(env, "device", "cuda:0")

    # Control variables
    last_x_robot = None
    move_forward_steps = 0
    y_filtered = 0.0

    for step in range(1, STEPS + 1):
        frame = getter()  # Returns RGBA
        img = to_numpy_img(frame)  # Converts to RGB
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        vel_l = vel_r = 0.0

        if ids is not None and len(ids) > 0:
            rvecs, tvecs = [], []
            objp = np.array([
                [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
                [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
                [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
                [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
            ], dtype=np.float32)

            for c in corners:
                ok, rvec, tvec = cv2.solvePnP(objp, c[0], camera_matrix, dist_coeffs)
                if ok:
                    rvecs.append(rvec)
                    tvecs.append(tvec)

            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                x_cam, y_cam, z_cam = tvec.flatten()
                x_robot = z_cam
                y_robot = -x_cam
                z_robot = -y_cam
                last_x_robot = x_robot

                # Exponential smoothing
                y_filtered = 0.85 * y_filtered + 0.15 * y_robot

                rot_mat, _ = cv2.Rodrigues(rvec)
                yaw_rad = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
                yaw_deg = np.degrees(yaw_rad)

                # Motion control
                vel_forward = 0.3
                deadband = 0.001
                Kp = 1.8

                if abs(y_filtered) > deadband:
                    turn_speed = 0.12
                    vel_l = turn_speed if y_filtered > 0 else -turn_speed
                    vel_r = -turn_speed if y_filtered > 0 else turn_speed
                elif x_robot > 0.12:
                    vel_l = vel_r = vel_forward
                else:
                    vel_l = vel_r = 0.0

                if step % PRINT_EVERY == 0:
                    print(f"[ARUCO] ID={ids[i][0]} | x={x_robot:.4f}m, y={y_robot:.4f}m | "
                          f"yaw={yaw_deg:.1f}° | vL={vel_l:.2f}, vR={vel_r:.2f}")

        else:
            if last_x_robot is not None and last_x_robot <= 0.253:
                move_forward_steps = 20  # ~5cm forward
                print("[INFO] ArUco lost — performing final 5cm advance.")
            elif step % PRINT_EVERY == 0:
                print("[INFO] Searching for ArUco marker...")

        # Final forward motion
        if move_forward_steps > 0:
            vel_l = vel_r = 0.3
            move_forward_steps -= 1

             # Check if we just finished the final advance
            if move_forward_steps == 0:
                print("[INFO] Final advance completed! Finishing docking...")
                break  # Exit the loop early

        # Apply velocities (camera is rear-mounted, so invert)
        action = torch.tensor([[-vel_l, -vel_r]], device=device, dtype=torch.float32)
        env.step(action)
        app.update()

    # Save ground truth
    duration = time.time() - t0
    ground_truth = save_ground_truth(env, duration, DOCKING_SUCCESS_DISTANCE)
    
    print(f"[INFO] Docking sequence completed in {duration:.1f} seconds")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
