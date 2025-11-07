# SPDX-License-Identifier: BSD-3-Clause
"""
P1.4 – ArUco-based docking with comprehensive ground truth logging
- Robust steering (nonlinear lateral gain + bias torque)
- Lighting-robust ArUco (histogram equalization + blur)
- No auto-respawn (dones patched to False for this script)
"""

from isaaclab.app import AppLauncher
import time, os, json
import numpy as np
import torch
import cv2
from datetime import datetime

# ==== General configuration ====
HEADLESS = False
DESIRED_SECONDS = 45
STEPS = DESIRED_SECONDS * 120
PRINT_EVERY = 20
REAR_CAMERA = True           # camera on back: invert forward command
STOP_DIST = 0.12             # stop ~12 cm before marker plane

# Control gains / limits
KP_DIST = 1.2                # forward gain
KP_YAW  = 6.0                # lateral/yaw gain (nonlinear scaling below)
WHEELBASE_HALF = 0.22        # half wheelbase for diff-drive mixing (m-ish)
V_MAX = 0.9                  # cap normalized forward command
W_MAX = 1.2                  # cap normalized turn command
BIAS  = 0.12                 # extra push to overcome friction/static stiction

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

# ==== Camera calibration (approx) ====
MARKER_SIZE = 0.05  # m
camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0,   0,   1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# ==== Docking thresholds ====
DOCKING_SUCCESS_DISTANCE = 0.50  # 50 cm success condition for logging
DOCKING_ACCEPTABLE_DISTANCE = 0.02  # 2 cm acceptable (not used in controller)


def to_numpy_img(t):
    if t is None:
        raise RuntimeError("Camera frame returned None.")
    arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in [3, 4]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.max() <= 1.0 + 1e-6:
        arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


def quaternion_to_euler(quat):
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees([roll, pitch, yaw])


def get_goal_robot_pose(stage, env_idx=0):
    from pxr import UsdGeom
    goal_path = f"/World/envs/env_{env_idx}/RobotGoal"
    goal_prim = stage.GetPrimAtPath(goal_path)
    if not goal_prim.IsValid():
        print(f"[WARN] Goal robot not found at {goal_path}")
        return None, None
    xformable = UsdGeom.Xformable(goal_prim)
    local_transform = xformable.GetLocalTransformation()
    translation = local_transform.ExtractTranslation()
    pos = np.array([translation[0], translation[1], translation[2]])
    rotation = local_transform.ExtractRotation()
    quat_gf = rotation.GetQuat()  # (w, x, y, z) ordering in USD
    quat = np.array([quat_gf.GetReal(),
                     quat_gf.GetImaginary()[0],
                     quat_gf.GetImaginary()[1],
                     quat_gf.GetImaginary()[2]])
    return pos, quat


def save_ground_truth(env, duration, success_distance):
    from omni.usd import get_context
    robot_state = env.robot.data.root_state_w[0].cpu().numpy()
    robot_pos = robot_state[:3]
    robot_quat = robot_state[3:7]
    robot_euler = quaternion_to_euler(robot_quat)
    stage = get_context().get_stage()
    goal_pos, goal_quat = get_goal_robot_pose(stage, env_idx=0)
    if goal_pos is None:
        goal_pos = np.array([np.nan, np.nan, np.nan])
        goal_quat = np.array([np.nan, np.nan, np.nan, np.nan])
        goal_euler = np.array([np.nan, np.nan, np.nan])
    else:
        goal_euler = quaternion_to_euler(goal_quat)
    distance = np.linalg.norm(robot_pos - goal_pos)
    success = distance <= success_distance
    ground_truth = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': float(duration),
            'success_threshold_m': float(success_distance),
            'docking_success': bool(success),
            'final_distance_m': float(distance)
        },
        'active_robot': {
            'position': {'x': float(robot_pos[0]), 'y': float(robot_pos[1]), 'z': float(robot_pos[2])},
            'orientation_quat': {'w': float(robot_quat[0]), 'x': float(robot_quat[1]),
                                 'y': float(robot_quat[2]), 'z': float(robot_quat[3])},
            'orientation_euler_deg': {'roll': float(robot_euler[0]),
                                      'pitch': float(robot_euler[1]),
                                      'yaw': float(robot_euler[2])}
        },
        'goal_robot': {
            'position': {'x': float(goal_pos[0]), 'y': float(goal_pos[1]), 'z': float(goal_pos[2])},
            'orientation_quat': {'w': float(goal_quat[0]), 'x': float(goal_quat[1]),
                                 'y': float(goal_quat[2]), 'z': float(goal_quat[3])},
            'orientation_euler_deg': {'roll': float(goal_euler[0]),
                                      'pitch': float(goal_euler[1]),
                                      'yaw': float(goal_euler[2])}
        }
    }
    log_dir = "/workspace/teko/logs"
    os.makedirs(log_dir, exist_ok=True)
    json_path = os.path.join(log_dir, f"docking_groundtruth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    csv_path = os.path.join(log_dir, "docking_groundtruth_latest.csv")
    csv_data = np.array([
        robot_pos[0], robot_pos[1], robot_pos[2],
        goal_pos[0], goal_pos[1], goal_pos[2],
        distance, float(success)
    ])
    np.savetxt(csv_path, csv_data.reshape(1, -1), delimiter=",",
               header="robot_x,robot_y,robot_z,goal_x,goal_y,goal_z,distance_m,success", comments="")
    print("\n" + "="*70)
    print("GROUND TRUTH SUMMARY")
    print("="*70)
    print(f"Active Robot Position: [{robot_pos[0]:.4f}, {robot_pos[1]:.4f}, {robot_pos[2]:.4f}]")
    print(f"Goal Robot Position:   [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]")
    print(f"Final Distance:        {distance:.4f} m ({distance*100:.2f} cm)")
    print(f"Docking Success:       {'✓ YES' if success else '✗ NO'}")
    print(f"Duration:              {duration:.1f} seconds")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")
    print("="*70 + "\n")
    return ground_truth


def main():
    app = AppLauncher(headless=HEADLESS).app
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    cfg = TekoEnvCfg()
    cfg.episode_length_s = 10_000.0  # long episode; we'll end manually
    env = TekoEnv(cfg=cfg)
    env.reset()

    # --- disable auto-respawn (force dones=False) ---
    def _never_dones():
        n = env.scene.cfg.num_envs
        dev = env.device
        return (torch.zeros(n, dtype=torch.bool, device=dev),
                torch.zeros(n, dtype=torch.bool, device=dev))
    env._get_dones = _never_dones  # monkey patch for this script only

    # camera handle
    cameras_list = getattr(env, "cameras", None)
    if not cameras_list:
        raise RuntimeError("Cameras not found in environment.")
    getter = getattr(cameras_list[0], "get_rgba", None)
    if getter is None:
        raise RuntimeError("Camera object has no 'get_rgba()' method.")

    print("[INFO] Starting ArUco-based docking with ground truth logging...")
    print(f"[INFO] Docking success threshold: {DOCKING_SUCCESS_DISTANCE*100:.1f} cm")
    t0 = time.time()
    device = getattr(env, "device", "cuda:0")

    last_x_robot = None
    move_forward_steps = 0
    y_filtered = 0.0

    for step in range(1, STEPS + 1):
        frame = getter()
        img = to_numpy_img(frame)

        # lighting-robust preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        corners, ids, _ = detector.detectMarkers(gray)
        vL = vR = 0.0

        if ids is not None and len(ids) > 0:
            # pose from a single marker (first valid)
            c = corners[0]
            objp = np.array([[-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                             [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                             [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                             [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]], dtype=np.float32)
            ok, rvec, tvec = cv2.solvePnP(objp, c[0], camera_matrix, dist_coeffs)
            if ok:
                x_cam, y_cam, z_cam = tvec.flatten()
                # camera->robot frame mapping (keep your convention)
                x_robot = z_cam
                y_robot = -x_cam
                last_x_robot = x_robot

                # smooth lateral
                y_filtered = 0.85 * y_filtered + 0.15 * y_robot

                # velocity commands (normalized)
                forward_err = max(0.0, x_robot - STOP_DIST)
                lateral_err = y_filtered
                lateral_scaled = np.tanh(3.0 * lateral_err)  # nonlinear gain

                v = KP_DIST * forward_err
                w = KP_YAW  * lateral_scaled
                v = float(np.clip(v, 0.0, V_MAX))
                w = float(np.clip(w, -W_MAX, W_MAX))

                if REAR_CAMERA:
                    v = -v

                vL_cmd = v - WHEELBASE_HALF * w
                vR_cmd = v + WHEELBASE_HALF * w
                max_cmd = max(1e-6, np.max(np.abs([vL_cmd, vR_cmd])))
                vL = vL_cmd / max_cmd
                vR = vR_cmd / max_cmd

                # bias torque to overcome friction
                vL = float(np.clip(vL + np.sign(vL) * BIAS, -1.0, 1.0))
                vR = float(np.clip(vR + np.sign(vR) * BIAS, -1.0, 1.0))

                if x_robot <= STOP_DIST:
                    vL = vR = 0.0

                if step % PRINT_EVERY == 0:
                    # yaw just for logging
                    rot_mat, _ = cv2.Rodrigues(rvec)
                    yaw_rad = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
                    yaw_deg = np.degrees(yaw_rad)
                    print(f"[ARUCO] ID={int(ids[0][0])} | x={x_robot:.4f}m, y={y_robot:.4f}m | "
                          f"yaw={yaw_deg:.1f}° | vL={vL:.2f}, vR={vR:.2f}")
        else:
            # small final nudge when close and marker lost
            if last_x_robot is not None and last_x_robot <= (STOP_DIST + 0.13):
                move_forward_steps = 20
                if step % PRINT_EVERY == 0:
                    print("[INFO] ArUco lost — performing final 5cm advance.")
            elif step % PRINT_EVERY == 0:
                print("[INFO] Searching for ArUco marker...")

        if move_forward_steps > 0:
            vL = vR = ( -0.3 if REAR_CAMERA else 0.3 )
            move_forward_steps -= 1
            if move_forward_steps == 0:
                print("[INFO] Final advance completed! Ending loop.")
                break

        action = torch.tensor([[vL, vR]], device=device, dtype=torch.float32)
        env.step(action)
        app.update()

    duration = time.time() - t0
    save_ground_truth(env, duration, DOCKING_SUCCESS_DISTANCE)
    print(f"[INFO] Docking sequence completed in {duration:.1f} seconds")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
