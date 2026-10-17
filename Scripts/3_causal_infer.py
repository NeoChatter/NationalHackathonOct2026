import argparse
import csv
import math
import os
import time
import cv2
import numpy as np
import torch
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# Import model architecture and image crop helper from steps 1 & 2
import importlib.util

CAMERAS = ('camera_fl', 'camera_fr', 'camera_cl', 'camera_cr')
CAMERA_TOPICS = {f'/constructor0/sensor/{c}/compressed_image': i for i, c in enumerate(CAMERAS)}
IMG_W, IMG_H = 256, 128


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class CausalSE2EKF:
    """
    Causal Extended Kalman Filter on SE(2) [x, y, yaw]:
    - Predicts using Neural Visual Ego-Motion (dx_body, dy_body, dyaw) and learned uncertainty Q
    - Updates using Anisotropic Top-K Neural Place Retrieval (solves Long Straights & Night Glare)
    """

    def __init__(self, x0: float, y0: float, yaw0: float):
        self.state = np.array([x0, y0, yaw0], dtype=np.float64)
        self.P = np.diag([0.15**2, 0.15**2, math.radians(1.0)**2])

    def predict(self, dx_body: float, dy_body: float, dyaw: float, q_var: np.ndarray):
        x, y, yaw = self.state
        c, s = math.cos(yaw), math.sin(yaw)

        self.state[0] = x + c * dx_body - s * dy_body
        self.state[1] = y + s * dx_body + c * dy_body
        self.state[2] = wrap_angle(yaw + dyaw)

        F_jac = np.array([
            [1.0, 0.0, -s * dx_body - c * dy_body],
            [0.0, 1.0,  c * dx_body - s * dy_body],
            [0.0, 0.0,  1.0],
        ], dtype=np.float64)

        # Rotate body-frame process noise into world frame
        R_rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        Q_body = np.diag([
            max(float(q_var[0]), 0.05**2),
            max(float(q_var[1]), 0.02**2),
            max(float(q_var[2]), math.radians(0.3)**2),
        ])
        Q_world = R_rot @ Q_body @ R_rot.T
        self.P = F_jac @ self.P @ F_jac.T + Q_world

    def update(self, z_meas: np.ndarray, R_cov: np.ndarray):
        y_res = np.array([
            z_meas[0] - self.state[0],
            z_meas[1] - self.state[1],
            wrap_angle(z_meas[2] - self.state[2]),
        ], dtype=np.float64)

        S = self.P + R_cov
        K = self.P @ np.linalg.inv(S)
        self.state = self.state + K @ y_res
        self.state[2] = wrap_angle(self.state[2])
        self.P = (np.eye(3) - K) @ self.P


def run_causal_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_mod = load_module_from_path('train_mod', os.path.join(os.path.dirname(__file__), '2_train_model.py'))
    build_mod = load_module_from_path('build_mod', os.path.join(os.path.dirname(__file__), '1_build_dataset.py'))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = train_mod.RacingLocNet(embed_dim=256).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    db_embeddings = ckpt['db_embeddings'].to(device)  # (N_db, 256)
    db_poses = ckpt['db_poses'].cpu().numpy()         # (N_db, 3): [x, y, yaw]
    val_start_ns = ckpt.get('val_start_timestamp_ns', 0) if args.val_only else 0

    print(f'[INIT] Loaded RacingLocNet & {len(db_embeddings):,} Neural Map Keyframes on {device}.')
    if val_start_ns > 0:
        print(f'[VAL MODE] Evaluating only on held-out validation laps (timestamp_ns >= {val_start_ns})')

    ekf = None
    first_gt_pose = None
    prev_img_tensor = None
    prev_fl_stamp_ns = 0

    latest_cam_img = [None] * len(CAMERAS)
    latest_cam_stamp = [0] * len(CAMERAS)

    predictions = []
    timings_ms = []
    reloc_hits = 0

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or '.', exist_ok=True)

    topics = list(CAMERA_TOPICS.keys()) + ['/tf']

    with open(args.log, 'w', newline='') as log_f:
        log_writer = csv.writer(log_f)
        log_writer.writerow(['frame', 'timestamp_ns', 'mode', 'top1_sim', 'total_ms', 'x', 'y', 'yaw'])

        with open(args.mcap_path, 'rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, channel, _, msg in reader.iter_decoded_messages(topics=topics):
                topic = channel.topic

                # Capture ONLY the very first pose at the start of the run (allowed by rules!)
                if topic == '/tf':
                    if ekf is None:
                        for t in getattr(msg, 'transforms', []) or []:
                            if t.header.frame_id.lstrip('/') == 'map' and t.child_frame_id.lstrip('/') == 'constructor0/base_link':
                                s_ns = build_mod.stamp_to_ns(t.header.stamp)
                                if s_ns >= val_start_ns:
                                    p = t.transform.translation
                                    yaw = build_mod.yaw_from_quat(t.transform.rotation)
                                    first_gt_pose = (float(p.x), float(p.y), yaw)
                    continue

                cam_idx = CAMERA_TOPICS[topic]
                s_ns = build_mod.stamp_to_ns(msg.header.stamp)
                if s_ns < val_start_ns or not hasattr(msg, 'data') or not msg.data:
                    continue

                t_start = time.perf_counter()
                latest_cam_img[cam_idx] = build_mod.crop_and_resize(msg.data)
                latest_cam_stamp[cam_idx] = s_ns

                # Trigger causal 10 Hz inference on every camera_fl frame
                if cam_idx != 0:
                    continue

                # If any secondary camera hasn't arrived yet on frame 1, duplicate camera_fl
                stacked = np.stack(
                    [im if im is not None else latest_cam_img[0] for im in latest_cam_img], axis=0
                ).astype(np.float32) / 255.0
                stacked = (stacked - stacked.mean(axis=(1, 2), keepdims=True)) / (
                    stacked.std(axis=(1, 2), keepdims=True) + 1e-5
                )
                curr_img_tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)  # (1, 4, H, W)

                # Initialize EKF on the first frame of the run
                if ekf is None:
                    if first_gt_pose is not None:
                        ekf = CausalSE2EKF(*first_gt_pose)
                        print(f'[START] Initialized EKF from First Pose of Run: x={first_gt_pose[0]:.2f}, y={first_gt_pose[1]:.2f}, yaw={math.degrees(first_gt_pose[2]):.1f}deg')
                    else:
                        # Global neural relocalization fallback if /tf is completely absent
                        with torch.no_grad():
                            _, emb_init, _ = model.forward_single(curr_img_tensor)
                            sims = (emb_init @ db_embeddings.T).squeeze(0)
                            best_idx = int(torch.argmax(sims).item())
                            ekf = CausalSE2EKF(*db_poses[best_idx])

                mode = 'VO_ONLY'
                top1_sim = 0.0

                with torch.no_grad():
                    if prev_img_tensor is not None:
                        dt = max((s_ns - prev_fl_stamp_ns) * 1e-9, 0.01)
                        dt_offs = [(s_ns - (st if st > 0 else s_ns)) * 1e-9 for st in latest_cam_stamp]
                        time_feat = torch.tensor([[dt] + dt_offs], dtype=torch.float32, device=device)

                        emb_curr, _, delta_pred, log_var = model(prev_img_tensor, curr_img_tensor, time_feat)
                        dx_b, dy_b, dyaw = delta_pred[0].cpu().numpy()
                        q_var = np.exp(log_var[0].cpu().numpy())

                        # 1. Causal EKF Prediction Step (Visual Ego-Motion)
                        ekf.predict(float(dx_b), float(dy_b), float(dyaw), q_var)
                    else:
                        _, emb_curr, _ = model.forward_single(curr_img_tensor)

                    # 2. Causal EKF Update Step (Spatially-Gated Anisotropic Neural Map Retrieval)
                    dx_db = db_poses[:, 0] - ekf.state[0]
                    dy_db = db_poses[:, 1] - ekf.state[1]
                    dyaw_db = np.abs(np.arctan2(np.sin(db_poses[:, 2] - ekf.state[2]), np.cos(db_poses[:, 2] - ekf.state[2])))

                    gate_mask = ((dx_db**2 + dy_db**2) <= 35.0**2) & (dyaw_db <= math.radians(35.0))
                    gate_indices = np.nonzero(gate_mask)[0]

                    if len(gate_indices) >= 5:
                        gate_embs = db_embeddings[gate_indices]
                        sims = (emb_curr @ gate_embs.T).squeeze(0)
                        k = min(8, len(gate_indices))
                        top_vals, top_local = torch.topk(sims, k=k)
                        top1_sim = float(top_vals[0].item())

                        if top1_sim > 0.45:
                            weights = torch.softmax(top_vals * 15.0, dim=0).cpu().numpy()
                            matched_poses = db_poses[gate_indices[top_local.cpu().numpy()]]

                            # Weighted mean measurement (handling circular mean for yaw)
                            z_x = float(np.sum(weights * matched_poses[:, 0]))
                            z_y = float(np.sum(weights * matched_poses[:, 1]))
                            z_yaw = math.atan2(
                                float(np.sum(weights * np.sin(matched_poses[:, 2]))),
                                float(np.sum(weights * np.cos(matched_poses[:, 2]))),
                            )

                            # Anisotropic sample covariance (naturally wide along straightaways, tight laterally!)
                            diff_xy = matched_poses[:, :2] - np.array([[z_x, z_y]])
                            cov_xy = (diff_xy.T * weights) @ diff_xy + np.eye(2) * (0.25 / max(top1_sim, 0.2))**2
                            R_cov = np.eye(3, dtype=np.float64) * (math.radians(2.0)**2)
                            R_cov[:2, :2] = cov_xy

                            ekf.update(np.array([z_x, z_y, z_yaw]), R_cov)
                            mode = 'EKF_FUSED'
                            reloc_hits += 1

                prev_img_tensor = curr_img_tensor
                prev_fl_stamp_ns = s_ns

                total_ms = (time.perf_counter() - t_start) * 1000.0
                timings_ms.append(total_ms)

                x, y, yaw = float(ekf.state[0]), float(ekf.state[1]), float(ekf.state[2])
                predictions.append((s_ns, x, y, yaw))
                n_frames = len(predictions)

                log_writer.writerow([n_frames, s_ns, mode, f'{top1_sim:.3f}', f'{total_ms:.2f}', f'{x:.4f}', f'{y:.4f}', f'{yaw:.4f}'])

                if n_frames == 1 or n_frames % 100 == 0:
                    avg_ms = np.mean(timings_ms[-100:])
                    print(
                        f'[{n_frames:5d}] {mode:9s} | Latency: {total_ms:4.1f}ms (avg:{avg_ms:4.1f}ms) | '
                        f'Sim:{top1_sim:.2f} | Pose: x={x:7.2f}, y={y:7.2f}, yaw={math.degrees(yaw):6.1f}deg'
                    )

    with open(args.output, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['timestamp_ns', 'x', 'y', 'yaw'])
        for s_ns, x, y, yaw in predictions:
            writer.writerow([s_ns, f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}'])

    t_arr = np.asarray(timings_ms)
    print('\n' + '=' * 70)
    print(f'=== NEURAL EKF INFERENCE COMPLETE ({len(predictions)} frames) ===')
    print(f'Latency (ms/frame) : Mean={t_arr.mean():.2f}ms | P95={np.percentile(t_arr, 95):.2f}ms (< 100ms target!)')
    print(f'Saved submission   : {args.output}')
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mcap_path', help='Path to .mcap file')
    parser.add_argument('--checkpoint', default='checkpoints/racing_model.pt')
    parser.add_argument('--output', default='submission.csv')
    parser.add_argument('--log', default='inference_telemetry.csv')
    parser.add_argument('--val-only', action='store_true', help='Run only on the held-out 20% validation laps')
    args = parser.parse_args()
    run_causal_inference(args)