#!/usr/bin/env python3
"""
Causal camera localization with sweep-selected filter settings.

Default settings:
    q_x_floor = 1.0 m per step
    r_scale   = 0.03
    gate_m    = 35.0
    gate_deg  = 35.0
    min_sim   = 0.45

Compatible with the existing RacingLocNet checkpoint.

--val-only is a RESET validation experiment:
it may initialize from the first eligible validation GT pose.

Without --val-only, only the first eligible GT pose of the recording
is captured for initialization. No subsequent GT poses enter the estimator.
"""

import argparse
import csv
import importlib.util
import math
import os
import time
from collections import deque

import numpy as np
import torch

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class CausalSE2EKF:
    """
    State: [x, y, yaw].

    The sweep-selected Q/R adjustments are applied by the inference loop,
    not hidden inside this class. This keeps the diagnostics compatible.
    """

    def __init__(self, x0, y0, yaw0):
        self.state = np.array([x0, y0, yaw0], dtype=np.float64)
        self.P = np.diag([
            0.15 ** 2,
            0.15 ** 2,
            math.radians(1.0) ** 2,
        ])

    def predict(self, dx_body, dy_body, dyaw, q_var):
        x, y, yaw = self.state
        c, s = math.cos(yaw), math.sin(yaw)

        self.state[0] = x + c * dx_body - s * dy_body
        self.state[1] = y + s * dx_body + c * dy_body
        self.state[2] = wrap_angle(yaw + dyaw)

        F = np.array([
            [1.0, 0.0, -s * dx_body - c * dy_body],
            [0.0, 1.0,  c * dx_body - s * dy_body],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        rotation = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        Q_body = np.diag([
            max(float(q_var[0]), 0.05 ** 2),
            max(float(q_var[1]), 0.02 ** 2),
            max(float(q_var[2]), math.radians(0.3) ** 2),
        ])

        self.P = (
            F @ self.P @ F.T
            + rotation @ Q_body @ rotation.T
        )
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, measurement, R):
        residual = np.array([
            measurement[0] - self.state[0],
            measurement[1] - self.state[1],
            wrap_angle(measurement[2] - self.state[2]),
        ], dtype=np.float64)

        S = self.P + R

        # Equivalent to P @ inv(S), without explicitly taking an inverse.
        K = np.linalg.solve(S.T, self.P.T).T

        self.state += K @ residual
        self.state[2] = wrap_angle(self.state[2])

        # Joseph covariance update for numerical stability.
        I_K = np.eye(3) - K
        self.P = I_K @ self.P @ I_K.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)


def load_model(train_module, checkpoint, device):
    """
    Construct the original architecture without downloading ImageNet weights.
    All model parameters are subsequently loaded from the checkpoint.
    """
    original_resnet18 = train_module.models.resnet18

    def resnet18_without_download(*args, **kwargs):
        kwargs["weights"] = None
        return original_resnet18(*args, **kwargs)

    train_module.models.resnet18 = resnet18_without_download
    try:
        model = train_module.RacingLocNet(embed_dim=256)
    finally:
        train_module.models.resnet18 = original_resnet18

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def select_causal_image(buffer, anchor_ns):
    """Newest encountered image whose exposure stamp is <= anchor_ns."""
    best_stamp = None
    best_image = None

    for stamp, image in buffer:
        if stamp <= anchor_ns:
            if best_stamp is None or stamp > best_stamp:
                best_stamp = stamp
                best_image = image

    return best_stamp, best_image


def run_causal_inference(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    here = os.path.dirname(os.path.abspath(__file__))

    train_module = load_module_from_path(
        "train_mod", os.path.join(here, "2_train_model.py")
    )
    build_module = load_module_from_path(
        "build_mod", os.path.join(here, "1_build_dataset.py")
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model = load_model(train_module, checkpoint, device)

    db_embeddings = checkpoint["db_embeddings"].to(
        device=device, dtype=torch.float32
    )
    db_poses = (
        checkpoint["db_poses"].cpu().numpy().astype(np.float64)
    )

    if len(db_poses) == 0:
        raise ValueError("The checkpoint contains an empty reference map.")
    if len(db_poses) != len(db_embeddings):
        raise ValueError("Reference poses and embeddings have different lengths.")

    val_start_ns = 0
    if args.val_only:
        val_start_ns = int(
            checkpoint.get("val_start_timestamp_ns", 0)
        )
        if val_start_ns <= 0:
            raise ValueError(
                "--val-only requires a checkpoint with a held-out split."
            )

    camera_topics = build_module.CAMERA_TOPICS
    camera_count = len(build_module.CAMERAS)

    # Retain older images so a future-stamped arrival cannot overwrite
    # the most recent image eligible for the current anchor.
    buffers = [
        deque(maxlen=64) for _ in range(camera_count)
    ]

    first_gt_pose = None
    first_gt_stamp = None
    ekf = None
    previous_image = None
    previous_stamp = None

    n_frames = 0
    n_updates = 0
    n_skipped_stamps = 0
    timings_ms = []

    gate_radius_sq = args.gate_m ** 2
    gate_angle = math.radians(args.gate_deg)
    loss_weights = np.array([1.0, 2.0, 5.0], dtype=np.float64)

    for path in (args.output, args.log):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    telemetry_fields = [
        "frame", "timestamp_ns", "mode", "top1_sim",
        "anchor_ms", "dt", "max_camera_age_s", "missing_cameras",
        "dx_body_pred", "dy_body_pred", "dyaw_pred",
        "q_x", "q_y", "q_yaw",
        "pred_x", "pred_y", "pred_yaw",
        "gate_n",
        "retrieved_x", "retrieved_y", "retrieved_yaw",
        "innovation_x", "innovation_y", "innovation_yaw", "nis",
        "R_xx", "R_yy", "R_yawyaw",
        "P_xx", "P_yy", "P_yawyaw",
        "x", "y", "yaw",
    ]

    print(f"[INIT] Device: {device}")
    print(f"[INIT] Reference frames: {len(db_poses):,}")
    print(
        f"[CONFIG] q_x_floor={args.q_x_floor:g}, "
        f"r_scale={args.r_scale:g}, "
        f"gate={args.gate_m:g}m/{args.gate_deg:g}deg, "
        f"min_sim={args.min_sim:g}, "
        f"fix_q_scaling={args.fix_q_scaling}"
    )

    if args.val_only:
        print(
            "[VAL] Reset validation experiment; "
            f"start timestamp = {val_start_ns}"
        )

    topics = list(camera_topics) + ["/tf"]

    with (
        open(args.output, "w", newline="") as output_file,
        open(args.log, "w", newline="") as log_file,
        open(args.mcap_path, "rb") as bag_file,
        torch.inference_mode(),
    ):
        output_writer = csv.writer(output_file)
        output_writer.writerow(["timestamp_ns", "x", "y", "yaw"])

        log_writer = csv.DictWriter(
            log_file, fieldnames=telemetry_fields
        )
        log_writer.writeheader()

        reader = make_reader(
            bag_file, decoder_factories=[DecoderFactory()]
        )

        for _, channel, _, msg in reader.iter_decoded_messages(
            topics=topics
        ):
            topic = channel.topic

            # Capture once, never repeatedly overwrite the initial pose.
            if topic == "/tf":
                if ekf is None and first_gt_pose is None:
                    for transform in (
                        getattr(msg, "transforms", []) or []
                    ):
                        parent = transform.header.frame_id.lstrip("/")
                        child = transform.child_frame_id.lstrip("/")

                        if (
                            parent != "map"
                            or child != "constructor0/base_link"
                        ):
                            continue

                        stamp = build_module.stamp_to_ns(
                            transform.header.stamp
                        )
                        if stamp < val_start_ns:
                            continue

                        position = transform.transform.translation
                        yaw = build_module.yaw_from_quat(
                            transform.transform.rotation
                        )
                        first_gt_pose = (
                            float(position.x),
                            float(position.y),
                            float(yaw),
                        )
                        first_gt_stamp = stamp
                        break
                continue

            if not hasattr(msg, "data") or len(msg.data) == 0:
                continue

            camera_idx = camera_topics[topic]
            stamp_ns = build_module.stamp_to_ns(msg.header.stamp)

            start_time = time.perf_counter()
            image = build_module.crop_and_resize(msg.data)
            buffers[camera_idx].append((stamp_ns, image))

            # Output is anchored to camera_fl.
            if camera_idx != 0 or stamp_ns < val_start_ns:
                continue

            if previous_stamp is not None and stamp_ns <= previous_stamp:
                n_skipped_stamps += 1
                continue

            selected_images = []
            offsets = []
            missing_cameras = 0

            for camera_buffer in buffers:
                selected_stamp, selected_image = select_causal_image(
                    camera_buffer, stamp_ns
                )

                if selected_image is None:
                    # Same startup fallback as before, but never use
                    # a future-stamped secondary camera image.
                    selected_image = image
                    selected_stamp = stamp_ns
                    missing_cameras += 1

                selected_images.append(selected_image)
                offsets.append((stamp_ns - selected_stamp) * 1e-9)

            stacked = (
                np.stack(selected_images, axis=0).astype(np.float32)
                / 255.0
            )
            stacked = (
                stacked - stacked.mean(axis=(1, 2), keepdims=True)
            ) / (
                stacked.std(axis=(1, 2), keepdims=True) + 1e-5
            )
            current_image = (
                torch.from_numpy(stacked).unsqueeze(0).to(device)
            )

            record = {
                name: float("nan") for name in telemetry_fields
            }
            record.update(
                timestamp_ns=int(stamp_ns),
                mode="VO_ONLY",
                top1_sim=0.0,
                max_camera_age_s=float(max(offsets)),
                missing_cameras=missing_cameras,
            )

            delta = None
            q_var = None

            if previous_image is None:
                _, embedding, _ = model.forward_single(current_image)
            else:
                dt = max(
                    (stamp_ns - previous_stamp) * 1e-9, 0.01
                )
                time_features = torch.tensor(
                    [[dt] + offsets],
                    dtype=torch.float32,
                    device=device,
                )

                embedding, _, delta_tensor, log_var = model(
                    previous_image, current_image, time_features
                )
                delta = (
                    delta_tensor[0].cpu().numpy().astype(np.float64)
                )
                q_var = np.exp(
                    log_var[0].cpu().numpy()
                ).astype(np.float64)

                # OFF by default: the supplied sweep did not use it.
                if args.fix_q_scaling:
                    q_var /= loss_weights

                # Tested change 1: longitudinal process-noise floor.
                q_var[0] = max(
                    float(q_var[0]), args.q_x_floor ** 2
                )

                record.update(
                    dt=dt,
                    dx_body_pred=float(delta[0]),
                    dy_body_pred=float(delta[1]),
                    dyaw_pred=float(delta[2]),
                    q_x=float(q_var[0]),
                    q_y=float(q_var[1]),
                    q_yaw=float(q_var[2]),
                )

            if ekf is None:
                if (
                    first_gt_pose is not None
                    and first_gt_stamp <= stamp_ns
                ):
                    ekf = CausalSE2EKF(*first_gt_pose)
                    print("[START] Initialized from the captured initial pose.")
                else:
                    similarities = (
                        embedding @ db_embeddings.T
                    ).squeeze(0)
                    best_idx = int(torch.argmax(similarities).item())
                    ekf = CausalSE2EKF(*db_poses[best_idx])
                    print("[START] Initialized by global image retrieval.")

            if delta is not None:
                ekf.predict(
                    float(delta[0]),
                    float(delta[1]),
                    float(delta[2]),
                    q_var,
                )

            record.update(
                pred_x=float(ekf.state[0]),
                pred_y=float(ekf.state[1]),
                pred_yaw=float(ekf.state[2]),
            )

            dx = db_poses[:, 0] - ekf.state[0]
            dy = db_poses[:, 1] - ekf.state[1]
            heading_difference = np.abs(np.arctan2(
                np.sin(db_poses[:, 2] - ekf.state[2]),
                np.cos(db_poses[:, 2] - ekf.state[2]),
            ))

            gate_mask = (
                (dx * dx + dy * dy <= gate_radius_sq)
                & (heading_difference <= gate_angle)
            )
            gate_indices = np.flatnonzero(gate_mask)
            record["gate_n"] = int(len(gate_indices))

            if len(gate_indices) >= 5:
                gate_tensor = torch.as_tensor(
                    gate_indices, dtype=torch.long, device=device
                )
                gate_embeddings = db_embeddings.index_select(
                    0, gate_tensor
                )
                similarities = (
                    embedding @ gate_embeddings.T
                ).squeeze(0)

                k = min(8, len(gate_indices))
                top_values, top_local = torch.topk(
                    similarities, k=k
                )
                top1_sim = float(top_values[0].item())
                record["top1_sim"] = top1_sim

                if top1_sim > args.min_sim:
                    weights = (
                        torch.softmax(top_values * 15.0, dim=0)
                        .cpu().numpy().astype(np.float64)
                    )
                    # Normalize again after conversion for numerical hygiene.
                    weights /= weights.sum()

                    matched = db_poses[
                        gate_indices[top_local.cpu().numpy()]
                    ]

                    z_x = float(np.sum(weights * matched[:, 0]))
                    z_y = float(np.sum(weights * matched[:, 1]))
                    z_yaw = math.atan2(
                        float(np.sum(weights * np.sin(matched[:, 2]))),
                        float(np.sum(weights * np.cos(matched[:, 2]))),
                    )
                    measurement = np.array(
                        [z_x, z_y, z_yaw], dtype=np.float64
                    )

                    diff_xy = (
                        matched[:, :2] - np.array([[z_x, z_y]])
                    )
                    covariance_xy = (
                        (diff_xy.T * weights) @ diff_xy
                        + np.eye(2)
                        * (0.25 / max(top1_sim, 0.2)) ** 2
                    )

                    R = np.eye(3, dtype=np.float64) * (
                        math.radians(2.0) ** 2
                    )

                    # Tested change 2: scale POSITION covariance only.
                    # This matches the diagnostic sweep.
                    R[:2, :2] = covariance_xy * args.r_scale

                    innovation = measurement - ekf.state
                    innovation[2] = wrap_angle(innovation[2])
                    S = ekf.P + R
                    nis = float(
                        innovation @ np.linalg.solve(S, innovation)
                    )

                    record.update(
                        mode="EKF_FUSED",
                        retrieved_x=z_x,
                        retrieved_y=z_y,
                        retrieved_yaw=z_yaw,
                        innovation_x=float(innovation[0]),
                        innovation_y=float(innovation[1]),
                        innovation_yaw=float(innovation[2]),
                        nis=nis,
                        R_xx=float(R[0, 0]),
                        R_yy=float(R[1, 1]),
                        R_yawyaw=float(R[2, 2]),
                    )

                    ekf.update(measurement, R)
                    n_updates += 1

            previous_image = current_image
            previous_stamp = stamp_ns

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000.0
            timings_ms.append(elapsed_ms)

            n_frames += 1
            x, y, yaw = map(float, ekf.state)

            output_writer.writerow([
                int(stamp_ns),
                f"{x:.6f}",
                f"{y:.6f}",
                f"{yaw:.6f}",
            ])

            record.update(
                frame=n_frames,
                anchor_ms=elapsed_ms,
                P_xx=float(ekf.P[0, 0]),
                P_yy=float(ekf.P[1, 1]),
                P_yawyaw=float(ekf.P[2, 2]),
                x=x, y=y, yaw=yaw,
            )
            log_writer.writerow(record)

            if n_frames == 1 or n_frames % 100 == 0:
                print(
                    f"[{n_frames:5d}] {record['mode']:9s} | "
                    f"sim={record['top1_sim']:.3f} | "
                    f"gate={record['gate_n']:4d} | "
                    f"anchor={elapsed_ms:.1f}ms | "
                    f"x={x:.2f}, y={y:.2f}, "
                    f"yaw={math.degrees(yaw):.1f}deg"
                )
                output_file.flush()
                log_file.flush()

    if n_frames == 0:
        raise RuntimeError(
            "No predictions were produced. Check the bag and validation split."
        )

    timing_array = np.asarray(timings_ms)
    print("\n" + "=" * 70)
    print(f"[DONE] Wrote {n_frames} poses to {args.output}")
    print(f"[DONE] Telemetry: {args.log}")
    print(f"[INFO] Retrieval updates: {n_updates}/{n_frames}")
    print(f"[INFO] Non-increasing anchor stamps skipped: {n_skipped_stamps}")
    print(
        f"[TIMING] Anchor processing mean={timing_array.mean():.2f}ms, "
        f"p95={np.percentile(timing_array, 95):.2f}ms"
    )
    print(
        "[TIMING] Excludes bag decoding and secondary-camera preprocessing "
        "performed before the anchor."
    )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap_path")
    parser.add_argument(
        "--checkpoint", default="checkpoints/racing_model.pt"
    )
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--log", default="inference_telemetry.csv")
    parser.add_argument("--val-only", action="store_true")

    # Winning OVERALL-RMSE settings from the supplied sweep.
    parser.add_argument("--q-x-floor", type=float, default=1.0)
    parser.add_argument("--r-scale", type=float, default=0.03)
    parser.add_argument("--gate-m", type=float, default=35.0)
    parser.add_argument("--gate-deg", type=float, default=35.0)
    parser.add_argument("--min-sim", type=float, default=0.45)

    parser.add_argument(
        "--fix-q-scaling",
        action="store_true",
        help="Optional experiment; OFF for the supplied sweep winner.",
    )

    args = parser.parse_args()

    if args.q_x_floor < 0:
        parser.error("--q-x-floor must be nonnegative.")
    if args.r_scale <= 0:
        parser.error("--r-scale must be positive.")
    if args.gate_m <= 0 or args.gate_deg <= 0:
        parser.error("Gate limits must be positive.")

    run_causal_inference(args)


if __name__ == "__main__":
    main()