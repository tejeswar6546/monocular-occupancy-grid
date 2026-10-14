#!/usr/bin/env python3
"""Build a local occupancy grid from monocular camera data.

The script reads images and camera calibration from a ROS 2 bag,
estimates metric depth with Depth Anything V2, estimates the floor
plane, and converts the result into a 2D occupancy grid.

Occupancy values:
  -1 unknown
   0 free
 100 occupied
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Occupancy grid settings

RESOLUTION = 0.10
X_MIN = -5.0
X_MAX = 5.0
Z_MIN = 0.0
Z_MAX = 16.0
MAX_DEPTH = 20.0

ANGLE_RESOLUTION = 0.25
MAX_OBSTACLE_CONNECTION_M = 0.30
RANSAC_THRESHOLD = 0.08
FLOOR_TOLERANCE = 0.10
FLOOR_IMAGE_MIN_Y_FRACTION = 0.42
MIN_FLOOR_POINTS_PER_BIN = 3
MIN_OBSTACLE_HEIGHT = 0.10
MAX_OBSTACLE_HEIGHT = 1.50

IMAGE_TOPIC = "/image_raw"
CAMINFO_TOPIC = "/camera_info"


def grid_line(r1, c1, r2, c2):
    n = max(abs(r2 - r1), abs(c2 - c1)) + 1
    rows = np.linspace(r1, r2, n).astype(int)
    cols = np.linspace(c1, c2, n).astype(int)
    return rows, cols


def mark_free_ray(grid, robot_row, robot_col, end_row, end_col):
    rr, cc = grid_line(robot_row, robot_col, end_row, end_col)
    for r, c in zip(rr, cc):
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            if grid[r, c] != 100:
                grid[r, c] = 0


def make_empty_grid():
    rows = int((Z_MAX - Z_MIN) / RESOLUTION)
    cols = int((X_MAX - X_MIN) / RESOLUTION)
    return np.full((rows, cols), -1, dtype=np.int16)


def rosimg_to_rgb(msg):
    h = int(msg.height)
    w = int(msg.width)
    enc = str(msg.encoding).lower()

    if isinstance(msg.data, np.ndarray):
        data = np.asarray(msg.data, dtype=np.uint8).reshape(-1)
    else:
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    step = int(msg.step)
    rows = data[: h * step].reshape(h, step)

    if enc == "rgb8":
        return rows[:, : w * 3].reshape(h, w, 3).copy()

    if enc == "bgr8":
        bgr = rows[:, : w * 3].reshape(h, w, 3)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if enc == "mono8":
        img = rows[:, :w]
        return np.repeat(img[:, :, None], 3, axis=2)

    if enc in ("uyvy", "yuv422"):
        img = rows[:, : w * 2].reshape(h, w, 2)
        return cv2.cvtColor(img, cv2.COLOR_YUV2RGB_UYVY)

    if enc in ("yuyv", "yuy2", "yuv422_yuy2"):
        img = rows[:, : w * 2].reshape(h, w, 2)
        return cv2.cvtColor(img, cv2.COLOR_YUV2RGB_YUY2)

    raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")


def extract_frames_and_camerainfo(bag: Path, out: Path, keep_every: int):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    rgb_dir = out / "rgb_frames"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    typestore = get_typestore(Stores.LATEST)

    camera_count = 0
    saved = 0
    caminfo_saved = False

    with AnyReader([bag], default_typestore=typestore) as reader:
        connections = [
            c for c in reader.connections
            if c.topic in {IMAGE_TOPIC, CAMINFO_TOPIC}
        ]

        if not any(c.topic == IMAGE_TOPIC for c in connections):
            raise RuntimeError(f"{IMAGE_TOPIC} not found in bag.")
        if not any(c.topic == CAMINFO_TOPIC for c in connections):
            raise RuntimeError(f"{CAMINFO_TOPIC} not found in bag.")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            if connection.topic == CAMINFO_TOPIC and not caminfo_saved:
                caminfo = {
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "distortion_model": str(msg.distortion_model),
                    "d": [float(x) for x in msg.d],
                    "k": [float(x) for x in msg.k],
                    "r": [float(x) for x in msg.r],
                    "p": [float(x) for x in msg.p],
                }

                if caminfo["k"][0] <= 0 or caminfo["k"][4] <= 0:
                    raise RuntimeError(
                        "camera_info found, but fx/fy in K look invalid."
                    )

                (out / "camera_info.json").write_text(
                    json.dumps(caminfo, indent=2)
                )
                caminfo_saved = True

            elif connection.topic == IMAGE_TOPIC:
                if camera_count % keep_every == 0:
                    rgb = rosimg_to_rgb(msg)
                    outfile = rgb_dir / f"frame_{saved:05d}.jpg"
                    Image.fromarray(rgb).save(outfile, quality=95)
                    saved += 1
                camera_count += 1

    if not caminfo_saved:
        raise RuntimeError("No valid /camera_info was saved.")

    print(f"[1/5] Camera messages: {camera_count}")
    print(f"[1/5] Saved frames:    {saved}")
    return saved


def rectify_frames(out: Path):
    rgb_dir = out / "rgb_frames"
    rect_dir = out / "rectified_frames"
    rect_dir.mkdir(parents=True, exist_ok=True)

    cam = json.loads((out / "camera_info.json").read_text())

    w = int(cam["width"])
    h = int(cam["height"])
    K = np.array(cam["k"], dtype=np.float64).reshape(3, 3)
    D = np.array(cam["d"], dtype=np.float64)
    R = np.array(cam["r"], dtype=np.float64).reshape(3, 3)
    P = np.array(cam["p"], dtype=np.float64).reshape(3, 4)

    newK = P[:, :3].copy()
    if newK[0, 0] <= 0 or newK[1, 1] <= 0:
        raise RuntimeError("CameraInfo projection matrix P has invalid fx/fy.")

    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, R, newK, (w, h), cv2.CV_32FC1
    )

    files = sorted(rgb_dir.glob("frame_*.jpg"))
    if not files:
        raise RuntimeError("No RGB frames were extracted.")

    for i, f in enumerate(files, start=1):
        img = cv2.imread(str(f))
        if img is None:
            raise RuntimeError(f"Could not read {f}")
        if img.shape[1] != w or img.shape[0] != h:
            raise RuntimeError(
                f"Image {f.name} is {img.shape[1]}x{img.shape[0]}, "
                f"but CameraInfo is {w}x{h}."
            )

        rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(rect_dir / f.name), rect)

        if i == 1 or i % 25 == 0 or i == len(files):
            print(f"[2/5] Rectify {i}/{len(files)}")


def run_depth_anything(da2_root: Path, out: Path):
    checkpoint = (
        da2_root
        / "checkpoints"
        / "depth_anything_v2_metric_hypersim_vits.pth"
    )
    run_py = da2_root / "run.py"

    if not run_py.exists():
        raise FileNotFoundError(f"Depth Anything run.py not found: {run_py}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Small Hypersim checkpoint not found: {checkpoint}")

    depth_dir = out / "depth_output"
    depth_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        str(run_py),
        "--encoder", "vits",
        "--load-from", str(checkpoint),
        "--max-depth", "20",
        "--img-path", str(out / "rectified_frames"),
        "--outdir", str(depth_dir),
        "--save-numpy",
    ]

    print("[3/5] Running Depth Anything V2 Metric Small (vits)...")
    subprocess.run(cmd, cwd=str(da2_root), check=True)


def depth_to_ogm(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    h, w = depth.shape
    grid = make_empty_grid()

    step = 4
    v, u = np.mgrid[0:h:step, 0:w:step]
    Z = depth[::step, ::step]

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    valid = np.isfinite(Z) & (Z > 0.30) & (Z < MAX_DEPTH)

    # Use the lower-center part of the image to estimate the floor

    candidate = (
        valid
        & (v > h * 0.70)
        & (u > w * 0.20)
        & (u < w * 0.80)
    )

    xf = X[candidate]
    yf = Y[candidate]
    zf = Z[candidate]

    if len(xf) < 20:
        return grid

    rng = np.random.default_rng(42)
    best_inliers = None
    best_count = 0

    # RANSAC floor fitting

    for _ in range(500):
        ids = rng.choice(len(xf), 3, replace=False)
        A = np.column_stack((xf[ids], zf[ids], np.ones(3)))
        if abs(np.linalg.det(A)) < 1e-8:
            continue

        a, b, c = np.linalg.solve(A, yf[ids])
        pred = a * xf + b * zf + c
        err = np.abs(yf - pred)
        inliers = err < RANSAC_THRESHOLD
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        return grid

    A = np.column_stack(
        (xf[best_inliers], zf[best_inliers], np.ones(best_count))
    )
    a, b, c = np.linalg.lstsq(
        A, yf[best_inliers], rcond=None
    )[0]

    floor_y = a * X + b * Z + c
    height = floor_y - Y

    floor_mask = (
        valid
        & (np.abs(height) <= FLOOR_TOLERANCE)
        & (v > h * FLOOR_IMAGE_MIN_Y_FRACTION)
    )
    floor_x = X[floor_mask]
    floor_z = Z[floor_mask]

    obstacle_mask = (
        valid
        & (height > MIN_OBSTACLE_HEIGHT)
        & (height < MAX_OBSTACLE_HEIGHT)
    )
    obstacle_x = X[obstacle_mask]
    obstacle_z = Z[obstacle_mask]

    angle_min = np.degrees(np.arctan2(-cx, fx))
    angle_max = np.degrees(np.arctan2(w - 1 - cx, fx))
    edges = np.arange(
        angle_min,
        angle_max + ANGLE_RESOLUTION,
        ANGLE_RESOLUTION,
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    n_bins = len(centers)

    obstacle_scan = np.full(n_bins, np.nan)
    floor_scan = np.full(n_bins, np.nan)

    if len(obstacle_x) > 0:
        angles = np.degrees(np.arctan2(obstacle_x, obstacle_z))
        ranges = np.sqrt(obstacle_x ** 2 + obstacle_z ** 2)
        bins = np.floor(
            (angles - angle_min) / ANGLE_RESOLUTION
        ).astype(int)

        for b_id, distance in zip(bins, ranges):
            if 0 <= b_id < n_bins:
                if (
                    np.isnan(obstacle_scan[b_id])
                    or distance < obstacle_scan[b_id]
                ):
                    obstacle_scan[b_id] = distance

    if len(floor_x) > 0:
        angles = np.degrees(np.arctan2(floor_x, floor_z))
        ranges = np.sqrt(floor_x ** 2 + floor_z ** 2)
        bins = np.floor(
            (angles - angle_min) / ANGLE_RESOLUTION
        ).astype(int)

        for b_id in range(n_bins):
            values = ranges[bins == b_id]
            if len(values) >= MIN_FLOOR_POINTS_PER_BIN:
                floor_scan[b_id] = np.percentile(values, 90)

    robot_col = int((0 - X_MIN) / RESOLUTION)
    robot_row = 0

    # Free-space raycasting.
    for angle_deg, floor_distance, obstacle_distance in zip(
        centers, floor_scan, obstacle_scan
    ):
        free_distance = np.nan

        if np.isfinite(floor_distance):
            free_distance = floor_distance

        if np.isfinite(obstacle_distance):
            before_obstacle = max(
                obstacle_distance - RESOLUTION, 0
            )
            if np.isfinite(free_distance):
                free_distance = min(
                    free_distance, before_obstacle
                )
            else:
                free_distance = before_obstacle

        if not np.isfinite(free_distance) or free_distance <= 0:
            continue

        free_distance = min(free_distance, Z_MAX)
        angle = np.radians(angle_deg)
        x = np.sin(angle) * free_distance
        z = np.cos(angle) * free_distance

        col = int((x - X_MIN) / RESOLUTION)
        row = int(z / RESOLUTION)

        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
            mark_free_ray(
                grid,
                robot_row,
                robot_col,
                row,
                col,
            )

    # Occupied endpoints.
    hits = []
    for angle_deg, distance in zip(centers, obstacle_scan):
        if not np.isfinite(distance) or distance > Z_MAX:
            continue

        angle = np.radians(angle_deg)
        x = np.sin(angle) * distance
        z = np.cos(angle) * distance

        col = int((x - X_MIN) / RESOLUTION)
        row = int(z / RESOLUTION)

        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
            grid[row, col] = 100
            hits.append((angle_deg, x, z, row, col))

    # Connect nearby obstacle endpoints.
    for i in range(len(hits) - 1):
        a1, x1, z1, r1, c1 = hits[i]
        a2, x2, z2, r2, c2 = hits[i + 1]

        if abs(a2 - a1) > 1.0:
            continue

        physical_gap = np.hypot(x2 - x1, z2 - z1)
        if physical_gap > MAX_OBSTACLE_CONNECTION_M:
            continue

        rr, cc = grid_line(r1, c1, r2, c2)
        for r, c in zip(rr, cc):
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                grid[r, c] = 100

    return grid


def render_ogm(grid, scale=4, annotate=True):
    img = np.full(grid.shape, 127, dtype=np.uint8)
    img[grid == 0] = 255
    img[grid == 100] = 0
    img = np.flipud(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if annotate:
        h, w = img.shape[:2]
        center = w // 2
        cv2.arrowedLine(
            img,
            (center, h - 4),
            (center, h - 18),
            (0, 0, 255),
            2,
            tipLength=0.4,
        )
        cv2.putText(
            img,
            "OCCUPANCY GRID",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    if scale > 1:
        img = cv2.resize(
            img,
            (img.shape[1] * scale, img.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    return img


def fit_to_cell(img, cell_w=720, cell_h=720, background=(255, 255, 255)):
    h, w = img.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.full(
        (cell_h, cell_w, 3),
        background,
        dtype=np.uint8,
    )
    x0 = (cell_w - new_w) // 2
    y0 = (cell_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def build_occupancy_outputs(out: Path):
    raw_dir = out / "rgb_frames"
    depth_dir = out / "depth_output"
    ogm_dir = out / "ogm_frames"
    comp_dir = out / "comparison_frames"

    ogm_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    cam = json.loads((out / "camera_info.json").read_text())

    # Use rectified camera intrinsics
    P = cam["p"]
    fx = float(P[0])
    fy = float(P[5])
    cx = float(P[2])
    cy = float(P[6])

    if fx <= 0 or fy <= 0:
        raise RuntimeError("Rectified intrinsics in CameraInfo P are invalid.")

    raw_files = sorted(raw_dir.glob("frame_*.jpg"))
    depth_files = sorted(
        depth_dir.glob("*_raw_depth_meter.npy")
    )
    count = min(len(raw_files), len(depth_files))

    print(f"[4/5] Raw frames: {len(raw_files)}")
    print(f"[4/5] Depth maps: {len(depth_files)}")
    print(f"[4/5] Using:      {count}")

    if count == 0:
        raise RuntimeError("No matching raw/depth files found.")

    for i in range(count):
        raw = cv2.imread(str(raw_files[i]))
        depth = np.load(depth_files[i])

        grid = depth_to_ogm(
            depth,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        ogm = render_ogm(grid)

        np.save(
            ogm_dir / f"ogm_{i:05d}.npy",
            grid,
        )
        cv2.imwrite(
            str(ogm_dir / f"ogm_{i:05d}.png"),
            ogm,
        )

        raw_panel = fit_to_cell(raw, 720, 720)
        ogm_panel = fit_to_cell(ogm, 720, 720)

        combined = np.hstack((raw_panel, ogm_panel))
        header = np.full(
            (60, combined.shape[1], 3),
            255,
            dtype=np.uint8,
        )

        cv2.putText(
            header,
            "ORIGINAL CAMERA",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            header,
            "CAMERA-BASED OCCUPANCY",
            (760, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
        )

        frame = np.vstack((header, combined))
        cv2.imwrite(
            str(comp_dir / f"comparison_{i:05d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        if i == 0 or (i + 1) % 25 == 0 or i == count - 1:
            print(f"[4/5] OGM {i+1}/{count}")

    return count


def create_h264_mp4(out: Path, framerate: int):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed or not available in PATH."
        )

    output_video = out / "camera_vs_occupancy.mp4"
    pattern = out / "comparison_frames" / "comparison_%05d.jpg"

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(framerate),
        "-i", str(pattern),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-movflags", "+faststart",
        str(output_video),
    ]

    print("[5/5] Creating H.264 MP4...")
    subprocess.run(cmd, check=True)
    return output_video


def parse_args():
    p = argparse.ArgumentParser(
        description="Camera-based occupancy-grid mapping pipeline."
    )

    p.add_argument(
        "--bag",
        type=Path,
        required=True,
        help="Path to the ROS 2 bag directory.",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory for generated occupancy-grid results.",
    )

    p.add_argument(
        "--da2-root",
        type=Path,
        required=True,
        help="Path to the Depth Anything V2 metric_depth directory.",
    )

    p.add_argument(
        "--every",
        type=int,
        default=10,
        help="Keep every Nth camera frame.",
    )

    p.add_argument(
        "--video-fps",
        type=int,
        default=8,
        help="Frame rate for the generated comparison video.",
    )

    p.add_argument(
        "--skip-depth",
        action="store_true",
        help="Reuse existing depth_output instead of rerunning Depth Anything V2.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    bag = args.bag.expanduser().resolve()
    out = args.output.expanduser().resolve()
    da2_root = args.da2_root.expanduser().resolve()

    if args.every < 1:
        raise ValueError("--every must be >= 1")

    if not bag.exists():
        raise FileNotFoundError(f"Bag not found: {bag}")
    if not da2_root.exists():
        raise FileNotFoundError(f"DA2 metric_depth folder not found: {da2_root}")

    for name in (
        "rgb_frames",
        "rectified_frames",
        "depth_output",
        "ogm_frames",
        "comparison_frames",
    ):
        (out / name).mkdir(parents=True, exist_ok=True)

    print("=== FINAL OCCUPANCY GRID ===")
    print(f"Bag:       {bag}")
    print(f"Output:    {out}")
    print(f"DA2:       {da2_root}")
    print("Model:     Depth Anything V2 Metric Hypersim Small (vits)")
    print("Grid:      16 m forward, 10 m wide, 0.10 m/cell")
    print()

    extract_frames_and_camerainfo(
        bag=bag,
        out=out,
        keep_every=args.every,
    )
    rectify_frames(out)

    if not args.skip_depth:
        run_depth_anything(da2_root, out)
    else:
        print("[3/5] Skipping DA2; reusing existing depth_output.")

    build_occupancy_outputs(out)
    video = create_h264_mp4(out, args.video_fps)

    print()
    print("=== DONE ===")
    print(f"Final MP4: {video}")
    print(f"OGM NPYs:  {out / 'ogm_frames'}")
    print(f"All files: {out}")


if __name__ == "__main__":
    main()
