# Monocular Occupancy Grid

This is a small project I built to generate a local occupancy grid using only a monocular camera.

The input comes from a recorded ROS 2 bag. I use the camera image together with the camera calibration, run Depth Anything V2 to estimate metric depth, and then use that depth to separate the floor from obstacles and build a 2D occupancy grid.

The main goal was to see how useful monocular depth could be for local robot navigation without relying on LiDAR.

## How it works

The pipeline is roughly:

```text
ROS 2 bag
   ↓
camera image + camera info
   ↓
image rectification
   ↓
Depth Anything V2
   ↓
metric depth
   ↓
3D projection
   ↓
floor estimation
   ↓
obstacle detection
   ↓
occupancy grid
```

For floor estimation I use RANSAC on points from the lower part of the image. Points above the estimated floor are treated as possible obstacles.

Free space is then raycast from the robot position until an obstacle or the visible floor limit is reached.

## Occupancy grid

The current grid is:

- 16 m forward
- 10 m wide
- 0.10 m per cell

The values are:

```text
-1 = unknown
 0 = free
100 = occupied
```

## Demo

A sample result is included here:

```text
demo/camera_vs_occupancy_debug.mp4
```

It shows the original camera frame next to the generated occupancy grid.

## Setup

Install the Python packages:

```bash
pip install -r requirements.txt
```

FFmpeg is also needed because the script creates a comparison video.

On macOS:

```bash
brew install ffmpeg
```

## Depth Anything V2

I use the metric depth version of Depth Anything V2 with the small ViT-S model.

Depth Anything V2 is not included in this repository, so it needs to be downloaded separately.

The folder passed to `--da2-root` should contain something like:

```text
run.py
checkpoints/
└── depth_anything_v2_metric_hypersim_vits.pth
```

## ROS bag

The script currently expects these topics:

```text
/image_raw
/camera_info
```

`/camera_info` is used to rectify the images and get the camera intrinsics.

## Running it

```bash
python3 src/occupancy_grid.py \
  --bag /path/to/your/rosbag \
  --da2-root /path/to/Depth-Anything-V2/metric_depth
```

The results will be saved in:

```text
output/
```

By default I process every 10th camera frame. You can change that with:

```bash
--every 5
```

You can also choose another output folder:

```bash
--output my_results
```

## Output

A run produces files like:

```text
output/
├── camera_info.json
├── rgb_frames/
├── rectified_frames/
├── depth_output/
├── ogm_frames/
├── comparison_frames/
└── camera_vs_occupancy.mp4
```

The occupancy grids are saved as both images and NumPy arrays so they can be used later for planning or other experiments.

## Next steps

Right now this repo is only focused on generating the occupancy grid.

I want to extend it later with local goal selection and path planning so the robot can use the occupancy grid to move around obstacles.

## Note

This is an experimental project and the occupancy grid depends heavily on the quality of the monocular depth estimate. It should not be treated as a safety-critical perception system.