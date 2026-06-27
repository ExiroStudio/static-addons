# Face Tracking Addon v1

A modular, pure external native behavior addon for tracking head position, rotation, and scale.

## Folder Structure

```
addons/face_tracking/
├── manifest.toml          # Addon definition and signal set
├── README.md              # This guide
├── runtime/               # Platform-specific binaries
│   └── linux-x86_64/
│       └── bootstrap      # Native runner entry point
├── src/                   # Implementation source code
│   ├── mod.rs             # Module entry point
│   ├── bootstrap/         # Protocol and lifecycle handling
│   ├── runtime/           # Model execution and orchestration
│   ├── tracker/           # Tracking state and interpolation
│   ├── filters/           # One Euro Filter smoothing
│   └── signals/           # HostApi signal publishing
├── models/                # BlazeFace / YOLO model weights
├── cache/                 # Runtime performance cache
└── temp/                  # Scratch buffers
```

## Internal Architecture

- **Bootstrap**: Handles the Phase 3b control transport (Stdin/Stdout).
- **Runtime**: Orchestrates the pipeline. Decision logic for Inference (10Hz) vs Tracking (30Hz). Implements Landmark Extraction and Pose Solving for stable rotation.
- **Tracker**: Manages the bounding box and interpolation. Resets if `lost_timeout` is exceeded. Correctly handles `tracking_age`.
- **Filters**: Implements the One Euro Filter for high-stability output.
- **Signals**: Native bridge to `HostApi::publish`. Includes motion state (velocity).

## Signals

| Signal | Type | Description |
|--------|------|-------------|
| `face.position` | `vec2` | Normalized center [0..1]. |
| `face.rotation` | `vec3` | Yaw, Pitch, Roll (degrees). |
| `face.scale` | `f32` | Normalized head size [0..1]. |
| `face.visible` | `bool` | True if tracking is active. |
| `face.confidence` | `f32` | Detection confidence [0..1]. |
| `face.tracking_age`| `f32` | Seconds since last real detection. |
| `face.velocity` | `vec2` | Pixel velocity [unit/sec]. |
| `face.angular_velocity`| `vec3` | Degrees/sec. |

## GPU Strategy
The addon initializes its OWN GPU context. It never touches the engine's `wgpu` device or textures. All preprocessing and inference happen within the addon's sandbox.
