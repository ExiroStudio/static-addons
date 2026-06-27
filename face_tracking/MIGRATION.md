# Face Tracking Engine Migration

## Why?
The previous BlazeFace (MediaPipe) ONNX model caused significant micro-jitter, resulting in positional variance and latency "rubber-banding" downstream in the geometric pose solver. Smoothing parameters had to be set artificially high to mask this noise, which compromised real-time responsiveness.

## What Changed?
1. **Model Swap**: The detector pipeline now relies exclusively on `scrfd_500m_bnkps.onnx`.
2. **Adapter Architecture**: The detection struct `DecodedFace` acts as a boundary adapter. The landmark extraction layer maps SCRFD's 5-point layout seamlessly to the existing 5-point geometry struct without any pose or tracking logic changes.
3. **CPU NMS Integration**: We integrated a robust candidate filtration, Top-16 hardcap, and an unoptimized overlapping CPU-side NMS loop that parses raw SCRFD logits directly. It executes in `<0.5ms` per frame.
4. **Anchor Switch**: Geometry projection anchor moved from the `nose` to the mathematical `centroid` of the 5 points to eliminate localized muscular noise from smiling or talking.

## Rollback Command
If any critical failure is detected in production, do not attempt a partial revert. Use the git history as the absolute source of truth.

```bash
git checkout HEAD~1
```
*(Run this command in the `ascii-realtime/addons/face_tracking` directory to fully revert the SCRFD cutover and restore the BlazeFace model, loader bindings, and NMS-less postprocessing).*
