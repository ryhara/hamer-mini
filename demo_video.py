# Video demo script for hamer-mini: runs the pipeline on every frame of an
# input video and writes an annotated output video.
# Reuses the drawing helpers from demo.py.
import argparse
import os
import time

import cv2
import numpy as np
import torch

from demo import (
    draw_hand,
    draw_hand_rotations,
    render_meshes_pyrender,
    try_import_pyrender,
)
from hamer_mini import HaMeRHandPose3dEstimationPipeline

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def open_writer(path: str, fps: float, size: tuple) -> cv2.VideoWriter:
    """Open a VideoWriter for ``path``, picking the codec from the extension."""
    ext = os.path.splitext(path)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if ext == ".avi" else "mp4v"))
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    return writer


def main():
    parser = argparse.ArgumentParser(description="hamer-mini video demo")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory, or output video file path")
    parser.add_argument("--body-conf", type=float, default=0.5, help="Person detection score threshold")
    parser.add_argument("--hand-conf", type=float, default=0.5, help="Hand keypoint confidence threshold")
    parser.add_argument("--rescale-factor", type=float, default=1.5, help="Hand bbox padding factor")
    parser.add_argument("--mesh", action="store_true", help="Also render the projected MANO mesh overlay")
    parser.add_argument("--rotation", action="store_true",
                        help="Visualize the MANO rotations: wrist (global_orient) frame and "
                             "per-joint (hand_pose) axes")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Process at most this many frames (0 = all)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"input: {args.video} ({width}x{height} @ {fps:.2f} fps, {total} frames)")

    if os.path.splitext(args.output)[1].lower() in VIDEO_EXTS:
        out_path = args.output
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    else:
        os.makedirs(args.output, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join(args.output, f"{stem}_out.mp4")
    writer = open_writer(out_path, fps, (width, height))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = HaMeRHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    faces_right = np.asarray(pipe.mano_faces, dtype=np.int64)

    # Probe the optional pyrender renderer once; reused for every frame.
    pyrender_mods = try_import_pyrender() if args.mesh else None

    n_frames = 0
    start = time.time()
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if args.max_frames and n_frames >= args.max_frames:
                break

            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            outputs = pipe.predict(image_rgb,
                                   body_conf=args.body_conf,
                                   hand_conf=args.hand_conf,
                                   rescale_factor=args.rescale_factor,
                                   return_vertices_2d=args.mesh)

            if pyrender_mods is not None and len(outputs) > 0:
                render_meshes_pyrender(frame_bgr, outputs, faces_right, *pyrender_mods)
            for ret in outputs:
                draw_hand(frame_bgr, ret, faces_right,
                          draw_mesh=args.mesh and pyrender_mods is None)
                if args.rotation and ret.get("hamer_preds") is not None:
                    draw_hand_rotations(frame_bgr, ret)

            writer.write(frame_bgr)
            n_frames += 1
            if n_frames % 30 == 0 or n_frames == total:
                elapsed = time.time() - start
                print(f"frame {n_frames}/{total or '?'} "
                      f"({n_frames / elapsed:.2f} fps, {len(outputs)} hand(s) in last frame)")
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - start
    print(f"processed {n_frames} frames in {elapsed:.1f}s ({n_frames / max(elapsed, 1e-6):.2f} fps)")
    print(f"saved visualization to {out_path}")


if __name__ == "__main__":
    main()
