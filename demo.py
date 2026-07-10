# Demo script for hamer-mini.
# Visualization approach based on: https://github.com/warmshao/WiLoR-mini/blob/main/tests/test_pipelines.py
import argparse
import os

# Captured before importing hamer_mini: mmpose (ViTPose) overwrites PYOPENGL_PLATFORM
# with 'osmesa' at import time, so remember what the user actually set.
_USER_PYOPENGL_PLATFORM = os.environ.get("PYOPENGL_PLATFORM")

import cv2
import numpy as np
import torch

from hamer_mini import (
    FINGER_NAMES,
    FINGERTIP_KEYPOINTS,
    FINGERTIP_PARENT_JOINTS,
    MANO_TO_KEYPOINT,
    HaMeRHandPose3dEstimationPipeline,
    axis_angle_to_euler,
    axis_angle_to_matrix,
    cumulative_joint_rotations,
    matrix_to_euler,
)

# 21 hand joints in OpenPose ordering: 0 = wrist, then 4 joints per finger
# (thumb, index, middle, ring, pinky) from base to tip.
HAND_SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4),
                 (0, 5), (5, 6), (6, 7), (7, 8),
                 (0, 9), (9, 10), (10, 11), (11, 12),
                 (0, 13), (13, 14), (14, 15), (15, 16),
                 (0, 17), (17, 18), (18, 19), (19, 20)]

# Axis colors (BGR): x=red, y=green, z=blue.
AXIS_COLORS = [(60, 60, 255), (60, 220, 60), (255, 120, 80)]


def hand_color(is_right) -> tuple:
    """BGR color used for everything drawn for one hand (bbox, mesh, skeleton)."""
    return (0, 255, 0) if is_right else (255, 128, 0)


def try_import_pyrender():
    """Import the optional pyrender/trimesh renderer, or return None if unavailable.

    Requires the 'render' extra (pip install hamer-mini[render]). Offscreen EGL
    rendering is used by default, which also works headless and avoids hanging on
    stale X11 DISPLAY forwarding; set PYOPENGL_PLATFORM to override. If no working
    OpenGL context can be created the demo falls back to the built-in OpenCV
    mesh renderer.
    """
    os.environ["PYOPENGL_PLATFORM"] = _USER_PYOPENGL_PLATFORM or "egl"
    try:
        import pyrender
        import trimesh
        renderer = pyrender.OffscreenRenderer(8, 8)  # probe that an OpenGL context works
        renderer.delete()
        return pyrender, trimesh
    except Exception as e:  # noqa: BLE001 - any failure means "not available here"
        print(f"pyrender unavailable ({e.__class__.__name__}: {e}); "
              "using the built-in OpenCV mesh renderer")
        return None


def _raymond_light_nodes(pyrender) -> list:
    """Three directional lights around the camera (as in the original HaMeR renderer)."""
    nodes = []
    for phi in [0.0, 2.0 / 3.0 * np.pi, 4.0 / 3.0 * np.pi]:
        theta = np.pi / 6.0
        z = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.column_stack([x, np.cross(z, x), z])
        nodes.append(pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                                   matrix=matrix))
    return nodes


def render_meshes_pyrender(image_bgr: np.ndarray, outputs: list, faces_right: np.ndarray,
                           pyrender, trimesh) -> None:
    """Render all hand meshes into one scene (correct inter-hand occlusion) and blend in place."""
    hands = [ret for ret in outputs if ret.get("hamer_preds") is not None]
    if not hands:
        return
    h, w = image_bgr.shape[:2]
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    # OpenCV camera (x right, y down, z forward) -> OpenGL camera (y up, z backward)
    rot180x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    for ret in hands:
        preds = ret["hamer_preds"]
        color = hand_color(ret["is_right"])
        # Left hand meshes are mirrored, so flip the triangle winding order
        faces = faces_right if ret["is_right"] else faces_right[:, [0, 2, 1]]
        vertices = preds["pred_vertices"][0] + preds["pred_cam_t_full"][0]
        mesh = trimesh.Trimesh(vertices.astype(np.float64), faces.copy())
        mesh.apply_transform(rot180x)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, alphaMode="OPAQUE",
            baseColorFactor=(color[2] / 255.0, color[1] / 255.0, color[0] / 255.0, 1.0))
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material))
    focal = hands[0]["hamer_preds"]["scaled_focal_length"]
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=w / 2.0, cy=h / 2.0, zfar=1e12)
    scene.add(camera, pose=np.eye(4))
    for node in _raymond_light_nodes(pyrender):
        scene.add_node(node)
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h, point_size=1.0)
    rgba = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)[0].astype(np.float32) / 255.0
    renderer.delete()
    alpha = rgba[:, :, 3:4]
    mesh_bgr = rgba[:, :, 2::-1] * 255.0
    image_bgr[:] = (mesh_bgr * alpha + image_bgr.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def draw_joints(image_bgr: np.ndarray, keypoints_2d: np.ndarray, color: tuple) -> None:
    """Draw the 21 2D hand joints connected by the finger skeleton."""
    pts = np.round(keypoints_2d).astype(int)
    for a, b in HAND_SKELETON:
        cv2.line(image_bgr, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(image_bgr, (x, y), 3, color, -1, cv2.LINE_AA)


def draw_rotation_axes(image_bgr: np.ndarray, origin: tuple, rotmat: np.ndarray,
                       length: float, thickness: int = 2) -> None:
    """Draw a rotated coordinate frame at ``origin`` (orthographic projection).

    The columns of ``rotmat`` are the frame's x/y/z axes in camera space;
    their (x, y) components are drawn directly in image space.
    """
    ox, oy = int(origin[0]), int(origin[1])
    for j in range(3):
        dx = int(rotmat[0, j] * length)
        dy = int(rotmat[1, j] * length)
        cv2.line(image_bgr, (ox, oy), (ox + dx, oy + dy),
                 AXIS_COLORS[j], thickness, cv2.LINE_AA)


def draw_hand_rotations(image_bgr: np.ndarray, ret: dict) -> None:
    """Draw the MANO rotations of one hand: the wrist (global_orient) frame at
    the wrist keypoint plus a small camera-space frame at every finger joint
    and fingertip (hand_pose composed along the kinematic chain)."""
    preds = ret["hamer_preds"]
    kpts = preds["pred_keypoints_2d"][0]
    global_orient = preds["global_orient"].reshape(3)
    hand_pose = preds["hand_pose"].reshape(15, 3)
    x1, y1, x2, y2 = ret["crop_bbox"]
    bbox_size = max(x2 - x1, y2 - y1)

    joint_rots = cumulative_joint_rotations(global_orient, hand_pose)
    axis_len = max(6.0, bbox_size * 0.06)
    for j, k in enumerate(MANO_TO_KEYPOINT):
        draw_rotation_axes(image_bgr, (kpts[k, 0], kpts[k, 1]), joint_rots[j],
                           axis_len, thickness=1)
    # Fingertips inherit the distal phalanx orientation.
    for j, k in zip(FINGERTIP_PARENT_JOINTS, FINGERTIP_KEYPOINTS):
        draw_rotation_axes(image_bgr, (kpts[k, 0], kpts[k, 1]), joint_rots[j],
                           axis_len, thickness=1)

    wrist_rot = axis_angle_to_matrix(global_orient)
    draw_rotation_axes(image_bgr, (kpts[0, 0], kpts[0, 1]), wrist_rot,
                       max(12.0, bbox_size * 0.2))


def print_mano_rotations(preds: dict) -> None:
    """Print the MANO parameters of one hand as axis-angle Euler angles."""
    global_orient = preds["global_orient"].reshape(3)
    hand_pose = preds["hand_pose"].reshape(15, 3)
    roll, pitch, yaw = axis_angle_to_euler(global_orient)
    print(f"  global_orient (deg): roll={roll:+.1f} pitch={pitch:+.1f} yaw={yaw:+.1f}")
    for j, (jr, jp, jy) in enumerate(axis_angle_to_euler(hand_pose)):
        print(f"  hand_pose[{j:2d}] (deg): roll={jr:+.1f} pitch={jp:+.1f} yaw={jy:+.1f}")
    tip_euler = matrix_to_euler(
        cumulative_joint_rotations(global_orient, hand_pose)[FINGERTIP_PARENT_JOINTS])
    for name, (tr, tp, ty) in zip(FINGER_NAMES, tip_euler):
        print(f"  fingertip {name:6s} (deg): roll={tr:+.1f} pitch={tp:+.1f} yaw={ty:+.1f}")
    print(f"  betas: {np.round(preds['betas'][0], 3).tolist()}")


def render_mesh(image_bgr: np.ndarray, vertices_2d: np.ndarray, vertices_cam: np.ndarray,
                faces: np.ndarray, color: tuple, alpha: float = 0.6) -> None:
    """Overlay the projected MANO mesh with flat shading and depth sorting.

    Args:
        vertices_2d: (778, 2) vertices in full image pixels.
        vertices_cam: (778, 3) vertices in the camera frame (used for depth and normals).
        faces: (1538, 3) triangle indices.
        color: BGR base color of the mesh.
        alpha: Blending weight of the mesh overlay.
    """
    overlay = image_bgr.copy()
    pts2d = np.round(vertices_2d).astype(np.int32)
    tri3d = vertices_cam[faces]  # (F, 3, 3)
    normals = np.cross(tri3d[:, 1] - tri3d[:, 0], tri3d[:, 2] - tri3d[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    # Shade each face by how much it points towards the camera (z axis)
    shade = 0.35 + 0.65 * np.abs(normals[:, 2])
    face_colors = np.asarray(color, dtype=np.float32)[None] * shade[:, None]
    depth = tri3d[:, :, 2].mean(axis=1)
    for f in np.argsort(-depth):  # painter's algorithm: far faces first
        cv2.fillConvexPoly(overlay, pts2d[faces[f]], face_colors[f].tolist())
    cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, dst=image_bgr)


def draw_hand(image_bgr: np.ndarray, ret: dict, faces_right: np.ndarray, draw_mesh: bool) -> None:
    """Draw the mesh overlay, 2D skeleton, model input crop bbox and confidence in place."""
    is_right = ret["is_right"]
    color = hand_color(is_right)

    preds = ret.get("hamer_preds")
    if preds is not None:
        if draw_mesh and "pred_vertices_2d" in preds:
            # Left hand meshes are mirrored, so flip the triangle winding order
            faces = faces_right if is_right else faces_right[:, [0, 2, 1]]
            vertices_cam = preds["pred_vertices"][0] + preds["pred_cam_t_full"][0]
            render_mesh(image_bgr, preds["pred_vertices_2d"][0], vertices_cam, faces, color)
        draw_joints(image_bgr, preds["pred_keypoints_2d"][0], color)

    # Draw the square crop region actually fed to HaMeR (detection box padded by
    # rescale_factor and expanded to the model aspect ratio), which matches the
    # mesh better than the tight keypoint-derived detection box.
    x1, y1, x2, y2 = [int(v) for v in ret["crop_bbox"]]
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
    label = f"{'right' if is_right else 'left'}"
    if ret["hand_score"] is not None:
        label += f" {ret['hand_score']:.2f}"
    cv2.putText(image_bgr, label, (max(0, x1), max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    parser = argparse.ArgumentParser(description="hamer-mini demo")
    parser.add_argument("--image", type=str, default="example_data/test1.jpg", help="Input image path")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory, or output image file path")
    parser.add_argument("--body-conf", type=float, default=0.5, help="Person detection score threshold")
    parser.add_argument("--hand-conf", type=float, default=0.5, help="Hand keypoint confidence threshold")
    parser.add_argument("--rescale-factor", type=float, default=1.5, help="Hand bbox padding factor")
    parser.add_argument("--mesh", action="store_true", help="Also render the projected MANO mesh overlay")
    parser.add_argument("--rotation", action="store_true",
                        help="Visualize the MANO rotations: wrist (global_orient) frame and "
                             "per-joint (hand_pose) axes, and print them as Euler angles")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = HaMeRHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=True)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    outputs = pipe.predict(image_rgb,
                           body_conf=args.body_conf,
                           hand_conf=args.hand_conf,
                           rescale_factor=args.rescale_factor,
                           return_vertices_2d=args.mesh)
    print(f"detected {len(outputs)} hand(s)")

    faces_right = np.asarray(pipe.mano_faces, dtype=np.int64)

    # Prefer the optional pyrender renderer for the mesh; fall back to OpenCV
    pyrender_mods = try_import_pyrender() if args.mesh and len(outputs) > 0 else None
    if pyrender_mods is not None:
        render_meshes_pyrender(image_bgr, outputs, faces_right, *pyrender_mods)

    for i, ret in enumerate(outputs):
        preds = ret["hamer_preds"]
        print(f"hand {i}: bbox={[round(v, 1) for v in ret['hand_bbox']]} "
              f"score={ret['hand_score']:.3f} is_right={ret['is_right']}")
        print(f"  pred_keypoints_3d: {preds['pred_keypoints_3d'].shape}")
        print(f"  pred_vertices:     {preds['pred_vertices'].shape}")
        print(f"  pred_keypoints_2d: {preds['pred_keypoints_2d'].shape}")
        if "pred_vertices_2d" in preds:
            print(f"  pred_vertices_2d:  {preds['pred_vertices_2d'].shape}")
        draw_hand(image_bgr, ret, faces_right,
                  draw_mesh=args.mesh and pyrender_mods is None)
        if args.rotation:
            draw_hand_rotations(image_bgr, ret)
            print_mano_rotations(preds)

    if os.path.splitext(args.output)[1].lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        out_path = args.output
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    else:
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, os.path.basename(args.image))
    cv2.imwrite(out_path, image_bgr)
    print(f"saved visualization to {out_path}")


if __name__ == "__main__":
    main()
