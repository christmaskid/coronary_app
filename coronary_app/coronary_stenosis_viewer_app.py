import json
import os
from datetime import datetime
from pathlib import Path
import tempfile
from itertools import product

import nibabel as nib
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d, splprep, splev
from scipy.ndimage import binary_dilation, binary_fill_holes, generate_binary_structure, label, map_coordinates
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from skimage.morphology import ball, binary_closing, skeletonize_3d


st.set_page_config(page_title="Coronary Stenosis Viewer", layout="wide")

RESULT_BUNDLE_SUBDIR = "analysis_bundles"


def apply_window(image, level, width):
    lower = level - width / 2
    upper = level + width / 2
    clipped = np.clip(image, lower, upper)
    return (clipped - lower) / (upper - lower)


@st.cache_data(show_spinner=False)
def load_nifti(uploaded_file):
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        image = nib.load(tmp_path)
        data = image.get_fdata()
        affine = image.affine.copy()
    finally:
        os.remove(tmp_path)
    return data, affine


@st.cache_data(show_spinner=False)
def load_nifti_from_path(file_path):
    image = nib.load(file_path)
    return image.get_fdata(), image.affine.copy()


def list_nifti_files(root_dir, max_results=500):
    root = Path(root_dir).expanduser()
    if not root.exists():
        return []
    files = []
    for current_root, _, filenames in os.walk(root):
        current_root = Path(current_root)
        for filename in filenames:
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                files.append(str((current_root / filename).relative_to(root)))
                if len(files) >= max_results:
                    return sorted(files)
    return sorted(files)


def list_relative_files(root_dir, suffixes=(), max_results=500):
    root = Path(root_dir).expanduser()
    if not root.exists():
        return []
    files = []
    for current_root, _, filenames in os.walk(root):
        current_root = Path(current_root)
        for filename in filenames:
            if suffixes and not any(filename.endswith(suffix) for suffix in suffixes):
                continue
            files.append(str((current_root / filename).relative_to(root)))
            if len(files) >= max_results:
                return sorted(dict.fromkeys(files))
    return sorted(dict.fromkeys(files))


def resolve_browse_path(root_dir, relative_path):
    return str(Path(root_dir).expanduser() / relative_path)


def to_json_safe(value):
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def bundle_case_name(ct_source, mask_source):
    ct_name = Path(ct_source).name if ct_source else "ct"
    mask_name = Path(mask_source).name if mask_source else "mask"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}__{ct_name}__{mask_name}".replace(os.sep, "_")


def save_analysis_bundle(output_dir, case_name, source_info, ct_data, ct_affine, mask_data, mask_affine, settings, result):
    output_path = Path(output_dir).expanduser() / RESULT_BUNDLE_SUBDIR
    output_path.mkdir(parents=True, exist_ok=True)
    bundle_path = output_path / f"{case_name}.npz"
    meta_path = output_path / f"{case_name}.json"

    angle_measurements = result["angle_measurements"]
    arrays = {
        "ct_data": ct_data,
        "ct_affine": ct_affine,
        "mask_data": mask_data,
        "mask_affine": mask_affine,
        "binary_mask": result["binary_mask"],
        "skeleton": result["skeleton"],
        "smoothed_coords_world": result["smoothed_coords_world"],
        "vote_ratio": result["vote_ratio"],
        "consensus_mask": result["consensus_mask"],
        "junction_indices_mm": result["junction_indices_mm"],
    }
    path_payload = []
    for idx, path in enumerate(result.get("paths", [])):
        prefix = f"path_{idx:02d}"
        path_payload.append(
            {
                "classification": path.get("classification", ""),
                "length": int(path.get("length", 0)),
                "start": path.get("start", []),
                "end": path.get("end", []),
                "type": path.get("type", ""),
            }
        )
        arrays[f"{prefix}__path"] = np.asarray(path.get("path", []), dtype=np.float32).reshape(-1, 3) if len(path.get("path", [])) else np.empty((0, 3), dtype=np.float32)
        arrays[f"{prefix}__smoothed_path_world"] = np.asarray(path.get("smoothed_path_world", []), dtype=np.float32).reshape(-1, 3) if len(path.get("smoothed_path_world", [])) else np.empty((0, 3), dtype=np.float32)
        arrays[f"{prefix}__segment_ids"] = np.asarray(path.get("segment_ids", []), dtype=np.int32)
        arrays[f"{prefix}__junctions_in_path"] = np.asarray(path.get("junctions_in_path", []), dtype=np.float32).reshape(-1, 3) if len(path.get("junctions_in_path", [])) else np.empty((0, 3), dtype=np.float32)
    for idx, measurement in enumerate(angle_measurements):
        prefix = f"angle_{idx:02d}"
        arrays[f"{prefix}__angle_deg"] = np.asarray([measurement["angle_deg"]], dtype=np.float32)
        arrays[f"{prefix}__straightened_volume"] = measurement["straightened_volume"]
        arrays[f"{prefix}__straightened_mask"] = measurement["straightened_mask"]
        arrays[f"{prefix}__measured_diameter_raw"] = measurement["measured_diameter_raw"]
        arrays[f"{prefix}__measured_diameter"] = measurement["measured_diameter"]
        arrays[f"{prefix}__predicted_diameter"] = measurement["predicted_diameter"]
        arrays[f"{prefix}__percent_stenosis"] = measurement["percent_stenosis"]
        arrays[f"{prefix}__severe_mask"] = measurement["severe_mask"].astype(np.uint8)
        arrays[f"{prefix}__s_top_mm"] = measurement["s_top_mm"]
        arrays[f"{prefix}__s_bot_mm"] = measurement["s_bot_mm"]
        arrays[f"{prefix}__x_b"] = measurement["x_b"]
        arrays[f"{prefix}__y_mm"] = measurement["y_mm"]
        arrays[f"{prefix}__junction_indices_mm"] = np.asarray(measurement.get("junction_indices_mm", []), dtype=np.float32)

    np.savez_compressed(bundle_path, **arrays)
    payload = {
        "case_name": case_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_info": to_json_safe(source_info),
        "settings": to_json_safe(settings),
        "paths": to_json_safe(path_payload),
        "consensus_regions": to_json_safe(result["consensus_regions"]),
        "bundle_path": str(bundle_path),
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(bundle_path), str(meta_path)


def load_analysis_bundle(bundle_file):
    data = np.load(bundle_file, allow_pickle=False)
    meta_path = Path(bundle_file).with_suffix(".json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    result = {
        "ct_data": data["ct_data"],
        "ct_affine": data["ct_affine"],
        "mask_data": data["mask_data"],
        "mask_affine": data["mask_affine"],
        "binary_mask": data["binary_mask"],
        "skeleton": data["skeleton"],
        "smoothed_coords_world": data["smoothed_coords_world"],
        "vote_ratio": data["vote_ratio"],
        "consensus_mask": data["consensus_mask"].astype(bool),
        "consensus_regions": [],
        "junction_indices_mm": data["junction_indices_mm"],
        "paths": [],
        "main_vessel": {"length": int(data["smoothed_coords_world"].shape[0])},
        "vessel_frames": None,
        "max_error": np.nan,
        "angle_measurements": [],
        "source_info": meta.get("source_info", {}),
        "settings": meta.get("settings", {}),
    }
    if "mask_data" in data.files:
        result["mask_data"] = data["mask_data"]

    path_keys = sorted({name.split("__", 1)[0] for name in data.files if name.startswith("path_")})
    path_meta_list = meta.get("paths", [])
    for idx, key in enumerate(path_keys):
        path_meta = path_meta_list[idx] if idx < len(path_meta_list) else {}
        result["paths"].append(
            {
                "classification": path_meta.get("classification", ""),
                "length": int(path_meta.get("length", 0)),
                "start": tuple(path_meta.get("start", [])),
                "end": tuple(path_meta.get("end", [])),
                "type": path_meta.get("type", ""),
                "path": data[f"{key}__path"],
                "smoothed_path_world": data[f"{key}__smoothed_path_world"],
                "segment_ids": data[f"{key}__segment_ids"],
                "junctions_in_path": data[f"{key}__junctions_in_path"],
            }
        )

    angle_keys = sorted({name.split("__", 1)[0] for name in data.files if name.startswith("angle_")})
    for key in angle_keys:
        result["angle_measurements"].append(
            {
                "angle_deg": float(data[f"{key}__angle_deg"][0]),
                "straightened_volume": data[f"{key}__straightened_volume"],
                "straightened_mask": data[f"{key}__straightened_mask"],
                "measured_diameter_raw": data[f"{key}__measured_diameter_raw"],
                "measured_diameter": data[f"{key}__measured_diameter"],
                "predicted_diameter": data[f"{key}__predicted_diameter"],
                "percent_stenosis": data[f"{key}__percent_stenosis"],
                "severe_mask": data[f"{key}__severe_mask"].astype(bool),
                "s_top_mm": data[f"{key}__s_top_mm"],
                "s_bot_mm": data[f"{key}__s_bot_mm"],
                "x_b": data[f"{key}__x_b"],
                "y_mm": data[f"{key}__y_mm"],
                "junction_indices_mm": data[f"{key}__junction_indices_mm"],
            }
        )
    if result["angle_measurements"]:
        result["consensus_regions"] = contiguous_true_regions(result["consensus_mask"])
        if result["paths"]:
            main_vessel = max(result["paths"], key=lambda item: item.get("length", 0))
            result["main_vessel"] = {"length": int(main_vessel.get("length", result["smoothed_coords_world"].shape[0]))}
        else:
            result["main_vessel"] = {"length": int(result["smoothed_coords_world"].shape[0])}
    return result


def preprocess_mask(mask_data):
    mask = (mask_data > 0).astype(np.uint8)
    mask = binary_closing(mask, footprint=ball(2))
    mask = binary_fill_holes(mask)
    components, num_features = label(mask)
    if num_features > 0:
        counts = np.bincount(components.ravel())
        counts[0] = 0
        mask = (components == counts.argmax()).astype(np.uint8)
    return mask


def build_graph(skeleton):
    graph = nx.Graph()
    voxels = np.argwhere(skeleton > 0)
    voxel_set = set(map(tuple, voxels))
    offsets = [offset for offset in product([-1, 0, 1], repeat=3) if offset != (0, 0, 0)]
    for voxel in voxels:
        voxel_t = tuple(voxel)
        for offset in offsets:
            neighbor = tuple(voxel + offset)
            if neighbor in voxel_set:
                graph.add_edge(voxel_t, neighbor)
    return graph


def deduplicate_bifurcations(bif_pts_voxels, graph, min_branch_len=10, proximity=1.5):
    if len(bif_pts_voxels) == 0:
        return []

    bif_pts_voxels = np.asarray(bif_pts_voxels)
    tree = cKDTree(bif_pts_voxels)
    visited = set()
    clusters = []

    for idx in range(len(bif_pts_voxels)):
        if idx in visited:
            continue
        cluster = set(tree.query_ball_point(bif_pts_voxels[idx], proximity))
        visited.update(cluster)
        clusters.append(list(cluster))

    def count_long_branches(bif_pt):
        branch_paths = []
        for neighbor in graph.neighbors(bif_pt):
            path = [bif_pt, neighbor]
            prev, curr = bif_pt, neighbor
            while graph.degree[curr] == 2:
                next_nodes = list(graph.neighbors(curr))
                next_node = next_nodes[0] if next_nodes[0] != prev else next_nodes[1]
                path.append(next_node)
                prev, curr = curr, next_node
            branch_paths.append(path)
        return sum(len(path) >= min_branch_len for path in branch_paths)

    return [sorted([tuple(bif_pts_voxels[i]) for i in cluster], key=count_long_branches, reverse=True)[0] for cluster in clusters]


def remove_short_appendages(endpoints, junctions, graph, min_branch_len=10):
    graph_pruned = graph.copy()
    edges_to_remove = set()

    for endpoint in endpoints:
        path = [endpoint]
        prev = None
        curr = endpoint
        while True:
            neighbors = [n for n in graph_pruned.neighbors(curr) if n != prev]
            if len(neighbors) == 0:
                break
            next_node = neighbors[0]
            if graph_pruned.degree[next_node] > 2:
                break
            path.append(next_node)
            if next_node in endpoints and next_node != endpoint:
                break
            prev, curr = curr, next_node

        if len(path) < min_branch_len:
            for idx in range(len(path) - 1):
                edges_to_remove.add(tuple(sorted((path[idx], path[idx + 1]))))

    for edge in edges_to_remove:
        if graph_pruned.has_edge(edge[0], edge[1]):
            graph_pruned.remove_edge(edge[0], edge[1])

    graph_pruned.remove_nodes_from(list(nx.isolates(graph_pruned)))
    new_endpoints = [node for node in graph_pruned.nodes if graph_pruned.degree(node) == 1]
    new_junctions = [node for node in graph_pruned.nodes if graph_pruned.degree(node) > 2]
    return graph_pruned, new_endpoints, new_junctions


def smooth_vessel_path(path_coords, smoothing_factor=5):
    if len(path_coords) < 4:
        return path_coords
    try:
        tck, _ = splprep(path_coords.T, s=smoothing_factor)
        u_fine = np.linspace(0, 1, len(path_coords) * 2)
        return np.stack(splev(u_fine, tck), axis=1)
    except Exception:
        return path_coords


def extract_non_overlapping_segments(graph, endpoints, junctions):
    key_nodes = set(endpoints + junctions)
    visited_edges = set()
    segments = []

    def trace_segment(start, neighbor):
        path = [start, neighbor]
        prev = start
        curr = neighbor
        while curr not in key_nodes:
            next_nodes = [n for n in graph.neighbors(curr) if n != prev]
            if len(next_nodes) != 1:
                break
            next_node = next_nodes[0]
            path.append(next_node)
            prev, curr = curr, next_node
        return path

    for node in key_nodes:
        for neighbor in graph.neighbors(node):
            edge = tuple(sorted((node, neighbor)))
            if edge in visited_edges:
                continue
            path = trace_segment(node, neighbor)
            for idx in range(len(path) - 1):
                visited_edges.add(tuple(sorted((path[idx], path[idx + 1]))))
            segments.append(
                {
                    "path": path,
                    "start": path[0],
                    "end": path[-1],
                    "length": len(path),
                    "neighbors": [],
                    "is_leaf": (path[0] in endpoints or path[-1] in endpoints),
                }
            )
    return segments


def build_segment_graph(segments):
    segment_graph = nx.Graph()
    for idx, seg in enumerate(segments):
        segment_graph.add_node(idx, segment=seg)

    node_to_segments = {}
    for idx, seg in enumerate(segments):
        for pt in [seg["start"], seg["end"]]:
            node_to_segments.setdefault(pt, []).append(idx)

    for pt, seg_ids in node_to_segments.items():
        for i in range(len(seg_ids)):
            for j in range(i + 1, len(seg_ids)):
                segment_graph.add_edge(seg_ids[i], seg_ids[j], junction=pt)
                segments[seg_ids[i]]["neighbors"].append(seg_ids[j])
                segments[seg_ids[j]]["neighbors"].append(seg_ids[i])
    return segment_graph


def find_segment_path(segment_graph, segments, seg_start_ids, seg_end_ids):
    best_path = None
    best_length = float("inf")
    for s in seg_start_ids:
        for t in seg_end_ids:
            if s == t:
                continue
            try:
                path = nx.shortest_path(segment_graph, s, t)
            except nx.NetworkXNoPath:
                continue
            length = sum(segments[i]["length"] for i in path) - len(path) + 1
            if length < best_length:
                best_length = length
                best_path = path
    return best_path


def stitch_segments(segment_indices, segments, path_type, affine):
    full_path = []
    junctions_in_path = []

    if len(segment_indices) == 1:
        full_path = segments[segment_indices[0]]["path"]
    else:
        seg_0 = segments[segment_indices[0]]
        seg_1 = segments[segment_indices[1]]
        if seg_0["end"] == seg_1["start"]:
            full_path = seg_0["path"] + seg_1["path"][1:]
        elif seg_0["end"] == seg_1["end"]:
            full_path = seg_0["path"] + seg_1["path"][-2::-1]
        elif seg_0["start"] == seg_1["start"]:
            full_path = seg_0["path"][::-1] + seg_1["path"][1:]
        elif seg_0["start"] == seg_1["end"]:
            full_path = seg_0["path"][::-1] + seg_1["path"][-2::-1]
        else:
            raise ValueError("Disconnected segments")

        for seg_id in segment_indices[2:]:
            seg = segments[seg_id]
            if full_path[-1] == seg["start"]:
                full_path.extend(seg["path"][1:])
            elif full_path[-1] == seg["end"]:
                full_path.extend(seg["path"][-2::-1])
            else:
                raise ValueError("Disconnected segments")
            junctions_in_path.append(full_path[-1])
        if junctions_in_path:
            junctions_in_path = junctions_in_path[:-1]

    full_path_world = nib.affines.apply_affine(affine, np.asarray(full_path))
    return {
        "segment_ids": segment_indices,
        "path": full_path,
        "length": len(full_path),
        "start": full_path[0],
        "end": full_path[-1],
        "junctions_in_path": junctions_in_path,
        "type": path_type,
        "smoothed_path_world": smooth_vessel_path(full_path_world),
    }


def extract_paths_from_segments(segment_graph, segments, endpoints, junctions, affine, min_path_length=20, modes=(1, 2, 3)):
    point_to_segments = {}
    all_paths = []

    for idx, seg in enumerate(segments):
        for pt in [seg["start"], seg["end"]]:
            point_to_segments.setdefault(pt, []).append(idx)

    def build_path(pt1, pt2, path_type):
        seg_start_ids = point_to_segments.get(pt1, [])
        seg_end_ids = point_to_segments.get(pt2, [])
        seg_path_ids = find_segment_path(segment_graph, segments, seg_start_ids, seg_end_ids)
        if seg_path_ids is None:
            return None
        full_path = stitch_segments(seg_path_ids, segments, path_type, affine)
        if full_path["length"] < min_path_length:
            return None
        return full_path

    if 1 in modes:
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                path = build_path(endpoints[i], endpoints[j], "endpoint_to_endpoint")
                if path is not None:
                    all_paths.append(path)

    if 2 in modes:
        for ep in endpoints:
            for jn in junctions:
                path = build_path(ep, jn, "endpoint_to_junction")
                if path is not None:
                    all_paths.append(path)

    if 3 in modes:
        for i in range(len(junctions)):
            for j in range(i + 1, len(junctions)):
                path = build_path(junctions[i], junctions[j], "junction_to_junction")
                if path is not None:
                    all_paths.append(path)

    return all_paths


def remove_overlapping_paths(paths, overlap_ratio_threshold=0.8):
    unique_paths = []
    for path in sorted(paths, key=lambda item: item["length"], reverse=True):
        path_set = set(path["path"])
        duplicate = False
        for existing in unique_paths:
            existing_set = set(existing["path"])
            overlap = len(path_set.intersection(existing_set))
            overlap_ratio = overlap / min(len(path_set), len(existing_set))
            if overlap_ratio > overlap_ratio_threshold:
                duplicate = True
                break
        if not duplicate:
            unique_paths.append(path)
    return unique_paths


def mark_path_classification(paths):
    if not paths:
        return paths, None
    main_vessel = max(paths, key=lambda item: item["length"])
    for path in paths:
        if path == main_vessel:
            path["classification"] = "main_vessel"
        elif path["length"] > main_vessel["length"] * 0.5:
            path["classification"] = "major_branch"
        elif path["length"] > main_vessel["length"] * 0.2:
            path["classification"] = "minor_branch"
        else:
            path["classification"] = "small_branch"
    return paths, main_vessel


def compute_tangents(points, window_size=5):
    tangents = np.gradient(points, axis=0)
    kernel = np.ones(window_size) / window_size
    tangents = np.array([np.convolve(tangents[:, i], kernel, mode="same") for i in range(3)]).T
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return tangents / (norms + 1e-8)


def rotation_matrix(axis, theta):
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    k = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    eye = np.eye(3)
    return eye + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def compute_fixed_frames(tangents, global_up=np.array([0, 0, 1])):
    frames = []
    prev_n = None
    prev_t = None
    for tangent in tangents:
        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
        if prev_n is None:
            normal = np.cross(global_up, tangent)
            if np.linalg.norm(normal) < 1e-6:
                normal = np.cross(np.array([0, 1, 0]), tangent)
        else:
            axis = np.cross(prev_t, tangent)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-8:
                normal = prev_n
            else:
                axis /= axis_norm
                angle = np.arccos(np.clip(np.dot(prev_t, tangent), -1.0, 1.0))
                normal = rotation_matrix(axis, angle) @ prev_n
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        binormal = np.cross(tangent, normal)
        binormal = binormal / (np.linalg.norm(binormal) + 1e-8)
        frames.append((tangent, normal, binormal))
        prev_n = normal
        prev_t = tangent
    return np.stack(frames)


def smooth_frames(frames, window_size=5):
    t, n, b = frames[:, 0], frames[:, 1], frames[:, 2]
    kernel = np.ones(window_size) / window_size
    n_smooth = np.array([np.convolve(n[:, i], kernel, mode="same") for i in range(3)]).T
    b_smooth = np.array([np.convolve(b[:, i], kernel, mode="same") for i in range(3)]).T
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)
    n_smooth = n_smooth / (np.linalg.norm(n_smooth, axis=1, keepdims=True) + 1e-8)
    b_smooth = b_smooth / (np.linalg.norm(b_smooth, axis=1, keepdims=True) + 1e-8)
    return np.stack([t, n_smooth, b_smooth], axis=1)


def enforce_frame_continuity(frames):
    continuous_frames = [frames[0]]
    for idx in range(1, len(frames)):
        t, n, b = frames[idx]
        _, prev_n, prev_b = continuous_frames[-1]
        if np.dot(n, prev_n) < 0 or np.dot(b, prev_b) < 0:
            n, b = -n, -b
        continuous_frames.append((t, n, b))
    return np.array(continuous_frames)


def generate_frames_for_vessel(vessel_coords, tangent_smooth_window=5, frame_smooth_window=5):
    tangents = compute_tangents(vessel_coords, window_size=tangent_smooth_window)
    frames = compute_fixed_frames(tangents)
    frames = smooth_frames(frames, window_size=frame_smooth_window)
    frames = enforce_frame_continuity(frames)
    orthogonality = []
    for t, n, b in frames:
        orthogonality.append(max(abs(np.dot(t, n)), abs(np.dot(t, b)), abs(np.dot(n, b))))
    return frames, float(np.max(orthogonality))


def refined_mask(mask, volume, threshold_p=0.4, dilation_iters=2, dilation_structure=None):
    n_slices, height, width = mask.shape
    center_point = (height // 2, width // 2)
    if dilation_structure is None:
        dilation_structure = generate_binary_structure(2, 1)
    refined = np.zeros_like(mask, dtype=np.uint8)

    for idx in range(n_slices):
        slice_img = volume[idx]
        slice_mask = mask[idx]
        ref_hu = slice_img[center_point[0], center_point[1]]
        valid_hu = (ref_hu - slice_img) <= threshold_p * np.abs(ref_hu)
        initial_filtered = (slice_mask == 1) & valid_hu
        dilated = binary_dilation(initial_filtered, structure=dilation_structure, iterations=dilation_iters)
        refined[idx] = (dilated & valid_hu).astype(np.uint8)
    return refined


def find_single_component(mask):
    n_slices, height, width = mask.shape
    center_point = (height // 2, width // 2)
    selected = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(n_slices):
        components, _ = label(mask[idx])
        selected_idx = components[center_point[0], center_point[1]]
        if mask[idx, center_point[0], center_point[1]] > 0:
            selected[idx] = (components == selected_idx).astype(np.uint8)
    return selected


def transform_volume(
    coords_world,
    original_affine,
    original_volume,
    mask=None,
    frames=None,
    rotated_angle=0,
    cross_section_radius=10,
    output_size=(128, 128),
    interp_mode="nearest",
    interp_order=1,
    mask_interp_mode="nearest",
    mask_interp_order=0,
):
    if frames is None:
        raise ValueError("Frames are required for straightening")

    grid_lin = np.linspace(-cross_section_radius, cross_section_radius, output_size[1])
    gx, gy = np.meshgrid(grid_lin, grid_lin, indexing="ij")
    gx = gx.flatten()
    gy = gy.flatten()
    inv_affine = np.linalg.inv(original_affine)

    transformed_volume = []
    transformed_mask = []
    for point, frame in zip(coords_world, frames):
        _, normal, binormal = frame
        theta = rotated_angle
        normal_rot = np.cos(theta) * normal + np.sin(theta) * binormal
        binormal_rot = -np.sin(theta) * normal + np.cos(theta) * binormal
        sample_points = point + np.outer(gx, normal_rot) + np.outer(gy, binormal_rot)
        voxel_coords = inv_affine @ np.hstack([sample_points, np.ones((sample_points.shape[0], 1))]).T
        voxel_coords = voxel_coords[:3]
        ct_values = map_coordinates(original_volume, voxel_coords, order=interp_order, mode=interp_mode, cval=0)
        transformed_volume.append(ct_values.reshape(output_size))
        if mask is not None:
            mask_values = map_coordinates(mask, voxel_coords, order=mask_interp_order, mode=mask_interp_mode, cval=0)
            transformed_mask.append(mask_values.reshape(output_size))

    transformed_volume = np.stack(transformed_volume, axis=0)
    if mask is None:
        return transformed_volume

    transformed_mask = np.stack(transformed_mask, axis=0)
    transformed_mask = refined_mask(transformed_mask, transformed_volume)
    transformed_mask = find_single_component(transformed_mask)
    return transformed_volume, transformed_mask


def extract_mask_boundaries_strict(mask_2d):
    height, width = mask_2d.shape
    top = np.full(width, np.nan)
    bottom = np.full(width, np.nan)
    center_height = height // 2
    for x in range(width):
        col = mask_2d[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            y_top = ys[0]
            y_bot = ys[-1]
            if np.all(col[y_top : y_bot + 1] == 1):
                top[x] = y_top
                bottom[x] = y_bot
        else:
            top[x] = center_height
            bottom[x] = center_height
    return top, bottom


def fill_missing_boundaries(top, bottom, resample_spacing):
    top = top.copy()
    bottom = bottom.copy()
    x_axis = np.arange(len(top)) * resample_spacing
    valid_top = ~np.isnan(top)
    valid_bottom = ~np.isnan(bottom)
    if np.sum(valid_top) >= 2:
        f_top = interp1d(x_axis[valid_top], top[valid_top], kind="linear", bounds_error=False, fill_value="extrapolate")
        top[~valid_top] = f_top(x_axis[~valid_top])
    if np.sum(valid_bottom) >= 2:
        f_bottom = interp1d(x_axis[valid_bottom], bottom[valid_bottom], kind="linear", bounds_error=False, fill_value="extrapolate")
        bottom[~valid_bottom] = f_bottom(x_axis[~valid_bottom])
    return top, bottom, x_axis


def predict_piecewise_diameter(x_axis_mm, measured_diameter_mm, control_positions_mm, resample_spacing):
    control_positions_mm = np.asarray(control_positions_mm, dtype=float)
    if control_positions_mm.size == 0:
        control_positions_mm = np.array([x_axis_mm[0], x_axis_mm[-1]], dtype=float)
    else:
        control_positions_mm = np.unique(np.clip(np.r_[x_axis_mm[0], control_positions_mm, x_axis_mm[-1]], x_axis_mm[0], x_axis_mm[-1]))
    control_indices = np.unique(np.clip(np.rint(control_positions_mm / resample_spacing).astype(int), 0, len(x_axis_mm) - 1))
    if len(control_indices) < 2:
        control_indices = np.array([0, len(x_axis_mm) - 1], dtype=int)
    predicted = np.interp(x_axis_mm, x_axis_mm[control_indices], measured_diameter_mm[control_indices])
    return np.maximum(predicted, 1e-6)


def predict_local_maximum_diameter(x_axis_mm, measured_diameter_mm, resample_spacing):
    control_positions_mm = []
    for idx in range(1, len(measured_diameter_mm) - 1):
        if measured_diameter_mm[idx] > measured_diameter_mm[idx - 1] and measured_diameter_mm[idx] > measured_diameter_mm[idx + 1]:
            control_positions_mm.append(x_axis_mm[idx])
    return predict_piecewise_diameter(x_axis_mm, measured_diameter_mm, control_positions_mm, resample_spacing)


def contiguous_true_regions(mask):
    regions = []
    start_idx = None
    for idx, value in enumerate(mask):
        if value and start_idx is None:
            start_idx = idx
        elif not value and start_idx is not None:
            regions.append((start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        regions.append((start_idx, len(mask) - 1))
    return regions


def find_closest_indices(points, path):
    indices = []
    for pt in points:
        distances = np.linalg.norm(path - pt, axis=1)
        indices.append(int(np.argmin(distances)))
    return np.asarray(indices, dtype=int)


def get_straightened_volume(
    smoothed_coords_world,
    ct_data,
    binary_mask,
    vessel_frames,
    affine,
    junction_indices_mm,
    theta=0,
    cross_section_radius=10,
    output_plane_size=(128, 128),
    interp_mode="nearest",
    interp_order=1,
    diameter_smoothing=True,
    diameter_smooth_window=11,
    diameter_smooth_polyorder=3,
    predict_diameter_mode="piecewise_linear",
    stenosis_threshold_ratio=0.5,
    resample_spacing=0.5,
):
    straightened_volume, straightened_mask = transform_volume(
        smoothed_coords_world,
        affine,
        ct_data,
        mask=binary_mask,
        frames=vessel_frames,
        rotated_angle=theta,
        cross_section_radius=cross_section_radius,
        output_size=output_plane_size,
        interp_mode=interp_mode,
        interp_order=interp_order,
    )

    straightened_volume_view = apply_window(straightened_volume[:, straightened_volume.shape[1] // 2, :].T, level=200, width=800)
    straightened_mask_view = straightened_mask[:, straightened_mask.shape[1] // 2, :].T
    s_top, s_bot = extract_mask_boundaries_strict(straightened_mask_view)
    s_top, s_bot, x_b = fill_missing_boundaries(s_top, s_bot, resample_spacing)
    y_mm = np.linspace(-cross_section_radius, cross_section_radius, output_plane_size[1])
    s_top_mm = y_mm[np.clip(s_top.astype(int), 0, len(y_mm) - 1)]
    s_bot_mm = y_mm[np.clip(s_bot.astype(int), 0, len(y_mm) - 1)]
    measured_diameter_raw = s_bot_mm - s_top_mm

    if diameter_smoothing and len(measured_diameter_raw) >= 5:
        window = min(diameter_smooth_window, len(measured_diameter_raw) if len(measured_diameter_raw) % 2 == 1 else len(measured_diameter_raw) - 1)
        window = max(window, 5)
        if window % 2 == 0:
            window -= 1
        polyorder = min(diameter_smooth_polyorder, window - 1)
        measured_diameter = savgol_filter(measured_diameter_raw, window, polyorder, mode="nearest")
    else:
        measured_diameter = measured_diameter_raw

    if predict_diameter_mode == "piecewise_linear":
        predicted_diameter = predict_piecewise_diameter(x_b, measured_diameter, junction_indices_mm, resample_spacing)
    else:
        predicted_diameter = predict_local_maximum_diameter(x_b, measured_diameter, resample_spacing)

    percent_stenosis = np.clip((1 - measured_diameter / predicted_diameter) * 100, 0, 100)
    severe_mask = percent_stenosis >= stenosis_threshold_ratio * 100

    return {
        "straightened_volume": straightened_volume,
        "straightened_mask": straightened_mask,
        "straightened_volume_view": straightened_volume_view,
        "straightened_mask_view": straightened_mask_view,
        "measured_diameter_raw": measured_diameter_raw,
        "measured_diameter": measured_diameter,
        "predicted_diameter": predicted_diameter,
        "percent_stenosis": percent_stenosis,
        "severe_mask": severe_mask,
        "s_top_mm": s_top_mm,
        "s_bot_mm": s_bot_mm,
        "x_b": x_b,
        "y_mm": y_mm,
    }


def run_coronary_analysis(ct_data, ct_affine, mask_data, mask_affine, settings):
    binary_mask = preprocess_mask(mask_data)
    skeleton = skeletonize_3d(binary_mask).astype(np.uint8)
    graph = build_graph(skeleton)
    endpoints = [node for node in graph.nodes if graph.degree(node) == 1]
    junctions = [node for node in graph.nodes if graph.degree(node) > 2]
    junctions = deduplicate_bifurcations(junctions, graph)
    graph_pruned, filtered_endpoints, filtered_junctions = remove_short_appendages(endpoints, junctions, graph, min_branch_len=settings["min_branch_len"])

    vessel_segments = extract_non_overlapping_segments(graph_pruned, filtered_endpoints, filtered_junctions)
    segment_graph = build_segment_graph(vessel_segments) if vessel_segments else nx.Graph()
    paths = extract_paths_from_segments(segment_graph, vessel_segments, filtered_endpoints, filtered_junctions, ct_affine, min_path_length=settings["min_path_length"])
    paths = remove_overlapping_paths(paths, overlap_ratio_threshold=settings["overlap_ratio_threshold"])
    paths, main_vessel = mark_path_classification(paths)
    if main_vessel is None:
        raise RuntimeError("No usable coronary vessel path found")

    smoothed_coords_world = main_vessel["smoothed_path_world"]
    vessel_frames, max_error = generate_frames_for_vessel(smoothed_coords_world)

    if len(main_vessel["junctions_in_path"]) > 0:
        junctions_world = nib.affines.apply_affine(ct_affine, np.asarray(main_vessel["junctions_in_path"]))
        junction_indices = find_closest_indices(junctions_world, smoothed_coords_world)
        junction_indices_mm = junction_indices.astype(float) * settings["resample_spacing"]
    else:
        junction_indices_mm = np.array([0.0, (len(smoothed_coords_world) - 1) * settings["resample_spacing"]], dtype=float)

    if len(junction_indices_mm) < 2:
        junction_indices_mm = np.array([0.0, (len(smoothed_coords_world) - 1) * settings["resample_spacing"]], dtype=float)

    angle_measurements = []
    for angle_deg in settings["angles_deg"]:
        theta = np.deg2rad(angle_deg)
        result = get_straightened_volume(
            smoothed_coords_world,
            ct_data,
            binary_mask,
            vessel_frames,
            ct_affine,
            junction_indices_mm,
            theta=theta,
            cross_section_radius=settings["cross_section_radius"],
            output_plane_size=settings["output_plane_size"],
            interp_mode=settings["interp_mode"],
            interp_order=settings["interp_order"],
            diameter_smoothing=True,
            diameter_smooth_window=settings["diameter_smooth_window"],
            diameter_smooth_polyorder=settings["diameter_smooth_polyorder"],
            predict_diameter_mode=settings["predict_diameter_mode"],
            stenosis_threshold_ratio=settings["stenosis_threshold_ratio"],
            resample_spacing=settings["resample_spacing"],
        )
        result["angle_deg"] = angle_deg
        result["junction_indices_mm"] = junction_indices_mm
        angle_measurements.append(result)

    vote_counts = np.sum(np.stack([item["severe_mask"].astype(int) for item in angle_measurements], axis=0), axis=0)
    vote_ratio = vote_counts / len(angle_measurements)
    consensus_mask = vote_ratio >= settings["vote_threshold"]
    consensus_regions = contiguous_true_regions(consensus_mask)

    return {
        "binary_mask": binary_mask,
        "skeleton": skeleton,
        "graph": graph,
        "paths": paths,
        "main_vessel": main_vessel,
        "smoothed_coords_world": smoothed_coords_world,
        "vessel_frames": vessel_frames,
        "max_error": max_error,
        "angle_measurements": angle_measurements,
        "vote_ratio": vote_ratio,
        "consensus_mask": consensus_mask,
        "consensus_regions": consensus_regions,
        "junction_indices_mm": junction_indices_mm,
    }


def build_input_figure(ct_data, mask_data, slice_idx, window_level, window_width):
    ct_slice = apply_window(ct_data[:, :, slice_idx], window_level, window_width)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Coronary view (Z slice {slice_idx})", "CT + mask"), horizontal_spacing=0.08)
    fig.add_trace(go.Heatmap(z=ct_slice, colorscale="Gray", showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=ct_slice, colorscale="Gray", showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=mask_data[:, :, slice_idx], colorscale="Reds", opacity=0.45, showscale=False, zmin=0, zmax=1), row=1, col=2)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def build_3d_figure(result, ct_affine):
    mask_pts = np.argwhere(result["binary_mask"] > 0)
    mask_world = nib.affines.apply_affine(ct_affine, mask_pts)
    skeleton_world = nib.affines.apply_affine(ct_affine, np.argwhere(result["skeleton"] > 0))
    centerline = result["smoothed_coords_world"]

    fig = go.Figure()
    sample = mask_world if len(mask_world) <= 15000 else mask_world[np.random.default_rng(7).choice(len(mask_world), 15000, replace=False)]
    if len(sample) > 0:
        fig.add_trace(go.Scatter3d(x=sample[:, 0], y=sample[:, 1], z=sample[:, 2], mode="markers", marker=dict(size=1, color="rgba(120,120,120,0.35)"), name="Mask"))
    if len(skeleton_world) > 0:
        fig.add_trace(go.Scatter3d(x=skeleton_world[:, 0], y=skeleton_world[:, 1], z=skeleton_world[:, 2], mode="markers", marker=dict(size=2, color="#ef4444"), name="Skeleton"))
    if len(centerline) > 0:
        fig.add_trace(go.Scatter3d(x=centerline[:, 0], y=centerline[:, 1], z=centerline[:, 2], mode="markers+lines", line=dict(color="#2563eb", width=5), marker=dict(size=2, color="#2563eb"), name="Selected centerline"))
    for idx, (start_idx, end_idx) in enumerate(result["consensus_regions"]):
        segment = centerline[int(start_idx) : int(end_idx) + 1]
        if len(segment) > 0:
            fig.add_trace(go.Scatter3d(x=segment[:, 0], y=segment[:, 1], z=segment[:, 2], mode="markers+lines", line=dict(color="#dc2626", width=6), marker=dict(size=3, color="#dc2626"), name="Consensus stenosis" if idx == 0 else None, showlegend=(idx == 0)))
    fig.update_layout(title="3D mask, skeleton, and selected centerline", scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"), height=650, margin=dict(l=0, r=0, t=45, b=0))
    return fig


def build_all_centerlines_figure(result, ct_affine):
    fig = go.Figure()

    mask_pts = np.argwhere(result["binary_mask"] > 0)
    if len(mask_pts) > 0:
        sample = mask_pts if len(mask_pts) <= 12000 else mask_pts[np.random.default_rng(7).choice(len(mask_pts), 12000, replace=False)]
        sample_world = nib.affines.apply_affine(ct_affine, sample)
        fig.add_trace(
            go.Scatter3d(
                x=sample_world[:, 0],
                y=sample_world[:, 1],
                z=sample_world[:, 2],
                mode="markers",
                marker=dict(size=1, color="gray", opacity=0.35),
                name="Mask",
            )
        )

    paths = result.get("paths", []) or []
    has_any_path_curve = False
    for idx, path in enumerate(paths):
        if not isinstance(path, dict):
            continue
        path_world = path.get("smoothed_path_world")
        if path_world is None:
            continue
        path_world = np.asarray(path_world)
        if path_world.ndim != 2 or path_world.shape[1] != 3 or len(path_world) == 0:
            continue
        has_any_path_curve = True
        is_main = path.get("classification") == "main_vessel"
        fig.add_trace(
            go.Scatter3d(
                x=path_world[:, 0],
                y=path_world[:, 1],
                z=path_world[:, 2],
                mode="lines",
                line=dict(color="#dc2626" if is_main else "rgba(37,99,235,0.35)", width=10 if is_main else 3),
                name="Main vessel" if is_main else f"Centerline {idx + 1}",
                showlegend=is_main or idx < 6,
            )
        )

    if not has_any_path_curve:
        centerline = np.asarray(result.get("smoothed_coords_world", []))
        if centerline.ndim == 2 and centerline.shape[1] == 3 and len(centerline) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=centerline[:, 0],
                    y=centerline[:, 1],
                    z=centerline[:, 2],
                    mode="markers+lines",
                    line=dict(color="#dc2626", width=10),
                    marker=dict(size=2, color="#dc2626"),
                    name="Main vessel",
                )
            )

    fig.update_layout(
        title="All extracted centerlines (main vessel highlighted in red)",
        scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
        height=700,
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


def build_straightened_cross_section_figure(measurement, settings, slice_idx=None):
    volume = measurement["straightened_volume"]
    mask = measurement["straightened_mask"]
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2
    slice_idx = int(np.clip(slice_idx, 0, volume.shape[0] - 1))
    center_point = (settings["output_plane_size"][0] // 2, settings["output_plane_size"][1] // 2)
    resample_spacing = float(settings.get("resample_spacing", 0.5))

    ct_slice = apply_window(volume[slice_idx], level=settings["window_level"], width=settings["window_width"])
    mask_slice = mask[slice_idx]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Straightened CT @ Slice {slice_idx}", "Straightened CT + Mask"), horizontal_spacing=0.08)
    fig.add_trace(go.Heatmap(z=ct_slice, colorscale="Gray", showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[center_point[1]], y=[center_point[0]], mode="markers", marker=dict(size=6, color="red"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=ct_slice, colorscale="Gray", showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=mask_slice, colorscale="Reds", opacity=0.45, showscale=False, zmin=0, zmax=1), row=1, col=2)
    fig.add_trace(go.Scatter(x=[center_point[1]], y=[center_point[0]], mode="markers", marker=dict(size=6, color="red"), showlegend=False), row=1, col=2)

    fig.update_layout(
        title=f"Straightened main vessel cross-sectional viewer | Slice {slice_idx}/{volume.shape[0] - 1} | Position: {slice_idx * resample_spacing:.1f} mm",
        margin=dict(l=20, r=20, t=70, b=80),
        height=560,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    return fig


def build_multiangle_longitudinal_figure(result, settings):
    angle_measurements = result.get("angle_measurements", [])
    if not angle_measurements:
        return go.Figure()

    rows = len(angle_measurements)
    subplot_titles = []
    for measurement in angle_measurements:
        severe_count = int(np.sum(measurement["severe_mask"]))
        severe_ratio = severe_count / max(len(measurement["severe_mask"]), 1) * 100
        subplot_titles.append(f"Angle {measurement['angle_deg']:.0f}° longitudinal view | Severe slices: {severe_count} ({severe_ratio:.1f}%)")

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=subplot_titles)

    for row_idx, measurement in enumerate(angle_measurements, start=1):
        x_b = measurement["x_b"]
        y_mm = measurement.get("y_mm")
        if y_mm is None:
            y_mm = np.linspace(-settings["cross_section_radius"], settings["cross_section_radius"], settings["output_plane_size"][1])

        if "straightened_volume_view" in measurement:
            view_img = measurement["straightened_volume_view"]
        else:
            view_img = apply_window(
                measurement["straightened_volume"][:, measurement["straightened_volume"].shape[1] // 2, :].T,
                level=settings["window_level"],
                width=settings["window_width"],
            )

        severe_mask = measurement["severe_mask"]
        normal_mask = ~severe_mask

        fig.add_trace(
            go.Heatmap(z=view_img, x=x_b, y=y_mm, colorscale="Gray", showscale=False, zmin=0, zmax=1),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_b[normal_mask], y=measurement["s_top_mm"][normal_mask], mode="lines", line=dict(color="green", width=2), name="Normal boundary", showlegend=(row_idx == 1)),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_b[normal_mask], y=measurement["s_bot_mm"][normal_mask], mode="lines", line=dict(color="green", width=2), showlegend=False),
            row=row_idx,
            col=1,
        )

        for region_idx, (start_idx, end_idx) in enumerate(contiguous_true_regions(severe_mask)):
            region_x = x_b[start_idx : end_idx + 1]
            region_top = measurement["s_top_mm"][start_idx : end_idx + 1]
            region_bot = measurement["s_bot_mm"][start_idx : end_idx + 1]
            fig.add_trace(
                go.Scatter(
                    x=region_x,
                    y=region_top,
                    mode="lines",
                    line=dict(color="red", width=3),
                    name="Stenotic boundary" if (row_idx == 1 and region_idx == 0) else None,
                    showlegend=(row_idx == 1 and region_idx == 0),
                ),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=region_x, y=region_bot, mode="lines", line=dict(color="red", width=3), showlegend=False),
                row=row_idx,
                col=1,
            )

        for junction_idx in np.asarray(measurement.get("junction_indices_mm", []), dtype=float):
            fig.add_trace(
                go.Scatter(x=[junction_idx, junction_idx], y=[y_mm[0], y_mm[-1]], mode="lines", line=dict(color="orange", dash="dash"), opacity=0.8, showlegend=False),
                row=row_idx,
                col=1,
            )

        fig.update_yaxes(title_text="Radial (mm)", row=row_idx, col=1)

    fig.update_xaxes(title_text="Length along vessel (mm)", row=rows, col=1)
    fig.update_layout(title="Multi-angle straightened longitudinal views and contours", height=max(500, 300 * rows), margin=dict(l=20, r=20, t=80, b=20))
    return fig


def build_straightened_figure(measurement, settings):
    volume = measurement["straightened_volume"]
    mask = measurement["straightened_mask"]
    init_idx = volume.shape[0] // 2
    center_point = (settings["output_plane_size"][0] // 2, settings["output_plane_size"][1] // 2)
    ct_frames = [apply_window(volume[idx], level=settings["window_level"], width=settings["window_width"]) for idx in range(volume.shape[0])]
    mask_frames = [mask[idx] for idx in range(mask.shape[0])]

    fig = make_subplots(rows=2, cols=2, row_heights=[0.62, 0.38], subplot_titles=(f"Straightened CT @ Slice {init_idx}", "Straightened CT + Mask", "Longitudinal View", "Diameter and Stenosis Profile"), vertical_spacing=0.16, horizontal_spacing=0.08)
    fig.add_trace(go.Heatmap(z=ct_frames[init_idx], colorscale="Gray", showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[center_point[1]], y=[center_point[0]], mode="markers", marker=dict(size=5, color="red"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=ct_frames[init_idx], colorscale="Gray", showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=mask_frames[init_idx], colorscale="Reds", opacity=0.4, showscale=False, zmin=0, zmax=1), row=1, col=2)
    fig.add_trace(go.Scatter(x=[center_point[1]], y=[center_point[0]], mode="markers", marker=dict(size=5, color="red"), showlegend=False), row=1, col=2)

    sagittal = volume[:, :, settings["output_plane_size"][1] // 2].T
    sagittal = apply_window(sagittal, level=settings["window_level"], width=settings["window_width"])
    axis_mm = np.linspace(-settings["cross_section_radius"], settings["cross_section_radius"], settings["output_plane_size"][1])
    x_b = measurement["x_b"]
    fig.add_trace(go.Heatmap(z=sagittal, x=x_b, y=axis_mm, colorscale="Gray", showscale=False, zmin=0, zmax=1), row=2, col=1)
    x_pos_init = x_b[init_idx] if init_idx < len(x_b) else x_b[-1]
    fig.add_trace(go.Scatter(x=[x_pos_init, x_pos_init], y=[axis_mm[0], axis_mm[-1]], mode="lines", line=dict(color="red", width=2, dash="dash"), name="Current Slice"), row=2, col=1)

    normal_mask = ~measurement["severe_mask"]
    fig.add_trace(go.Scatter(x=x_b[normal_mask], y=measurement["s_top_mm"][normal_mask], mode="lines", line=dict(color="green", width=2), name="Normal"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_b[normal_mask], y=measurement["s_bot_mm"][normal_mask], mode="lines", line=dict(color="green", width=2), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_b[measurement["severe_mask"]], y=measurement["s_top_mm"][measurement["severe_mask"]], mode="lines", line=dict(color="red", width=2.5), name="Stenotic (>=50%)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_b[measurement["severe_mask"]], y=measurement["s_bot_mm"][measurement["severe_mask"]], mode="lines", line=dict(color="red", width=2.5), showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=x_b, y=measurement["measured_diameter_raw"], mode="lines", line=dict(color="black", width=1), name="Measured Diameter (raw)"), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_b, y=measurement["measured_diameter"], mode="lines", line=dict(color="blue", width=2), name="Measured Diameter (smoothed)"), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_b, y=measurement["predicted_diameter"], mode="lines", line=dict(color="green", width=2, dash="dash"), name="Predicted Diameter"), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_b, y=np.where(measurement["severe_mask"], measurement["measured_diameter"], np.nan), mode="lines", line=dict(color="rgba(255,0,0,0)"), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_b, y=np.where(measurement["severe_mask"], measurement["predicted_diameter"], np.nan), mode="lines", line=dict(color="rgba(255,0,0,0)"), fill="tonexty", fillcolor="rgba(255,0,0,0.3)", name="Severe Stenosis"), row=2, col=2)

    for idx in np.asarray(measurement.get("junction_indices_mm", []), dtype=float):
        fig.add_vline(x=idx, row=2, col=1, line_color="orange", line_dash="dash", opacity=0.8)
        fig.add_vline(x=idx, row=2, col=2, line_color="orange", line_dash="dash", opacity=0.8)

    fig.update_layout(title="Straightened vessel viewer", height=900)
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, autorange="reversed", row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, autorange="reversed", row=1, col=2)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_xaxes(title_text="Vessel Length (mm)", row=2, col=1)
    fig.update_yaxes(title_text="Radial Distance (mm)", row=2, col=1)
    fig.update_xaxes(title_text="Vessel Length (mm)", row=2, col=2)
    fig.update_yaxes(title_text="Diameter (mm)", row=2, col=2)
    return fig


def build_consensus_figure(result, ct_affine):
    centerline = result["smoothed_coords_world"]
    ref_measurement = result["angle_measurements"][0]
    ref_x = ref_measurement["x_b"]

    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene", "rowspan": 2}, {"type": "xy"}], [None, {"type": "xy"}]], subplot_titles=("3D centerline + consensus", "Per-slice stenosis vote across angles", "Consensus diameter profile"))

    mask_pts = np.argwhere(result["binary_mask"] > 0)
    mask_world = nib.affines.apply_affine(ct_affine, mask_pts)
    skeleton_world = nib.affines.apply_affine(ct_affine, np.argwhere(result["skeleton"] > 0))
    sample = mask_world if len(mask_world) <= 12000 else mask_world[np.random.default_rng(7).choice(len(mask_world), 12000, replace=False)]

    if len(sample) > 0:
        fig.add_trace(go.Scatter3d(x=sample[:, 0], y=sample[:, 1], z=sample[:, 2], mode="markers", marker=dict(size=1, color="gray", opacity=0.35), name="Mask"), row=1, col=1)
    if len(skeleton_world) > 0:
        fig.add_trace(go.Scatter3d(x=skeleton_world[:, 0], y=skeleton_world[:, 1], z=skeleton_world[:, 2], mode="markers", marker=dict(size=2, color="#444444"), name="Skeleton"), row=1, col=1)
    if len(centerline) > 0:
        fig.add_trace(go.Scatter3d(x=centerline[:, 0], y=centerline[:, 1], z=centerline[:, 2], mode="markers+lines", line=dict(color="#2563eb", width=5), marker=dict(size=2, color="#2563eb"), name="Selected centerline"), row=1, col=1)

    for idx, (start_idx, end_idx) in enumerate(result["consensus_regions"]):
        segment = centerline[int(start_idx) : int(end_idx) + 1]
        if len(segment) > 0:
            region_percent = np.nanmean(np.stack([m["percent_stenosis"] for m in result["angle_measurements"]], axis=0)[:, int(start_idx) : int(end_idx) + 1])
            fig.add_trace(go.Scatter3d(x=segment[:, 0], y=segment[:, 1], z=segment[:, 2], mode="markers+lines", line=dict(color="#dc2626", width=6), marker=dict(size=3, color="#dc2626"), name="Consensus stenosis" if idx == 0 else None, showlegend=(idx == 0)), row=1, col=1)
            cx, cy, cz = np.mean(segment, axis=0)
            fig.add_trace(go.Scatter3d(x=[cx], y=[cy], z=[cz], mode="text", text=[f"{region_percent:.1f}%"], textfont=dict(color="black", size=12), showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=ref_x, y=result["vote_ratio"] * 100, mode="lines", line=dict(color="orange", width=2), name="Severe vote ratio (%)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ref_x, y=np.full_like(ref_x, 50), mode="lines", line=dict(color="red", width=1, dash="dash"), name="Vote threshold (50%)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ref_x, y=np.where(result["consensus_mask"], result["vote_ratio"] * 100, np.nan), mode="lines", line=dict(color="red", width=2), fill="tozeroy", fillcolor="rgba(255,0,0,0.25)", name="Consensus stenosis"), row=1, col=2)

    fig.add_trace(go.Scatter(x=ref_x, y=ref_measurement["measured_diameter"], mode="lines", line=dict(color="blue", width=2), name="Measured Diameter"), row=2, col=2)
    fig.add_trace(go.Scatter(x=ref_x, y=ref_measurement["predicted_diameter"], mode="lines", line=dict(color="green", width=2, dash="dash"), name="Reference Diameter (angle 0°)"), row=2, col=2)
    for idx, (start_idx, end_idx) in enumerate(result["consensus_regions"]):
        region_x = ref_x[int(start_idx) : int(end_idx) + 1]
        region_measured = ref_measurement["measured_diameter"][int(start_idx) : int(end_idx) + 1]
        region_predicted = ref_measurement["predicted_diameter"][int(start_idx) : int(end_idx) + 1]
        polygon_x = np.concatenate([region_x, region_x[::-1]])
        polygon_y = np.concatenate([region_measured, region_predicted[::-1]])
        fig.add_trace(go.Scatter(x=polygon_x, y=polygon_y, mode="lines", line=dict(color="rgba(255,0,0,0)"), fill="toself", fillcolor="rgba(255,0,0,0.25)", name="Consensus stenosis" if idx == 0 else None, showlegend=(idx == 0), hoverinfo="skip"), row=2, col=2)

    fig.update_xaxes(title_text="Length along vessel (mm)", row=1, col=2)
    fig.update_yaxes(title_text="Vote Ratio (%)", row=1, col=2)
    fig.update_xaxes(title_text="Length along vessel (mm)", row=2, col=2)
    fig.update_yaxes(title_text="Diameter (mm)", row=2, col=2)
    fig.update_layout(title="Consensus stenosis summary", height=850)
    return fig


st.title("Coronary Stenosis Viewer")
st.caption("Input CT mask, centerline extraction, straightening, diameter profile, and multi-angle stenosis consensus for coronary arteries.")
results_default_dir = str(Path(os.getcwd()) / "viewer_results")

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_inputs" not in st.session_state:
    st.session_state.analysis_inputs = {}
if "saved_bundle_path" not in st.session_state:
    st.session_state.saved_bundle_path = None
if "save_log" not in st.session_state:
    st.session_state.save_log = []


def render_results(result, ct_data, ct_affine, window_level, window_width, settings):
    metric_cols = st.columns(4)
    metric_cols[0].metric("Mask voxels", f"{int(result['binary_mask'].sum()):,}")
    metric_cols[1].metric("Extracted paths", f"{len(result['paths'])}")
    metric_cols[2].metric("Main vessel length", f"{result['main_vessel']['length']} voxels")
    metric_cols[3].metric("Frame orthogonality error", f"{result.get('max_error', np.nan):.4f}")

    tab_visual, tab_table = st.tabs(["Visualization", "Saved"])

    with tab_visual:
        st.subheader("Coronary view CT and CT + mask (Z-axis slicing)")
        slice_idx = st.slider("Preview Z slice", min_value=0, max_value=ct_data.shape[2] - 1, value=ct_data.shape[2] // 2, key="preview_slice_z")
        st.plotly_chart(build_input_figure(ct_data, result["binary_mask"], slice_idx, window_level, window_width), use_container_width=True)

        st.subheader("All extracted centerlines (main vessel highlighted)")
        st.plotly_chart(build_all_centerlines_figure(result, ct_affine), use_container_width=True)

        st.subheader("Straightened main vessel cross-sectional viewer (CT and CT + mask)")
        if not result.get("angle_measurements"):
            st.info("No straightening data is available in this result bundle.")
        else:
            angle_labels = [f"{measurement['angle_deg']:.0f}°" for measurement in result["angle_measurements"]]
            angle_idx = st.selectbox("Cross-sectional angle", options=list(range(len(angle_labels))), format_func=lambda idx: angle_labels[idx], index=0, key="cross_section_angle_idx")
            selected_measurement = dict(result["angle_measurements"][angle_idx])
            selected_measurement["junction_indices_mm"] = result.get("junction_indices_mm", selected_measurement.get("junction_indices_mm", []))
            cross_slice_idx = st.slider(
                "Cross-sectional slice",
                min_value=0,
                max_value=selected_measurement["straightened_volume"].shape[0] - 1,
                value=selected_measurement["straightened_volume"].shape[0] // 2,
                key="cross_section_slice_idx",
            )
            st.plotly_chart(build_straightened_cross_section_figure(selected_measurement, settings, slice_idx=cross_slice_idx), use_container_width=True)

        st.subheader("Multi-angle straightened longitudinal views and contours")
        if not result.get("angle_measurements"):
            st.info("No multi-angle data is available in this result bundle.")
        else:
            st.plotly_chart(build_multiangle_longitudinal_figure(result, settings), use_container_width=True)

        st.subheader("Multi-angle consensus summary")
        if not result.get("angle_measurements"):
            st.info("No consensus data is available in this result bundle.")
        else:
            st.plotly_chart(build_consensus_figure(result, ct_affine), use_container_width=True)
            b1, b2, b3 = st.columns(3)
            b1.metric("Consensus stenotic slices", f"{int(np.sum(result['consensus_mask']))} / {len(result['consensus_mask'])}")
            b2.metric("Angles evaluated", f"{len(result['angle_measurements'])}")
            b3.metric("Vote threshold", f"{settings.get('vote_threshold', 0.5) * 100:.0f}%")

            consensus_rows = []
            for region_idx, (start_idx, end_idx) in enumerate(result["consensus_regions"]):
                region_vote = float(np.nanmean(result["vote_ratio"][int(start_idx) : int(end_idx) + 1]) * 100)
                consensus_rows.append(
                    {
                        "region": region_idx + 1,
                        "slice_start": int(start_idx),
                        "slice_end": int(end_idx),
                        "vote_ratio_%": round(region_vote, 1),
                    }
                )
            if consensus_rows:
                st.dataframe(consensus_rows, use_container_width=True)

    with tab_table:
        st.subheader("Extracted path summary")
        path_rows = []
        for path in result["paths"] if result.get("paths") else []:
            if isinstance(path, dict):
                path_rows.append(
                    {
                        "classification": path.get("classification", ""),
                        "length_voxels": int(path.get("length", 0)),
                        "start": str(path.get("start", "")),
                        "end": str(path.get("end", "")),
                        "junctions_in_path": int(len(path.get("junctions_in_path", []))) if isinstance(path.get("junctions_in_path", []), list) else int(path.get("junctions_in_path", 0)),
                        "type": path.get("type", ""),
                    }
                )
        if path_rows:
            st.dataframe(path_rows, use_container_width=True)
        else:
            st.info("No path summary available in this result.")

        if st.session_state.saved_bundle_path and Path(st.session_state.saved_bundle_path).exists():
            st.subheader("Saved result bundle")
            st.code(st.session_state.saved_bundle_path)
            bundle_bytes = Path(st.session_state.saved_bundle_path).read_bytes()
            st.download_button(
                "Download saved bundle",
                data=bundle_bytes,
                file_name=Path(st.session_state.saved_bundle_path).name,
                mime="application/octet-stream",
                use_container_width=True,
            )


def append_save_log(message):
    st.session_state.save_log.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {message}")


with st.sidebar:
    st.header("Input Source")
    input_mode = st.radio("Choose input mode", ["Upload local files", "Browse remote server", "Open saved result"], key="input_mode")
    ct_upload = None
    mask_upload = None
    ct_path = None
    mask_path = None
    selected_bundle = None
    save_after_run = False

    if input_mode == "Upload local files":
        st.header("Inputs")
        ct_upload = st.file_uploader("CT NIfTI", type=["nii", "gz"], key="ct_upload")
        mask_upload = st.file_uploader("Coronary mask NIfTI", type=["nii", "gz"], key="mask_upload")
        save_after_run = st.checkbox("Save analysis bundle", value=True, key="save_after_run")
        if save_after_run:
            results_dir = st.text_input("Save results to", value=results_default_dir, key="results_dir")
    elif input_mode == "Browse remote server":
        st.header("Server files")
        browse_root = st.text_input("Browse root directory", value="/data", key="browse_root")
        server_files = list_nifti_files(browse_root)
        if not server_files:
            st.warning("No NIfTI files found under the selected browse root.")
        else:
            ct_rel_path = st.selectbox("CT file on server", server_files, key="ct_path")
            mask_rel_path = st.selectbox("Mask file on server", server_files, key="mask_path")
            ct_path = resolve_browse_path(browse_root, ct_rel_path)
            mask_path = resolve_browse_path(browse_root, mask_rel_path)
        save_after_run = st.checkbox("Save analysis bundle", value=True, key="save_after_run")
        if save_after_run:
            results_dir = st.text_input("Save results to", value=results_default_dir, key="results_dir")
    else:
        st.header("Saved results")
        results_dir = st.text_input("Open results from directory", value=results_default_dir, key="results_dir")
        bundle_root = Path(results_dir).expanduser() / RESULT_BUNDLE_SUBDIR
        saved_bundles = list_relative_files(bundle_root, suffixes=(".npz",))
        if not saved_bundles:
            st.info("No saved bundles found yet.")
        else:
            selected_bundle_rel = st.selectbox("Saved bundle", saved_bundles, key="selected_bundle")
            selected_bundle = resolve_browse_path(bundle_root, selected_bundle_rel)

    run_button = st.button("Run coronary analysis", type="primary", use_container_width=True, key="run_button")

    st.header("Analysis")
    window_level = st.number_input("Window level", value=200.0, step=10.0, key="window_level")
    window_width = st.number_input("Window width", value=800.0, step=25.0, key="window_width")
    resample_spacing = st.number_input("Resample spacing (mm)", min_value=0.1, value=0.5, step=0.1, key="resample_spacing")
    cross_section_radius = st.number_input("Cross-section radius (mm)", min_value=1.0, value=10.0, step=1.0, key="cross_section_radius")
    output_size = st.number_input("Output plane size", min_value=32, value=128, step=16, key="output_size")
    vote_threshold = st.slider("Consensus vote threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05, key="vote_threshold")
    angles_text = st.text_input("Angles in degrees", value="0,30,60,90,120,150", key="angles_text")
    min_path_length = st.number_input("Minimum path length", min_value=5, value=20, step=1, key="min_path_length")
    min_branch_len = st.number_input("Minimum branch length", min_value=3, value=10, step=1, key="min_branch_len")
    overlap_ratio_threshold = st.slider("Overlap ratio threshold", min_value=0.5, max_value=1.0, value=0.95, step=0.01, key="overlap_ratio_threshold")
    refine_threshold = st.slider("Mask refinement threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05, key="refine_threshold")
    stenosis_threshold_ratio = st.slider("Severe stenosis threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05, key="stenosis_threshold_ratio")
    diameter_mode = st.radio("Diameter reference", ["piecewise_linear", "local_maximum"], index=0, key="diameter_mode")
    interp_mode = st.selectbox("Interpolation mode", ["nearest", "linear", "reflect", "constant"], index=0, key="interp_mode")
    interp_order = st.slider("Interpolation order", min_value=0, max_value=3, value=1, step=1, key="interp_order")


    st.divider()
    st.subheader("Save log")
    if st.session_state.save_log:
        st.text_area("Recent saved results", value="\n".join(reversed(st.session_state.save_log[-8:])), height=180, label_visibility="collapsed")
    else:
        st.caption("No saved results yet.")


loaded_bundle_result = None
if input_mode == "Open saved result" and selected_bundle:
    try:
        loaded_bundle_result = load_analysis_bundle(selected_bundle)
        st.session_state.analysis_result = loaded_bundle_result
        st.session_state.analysis_inputs = {
            "ct_data": loaded_bundle_result["ct_data"],
            "ct_affine": loaded_bundle_result["ct_affine"],
            "settings": loaded_bundle_result.get("settings", {}),
        }
        st.session_state.saved_bundle_path = selected_bundle
    except Exception as exc:
        st.exception(exc)

if run_button:
    if input_mode == "Open saved result":
        st.info("Saved result mode loads the selected bundle automatically; no rerun is needed.")
        st.stop()

    if input_mode == "Upload local files":
        if ct_upload is None or mask_upload is None:
            st.error("Please upload both CT and mask files.")
            st.stop()
        ct_data, ct_affine = load_nifti(ct_upload)
        mask_data, mask_affine = load_nifti(mask_upload)
        source_info = {"ct_source": ct_upload.name, "mask_source": mask_upload.name, "mode": input_mode}
    else:
        if ct_path is None or mask_path is None:
            st.error("Please select both CT and mask files from the server.")
            st.stop()
        ct_data, ct_affine = load_nifti_from_path(ct_path)
        mask_data, mask_affine = load_nifti_from_path(mask_path)
        source_info = {"ct_source": ct_path, "mask_source": mask_path, "mode": input_mode}

    ct_data = ct_data.astype(np.float32)
    mask_data = mask_data.astype(np.uint8)
    if ct_data.shape != mask_data.shape:
        st.error(f"CT shape {ct_data.shape} does not match mask shape {mask_data.shape}.")
        st.stop()

    try:
        angles_deg = [float(value.strip()) for value in angles_text.split(",") if value.strip()]
    except ValueError:
        st.error("Angles must be a comma-separated list of numbers.")
        st.stop()
    if not angles_deg:
        angles_deg = [0, 30, 60, 90, 120, 150]

    settings = {
        "window_level": float(window_level),
        "window_width": float(window_width),
        "resample_spacing": float(resample_spacing),
        "cross_section_radius": float(cross_section_radius),
        "output_plane_size": (int(output_size), int(output_size)),
        "vote_threshold": float(vote_threshold),
        "angles_deg": angles_deg,
        "min_path_length": int(min_path_length),
        "min_branch_len": int(min_branch_len),
        "overlap_ratio_threshold": float(overlap_ratio_threshold),
        "refine_threshold": float(refine_threshold),
        "diameter_smooth_window": 11,
        "diameter_smooth_polyorder": 3,
        "predict_diameter_mode": diameter_mode,
        "interp_mode": interp_mode,
        "interp_order": int(interp_order),
        "stenosis_threshold_ratio": float(stenosis_threshold_ratio),
    }

    with st.status("🔄 Running coronary analysis...", expanded=True) as status:
        try:
            st.write("📊 Preprocessing mask and skeleton extraction...")
            status.update(label="🔄 Preprocessing mask and skeleton extraction...", state="running")
            
            st.write("🔍 Building vascular graph and extracting vessel paths...")
            status.update(label="🔄 Building vascular graph and extracting vessel paths...", state="running")
            
            st.write("➡️ Extracting vessel centerline and computing reference frames...")
            status.update(label="🔄 Extracting vessel centerline and computing reference frames...", state="running")
            
            st.write("📐 Straightening vessel segments for diameter measurement...")
            status.update(label="🔄 Straightening vessel segments for diameter measurement...", state="running")
            
            result = run_coronary_analysis(ct_data, ct_affine, mask_data, mask_affine, settings)
            
            st.write("✅ Analysis complete!")
            status.update(label="✅ Analysis complete!", state="complete")
        except Exception as exc:
            status.update(label="❌ Analysis failed", state="error")
            st.exception(exc)
            st.stop()

    st.session_state.analysis_result = result
    st.session_state.analysis_inputs = {
        "ct_data": ct_data,
        "ct_affine": ct_affine,
        "mask_data": mask_data,
        "mask_affine": mask_affine,
        "settings": settings,
    }

    st.session_state.saved_bundle_path = None
    if save_after_run:
        case_name = bundle_case_name(source_info["ct_source"], source_info["mask_source"])
        try:
            bundle_path, meta_path = save_analysis_bundle(results_dir, case_name, source_info, ct_data, ct_affine, mask_data, mask_affine, settings, result)
            st.success(f"Saved analysis bundle to {bundle_path}")
            st.caption(f"Metadata: {meta_path}")
            st.session_state.saved_bundle_path = bundle_path
            append_save_log(bundle_path)
        except Exception as exc:
            st.warning(f"Analysis finished, but saving failed: {exc}")

active_result = st.session_state.analysis_result
if active_result is not None:
    active_ct = st.session_state.analysis_inputs.get("ct_data")
    active_affine = st.session_state.analysis_inputs.get("ct_affine")
    active_settings = st.session_state.analysis_inputs.get("settings", {})
    if active_ct is None or active_affine is None:
        st.warning("A saved result is present, but the source image data is missing in session state.")
    else:
        display_settings = dict(active_settings)
        display_settings["window_level"] = window_level
        display_settings["window_width"] = window_width
        display_settings["resample_spacing"] = float(resample_spacing)
        display_settings["cross_section_radius"] = float(cross_section_radius)
        display_settings["output_plane_size"] = (int(output_size), int(output_size))
        display_settings["vote_threshold"] = float(vote_threshold)
        display_settings["interp_mode"] = interp_mode
        display_settings["interp_order"] = int(interp_order)
        display_settings["stenosis_threshold_ratio"] = float(stenosis_threshold_ratio)
        render_results(active_result, active_ct, active_affine, window_level, window_width, display_settings)
else:
    st.info("Choose an input mode, then run the analysis or open a saved bundle.")
