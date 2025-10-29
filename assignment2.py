import argparse
import csv
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


@dataclass
class Annotation:
    object_id: int
    flag: int
    frame_idx: int
    x: int
    y: int
    w: int
    h: int
    class_id: int

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        x1 = self.x
        y1 = self.y
        x2 = self.x + self.w
        y2 = self.y + self.h
        return x1, y1, x2, y2

    @property
    def centroid(self) -> Tuple[float, float]:
        return float(self.x + self.w / 2.0), float(self.y + self.h / 2.0)

    @property
    def area(self) -> int:
        return int(self.w * self.h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assignment 2: Bipartite Object Matching on VIRAT Frame Pairs"
    )
    parser.add_argument(
        "--index",
        type=str,
        default=os.path.join("cv_data_hw2", "index.txt"),
        help="Path to index.txt listing pairs (img1,ann1,img2,ann2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to write visualizations and logs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    # Weights and thresholds (defaults per PLAN.md)
    parser.add_argument("--iou_weight", type=float, default=0.6, help="Weight for (1 - IoU)")
    parser.add_argument(
        "--centroid_weight", type=float, default=0.4, help="Weight for normalized centroid distance"
    )
    parser.add_argument(
        "--hist_weight", type=float, default=0.2, help="Weight for appearance histogram distance"
    )
    parser.add_argument(
        "--use_appearance",
        choices=["on", "off"],
        default="on",
        help="Enable appearance term (HSV histogram distance)",
    )
    parser.add_argument(
        "--tau_centroid",
        type=float,
        default=0.7,
        help="Gating: max normalized centroid distance; above this set cost to M",
    )
    parser.add_argument(
        "--area_ratio_max",
        type=float,
        default=5.0,
        help="Gating: max area ratio between boxes; above this set cost to M",
    )
    parser.add_argument(
        "--forbid_cross_class",
        action="store_true",
        default=True,
        help="Strictly forbid cross-class matches (incl. Unknown only matches Unknown)",
    )
    parser.add_argument(
        "--tau_match",
        type=float,
        default=0.8,
        help="Post-assignment threshold; above this cost mark as unmatched",
    )
    parser.add_argument(
        "--draw_legend",
        action="store_true",
        help="Draw a small legend with match indices and classes",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Process at most this many pairs (0 = all)",
    )
    parser.add_argument(
        "--select_diverse",
        action="store_true",
        help="Select 5 diverse pairs across sources for report outputs",
    )
    parser.add_argument(
        "--color_seed",
        type=int,
        default=123,
        help="Random seed for deterministic color assignment",
    )
    # Further arguments added in later steps (appearance, weights, thresholds, etc.)
    return parser.parse_args()


def _resolve_index_path(raw_path: str) -> str:
    """
    Resolve the index file path robustly across environments (e.g., HPC).
    - Accept a direct file path, or a directory containing index.txt
    - Try common fallbacks relative to CWD and script directory
    """
    # 1) If a directory is passed, assume index.txt within
    if os.path.isdir(raw_path):
        cand = os.path.join(raw_path, "index.txt")
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    # 2) If a file path is passed and exists, use it
    if os.path.isfile(raw_path):
        return os.path.abspath(raw_path)

    # 3) Try relative to CWD: cv_data_hw2/index.txt
    cand = os.path.join(os.getcwd(), "cv_data_hw2", "index.txt")
    if os.path.isfile(cand):
        return os.path.abspath(cand)

    # 4) Try relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(script_dir, "cv_data_hw2", "index.txt")
    if os.path.isfile(cand):
        return os.path.abspath(cand)

    # 5) As a last resort, attempt a shallow search from CWD
    for root, dirs, files in os.walk(os.getcwd()):
        if "index.txt" in files and os.path.basename(root) == "cv_data_hw2":
            return os.path.abspath(os.path.join(root, "index.txt"))
        # Limit depth to avoid expensive walks on HPC
        depth = os.path.relpath(root, os.getcwd()).count(os.sep)
        if depth > 3:
            dirs[:] = []
    # If none found, return raw path to trigger a clear error
    return raw_path


def read_index(index_path: str) -> List[Tuple[str, str, str, str]]:
    index_path = _resolve_index_path(index_path)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"index file not found: {index_path}. Pass --index /path/to/cv_data_hw2/index.txt or the cv_data_hw2 directory."
        )

    pairs: List[Tuple[str, str, str, str]] = []
    with open(index_path, "r", newline="") as f:
        reader = csv.reader(f)
        line_no = 0
        for row in reader:
            line_no += 1
            if not row:
                logging.debug("Skipping empty line in index.txt at line %d", line_no)
                continue
            if len(row) != 4:
                # Some CSVs may not use commas if read as a single string; split manually
                line = row[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 4:
                    raise ValueError(
                        f"Each line must contain 4 comma-separated paths, got: {row}"
                    )
                img1, ann1, img2, ann2 = parts
            else:
                img1, ann1, img2, ann2 = [p.strip() for p in row]

            # Resolve relative paths against the index file's directory
            base_dir = os.path.dirname(index_path)
            abs_paths: List[str] = []
            for p in (img1, ann1, img2, ann2):
                ap = p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))
                if not os.path.isfile(ap):
                    raise FileNotFoundError(
                        f"Path from index does not exist: {ap} (original entry: {p})"
                    )
                abs_paths.append(ap)

            pairs.append(tuple(abs_paths))

    if len(pairs) == 0:
        raise ValueError("No pairs found in index file.")

    logging.info("Loaded %d pairs from %s", len(pairs), index_path)
    return pairs


def parse_annotation_file(path: str) -> List[Annotation]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"annotation file not found: {path}")
    annotations: List[Annotation] = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(
                    f"Annotation line must have 8 fields at {path}:{line_num}, got {len(parts)}"
                )
            vals = list(map(int, parts))
            ann = Annotation(
                object_id=vals[0],
                flag=vals[1],
                frame_idx=vals[2],
                x=vals[3],
                y=vals[4],
                w=vals[5],
                h=vals[6],
                class_id=vals[7],
            )
            # Basic sanity checks
            if ann.w <= 0 or ann.h <= 0:
                # Skip invalid boxes rather than failing the run
                logging.debug(
                    "Skipping invalid bbox (non-positive size) at %s:%d -> %s",
                    path,
                    line_num,
                    ann,
                )
                continue
            annotations.append(ann)
    logging.debug("Parsed %d annotations from %s", len(annotations), path)
    return annotations


# -------------------- BBox Utilities (Geometry) --------------------
def box_area_xyxy(xyxy: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = xyxy
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = box_area_xyxy(a)
    b_area = box_area_xyxy(b)
    union = max(1, a_area + b_area - inter)
    return float(inter) / float(union)


def centroid_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    return float(np.hypot(ax - bx, ay - by))


def normalized_centroid_distance(
    a: Tuple[float, float], b: Tuple[float, float], image_w: int, image_h: int
) -> float:
    diag = float(np.hypot(image_w, image_h))
    if diag <= 0:
        return 0.0
    return centroid_distance(a, b) / diag


def _class_compatible(c1: int, c2: int, forbid_cross: bool) -> bool:
    if not forbid_cross:
        return True
    # Strict policy: match only if exactly same class id
    return c1 == c2


def _area_ratio_ok(a_area: int, b_area: int, max_ratio: float) -> bool:
    if a_area <= 0 or b_area <= 0:
        return False
    big = max(a_area, b_area)
    small = min(a_area, b_area)
    return (big / small) <= max_ratio


def _load_image(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    h, w = img.shape[:2]
    return img, (w, h)


def _clip_box_to_image(xyxy: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return x1, y1, x2, y2


def _compute_hsv_hist(img_bgr: np.ndarray, xyxy: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    h_img, w_img = img_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_box_to_image(xyxy, w_img, h_img)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 4], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def _precompute_hists(img_bgr: Optional[np.ndarray], anns: List[Annotation]) -> List[Optional[np.ndarray]]:
    if img_bgr is None:
        return [None for _ in anns]
    hists: List[Optional[np.ndarray]] = []
    for a in anns:
        h = _compute_hsv_hist(img_bgr, a.xyxy)
        hists.append(h)
    return hists


def build_cost_matrix(
    anns1: List[Annotation],
    anns2: List[Annotation],
    img1_shape: Tuple[int, int],
    img2_shape: Tuple[int, int],
    iou_weight: float,
    centroid_weight: float,
    use_appearance: bool,
    hist_weight: float,
    tau_centroid: float,
    area_ratio_max: float,
    forbid_cross_class: bool,
) -> np.ndarray:
    # Note: appearance term is stubbed for now; implemented in a later step
    M = 1e6
    n1 = len(anns1)
    n2 = len(anns2)
    C = np.full((n1, n2), M, dtype=np.float32)

    # For centroid normalization, use the average of image diagonals if they differ
    w1, h1 = img1_shape
    w2, h2 = img2_shape
    diag = float(np.hypot((w1 + w2) / 2.0, (h1 + h2) / 2.0))
    if diag <= 0:
        diag = 1.0

    # Precompute histograms if requested
    hists1: List[Optional[np.ndarray]] = [None] * len(anns1)
    hists2: List[Optional[np.ndarray]] = [None] * len(anns2)

    # We will pass images separately in main when calling this function; use placeholders here
    # Appearance term will be integrated via separate function that calls this with precomputed hists

    for i, a in enumerate(anns1):
        a_box = a.xyxy
        a_c = a.centroid
        a_area = a.area
        for j, b in enumerate(anns2):
            if not _class_compatible(a.class_id, b.class_id, forbid_cross_class):
                # Strictly forbid cross-class matches
                continue
            b_box = b.xyxy
            b_c = b.centroid
            b_area = b.area

            # Size gating
            if not _area_ratio_ok(a_area, b_area, area_ratio_max):
                continue

            # Distance gating
            d_norm = centroid_distance(a_c, b_c) / diag
            if d_norm > tau_centroid:
                continue

            iou = iou_xyxy(a_box, b_box)

            d_hist = 0.0
            if use_appearance:
                # If histograms were not provided, d_hist remains 0; actual computation occurs in wrapper
                d_hist = 0.0

            cost = (iou_weight * (1.0 - iou)) + (centroid_weight * d_norm)
            if use_appearance:
                cost += hist_weight * d_hist

            C[i, j] = float(cost)

    logging.debug("Built cost matrix of shape %s", C.shape)
    return C


def build_cost_matrix_with_appearance(
    anns1: List[Annotation],
    anns2: List[Annotation],
    img1_bgr: Optional[np.ndarray],
    img2_bgr: Optional[np.ndarray],
    img1_shape: Tuple[int, int],
    img2_shape: Tuple[int, int],
    iou_weight: float,
    centroid_weight: float,
    use_appearance: bool,
    hist_weight: float,
    tau_centroid: float,
    area_ratio_max: float,
    forbid_cross_class: bool,
) -> np.ndarray:
    M = 1e6
    n1 = len(anns1)
    n2 = len(anns2)
    C = np.full((n1, n2), M, dtype=np.float32)

    w1, h1 = img1_shape
    w2, h2 = img2_shape
    diag = float(np.hypot((w1 + w2) / 2.0, (h1 + h2) / 2.0))
    if diag <= 0:
        diag = 1.0

    hists1 = _precompute_hists(img1_bgr if use_appearance else None, anns1)
    hists2 = _precompute_hists(img2_bgr if use_appearance else None, anns2)

    for i, a in enumerate(anns1):
        a_box = a.xyxy
        a_c = a.centroid
        a_area = a.area
        ha = hists1[i]
        for j, b in enumerate(anns2):
            if not _class_compatible(a.class_id, b.class_id, forbid_cross_class):
                continue
            b_box = b.xyxy
            b_c = b.centroid
            b_area = b.area
            if not _area_ratio_ok(a_area, b_area, area_ratio_max):
                continue
            d_norm = centroid_distance(a_c, b_c) / diag
            if d_norm > tau_centroid:
                continue
            iou = iou_xyxy(a_box, b_box)

            d_hist = 0.0
            if use_appearance:
                hb = hists2[j]
                if ha is None or hb is None:
                    d_hist = 1.0
                else:
                    # Bhattacharyya distance in [0,1]
                    d_hist = float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))

            cost = (iou_weight * (1.0 - iou)) + (centroid_weight * d_norm)
            if use_appearance:
                cost += hist_weight * d_hist

            C[i, j] = float(cost)

    logging.debug("Built cost matrix (appearance=%s) shape %s", use_appearance, C.shape)
    return C


def run_hungarian_matching(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if C.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    row_ind, col_ind = linear_sum_assignment(C)
    return row_ind, col_ind


# -------------------- Visualization --------------------
CLASS_NAMES = {
    0: "unknown",
    1: "person",
    2: "car",
    3: "other_vehicle",
    4: "other_object",
    5: "bike",
}


def _generate_colors(n: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    rng = np.random.RandomState(seed)
    colors: List[Tuple[int, int, int]] = []
    for _ in range(n):
        c = rng.randint(0, 255, size=3).tolist()
        colors.append((int(c[0]), int(c[1]), int(c[2])))
    return colors


def _put_label(img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_boxes(img: np.ndarray, anns: List[Annotation], indices: List[int], colors: List[Tuple[int, int, int]], label_prefix: str) -> None:
    h, w = img.shape[:2]
    thickness = max(1, int(round(min(h, w) / 300)))
    for idx, ann_index in enumerate(indices):
        ann = anns[ann_index]
        color = colors[idx]
        x1, y1, x2, y2 = _clip_box_to_image(ann.xyxy, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f"{label_prefix}{idx}: {CLASS_NAMES.get(ann.class_id, str(ann.class_id))}"
        _put_label(img, label, (x1, max(0, y1 - 5)), color)


def _make_same_height_and_concat(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    if h1 != h2:
        scale = h1 / float(h2)
        new_w2 = max(1, int(round(w2 * scale)))
        right = cv2.resize(right, (new_w2, h1), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([left, right])


def visualize_pair(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    anns1: List[Annotation],
    anns2: List[Annotation],
    matches: List[Tuple[int, int, float]],
    unmatched_1: List[int],
    unmatched_2: List[int],
    out_path: str,
    draw_legend: bool = False,
    color_seed: int = 123,
) -> None:
    img1 = img1_bgr.copy()
    img2 = img2_bgr.copy()

    colors = _generate_colors(len(matches), seed=color_seed)
    # Draw matched
    idxs1 = [m[0] for m in matches]
    idxs2 = [m[1] for m in matches]
    _draw_boxes(img1, anns1, idxs1, colors, label_prefix="m")
    _draw_boxes(img2, anns2, idxs2, colors, label_prefix="m")

    # Draw unmatched in gray
    gray = (192, 192, 192)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    thick1 = max(1, int(round(min(h1, w1) / 300)))
    thick2 = max(1, int(round(min(h2, w2) / 300)))

    for i in unmatched_1:
        x1, y1, x2, y2 = _clip_box_to_image(anns1[i].xyxy, w1, h1)
        cv2.rectangle(img1, (x1, y1), (x2, y2), gray, thick1)
        _put_label(img1, f"u:{CLASS_NAMES.get(anns1[i].class_id, str(anns1[i].class_id))}", (x1, max(0, y1 - 5)), gray)
    for j in unmatched_2:
        x1, y1, x2, y2 = _clip_box_to_image(anns2[j].xyxy, w2, h2)
        cv2.rectangle(img2, (x1, y1), (x2, y2), gray, thick2)
        _put_label(img2, f"u:{CLASS_NAMES.get(anns2[j].class_id, str(anns2[j].class_id))}", (x1, max(0, y1 - 5)), gray)

    stitched = _make_same_height_and_concat(img1, img2)

    if draw_legend and len(matches) > 0:
        # simple legend: draw colored squares
        legend_h = 20 * min(10, len(matches)) + 10
        legend_w = 220
        legend = np.ones((legend_h, legend_w, 3), dtype=np.uint8) * 255
        for k, (i_idx, j_idx, cost) in enumerate(matches[:10]):
            cv2.rectangle(legend, (5, 5 + 20 * k), (25, 25 + 20 * k), colors[k], -1)
            text = f"m{k}: c={cost:.2f}"
            _put_label(legend, text, (35, 22 + 20 * k), (0, 0, 0))
        # place legend at top-left of stitched
        lh, lw = legend.shape[:2]
        sh, sw = stitched.shape[:2]
        if lh <= sh and lw <= sw:
            stitched[0:lh, 0:lw] = legend

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, stitched)
    logging.info("Saved visualization to %s", out_path)


def _pair_group_key(img_path: str) -> str:
    # Group by the immediate pair directory name under data/
    # e.g., cv_data_hw2/data/Pair_X/... -> group key = Pair_X
    d = os.path.dirname(img_path).replace("\\", "/")
    parts = d.split("/")
    return parts[-1] if parts else d


def select_diverse_pairs(pairs: List[Tuple[str, str, str, str]], k: int = 5) -> List[Tuple[str, str, str, str]]:
    if k <= 0 or len(pairs) <= k:
        return pairs[:k] if k > 0 else pairs
    # First pass: pick one from as many distinct groups as possible
    seen: set = set()
    selected: List[Tuple[str, str, str, str]] = []
    for (img1, ann1, img2, ann2) in pairs:
        g = _pair_group_key(img1)
        if g not in seen:
            selected.append((img1, ann1, img2, ann2))
            seen.add(g)
        if len(selected) == k:
            break
    # If still fewer than k, fill remaining by spacing over the list
    if len(selected) < k:
        step = max(1, len(pairs) // (k - len(selected) + 1))
        idx = 0
        while len(selected) < k and idx < len(pairs):
            cand = pairs[idx]
            if cand not in selected:
                selected.append(cand)
            idx += step
    return selected[:k]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Reading index from %s", args.index)
    pairs = read_index(args.index)
    # Optionally select diverse subset or cap total pairs
    if args.select_diverse:
        logging.info("Selecting up to 5 diverse pairs for processing")
        pairs = select_diverse_pairs(pairs, k=5)
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    sample_count = len(pairs)

    for i in range(sample_count):
        img1, ann1, img2, ann2 = pairs[i]
        logging.info(
            "[%d/%d] Pair: %s | %s",
            i + 1,
            sample_count,
            os.path.basename(img1),
            os.path.basename(img2),
        )
        try:
            anns1 = parse_annotation_file(ann1)
            anns2 = parse_annotation_file(ann2)
        except Exception as e:
            logging.exception("Failed parsing annotations for pair %d: %s", i + 1, e)
            continue
        # Summarize parsed annotations (counts only at this stage)
        logging.info(
            "Objects parsed -> frame1=%d, frame2=%d", len(anns1), len(anns2)
        )

        # Compute cost matrix
        try:
            img1_bgr, (w1, h1) = _load_image(img1)
            img2_bgr, (w2, h2) = _load_image(img2)
            if (w1, h1) != (w2, h2):
                logging.warning(
                    "Image shapes differ: (%d,%d) vs (%d,%d); using avg diagonal for normalization",
                    w1,
                    h1,
                    w2,
                    h2,
                )
            use_app = (args.use_appearance == "on")
            C = build_cost_matrix_with_appearance(
                anns1=anns1,
                anns2=anns2,
                img1_bgr=img1_bgr,
                img2_bgr=img2_bgr,
                img1_shape=(w1, h1),
                img2_shape=(w2, h2),
                iou_weight=args.iou_weight,
                centroid_weight=args.centroid_weight,
                use_appearance=use_app,
                hist_weight=args.hist_weight,
                tau_centroid=args.tau_centroid,
                area_ratio_max=args.area_ratio_max,
                forbid_cross_class=args.forbid_cross_class,
            )
            logging.info("Cost matrix shape: %s (min=%.4f max=%.4f)", C.shape, float(np.min(C)) if C.size else float('nan'), float(np.max(C)) if C.size else float('nan'))
        except Exception as e:
            logging.exception("Failed building cost matrix for pair %d: %s", i + 1, e)
            continue

        # Hungarian matching and unmatched thresholding
        try:
            rows, cols = run_hungarian_matching(C)
            M = 1e6
            matches: List[Tuple[int, int, float]] = []
            unmatched_1 = set(range(len(anns1)))
            unmatched_2 = set(range(len(anns2)))
            for r, c in zip(rows, cols):
                cost = float(C[r, c])
                if cost >= args.tau_match or cost >= M:
                    # treat as unmatched
                    continue
                matches.append((int(r), int(c), cost))
                if r in unmatched_1:
                    unmatched_1.remove(r)
                if c in unmatched_2:
                    unmatched_2.remove(c)

            logging.info(
                "Matches: %d | Unmatched frame1: %d | Unmatched frame2: %d",
                len(matches),
                len(unmatched_1),
                len(unmatched_2),
            )

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                for (r, c, cost) in matches[:10]:
                    a = anns1[r]
                    b = anns2[c]
                    logging.debug(
                        " match r=%d (cls=%d, box=%s) <-> c=%d (cls=%d, box=%s) | cost=%.4f",
                        r,
                        a.class_id,
                        a.xyxy,
                        c,
                        b.class_id,
                        b.xyxy,
                        cost,
                    )

            # Visualization output per pair
            try:
                pair_basename = os.path.basename(os.path.dirname(ann1))
                out_path = os.path.join(args.output_dir, f"{pair_basename}_stitched.png")
                visualize_pair(
                    img1_bgr=img1_bgr,
                    img2_bgr=img2_bgr,
                    anns1=anns1,
                    anns2=anns2,
                    matches=matches,
                    unmatched_1=sorted(list(unmatched_1)),
                    unmatched_2=sorted(list(unmatched_2)),
                    out_path=out_path,
                    draw_legend=args.draw_legend,
                    color_seed=args.color_seed,
                )
            except Exception as e:
                logging.exception("Failed visualization for pair %d: %s", i + 1, e)
        except Exception as e:
            logging.exception("Failed matching for pair %d: %s", i + 1, e)
            continue


if __name__ == "__main__":
    main()


