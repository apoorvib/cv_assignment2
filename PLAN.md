## Assignment 2 Plan — Bipartite Object Matching on VIRAT Frame Pairs

### Objectives
- Parse frame pairs and annotations from `cv_data_hw2/index.txt` and `data/`.
- Build a cost matrix between objects across two frames using geometry (+ optional appearance).
- Run bipartite matching (Hungarian) to find minimum-cost assignments.
- Visualize matched objects with identical colors across the two frames; denote unmatched.
- Produce 5 diverse stitched examples for the report.
- Deliver a single, well-structured Python script and a 1–2 page report.

### Dataset and Annotation Schema
- Input root: `cv_data_hw2/` (assumed extracted).
- Index: `cv_data_hw2/index.txt` with rows formatted as: `[img1],[ann1],[img2],[ann2]`.
- Pair directories under `cv_data_hw2/data/` contain two `.png` frames and two `.annotation.txt` files.
- Annotation line format (8 ints): `[object_id, flag, frame_idx, x, y, w, h, class_id]`.
  - Coordinates are pixel-based; `(x, y)` is top-left, `(w, h)` width and height.
  - `class_id`: 0 Unknown, 1 person, 2 car, 3 other vehicle, 4 other object, 5 bike.

### Matching Policy (Critical for Report)
- Strict class consistency: cross-class matches are forbidden (including Unknown=0). We enforce this by setting cost to a large constant `M` for class-mismatched pairs. We will explicitly mention this constraint in the code and report.

### Cost Matrix Design
- Signals:
  - IoU of axis-aligned bounding boxes.
  - Normalized centroid distance: Euclidean distance between bbox centers divided by image diagonal.
  - Optional appearance similarity: HSV histogram distance inside the bbox (16×16×4 bins; Bhattacharyya or L1 distance), enabled by default and toggleable via CLI.
- Combined cost (when appearance enabled):
  - `cost(i,j) = w1*(1 − IoU) + w2*D_centroid_norm + w4*D_hist`
- Defaults:
  - `w1 = 0.6`, `w2 = 0.4`, `w4 = 0.2` (only if appearance used).
  - Gating to prune bad pairs by setting `cost = M` if any of the following:
    - `D_centroid_norm > 0.7` (too far apart)
    - Area ratio between boxes > 5× or < 1/5× (too dissimilar in size)
    - Classes differ (strict policy)
- Constants:
  - `M = 1e6` (large cost to effectively forbid matches).

### Assignment Algorithm
- Use `scipy.optimize.linear_sum_assignment` on the rectangular cost matrix.
- Support unequal set sizes; Hungarian returns assignments up to `min(|S1|, |S2|)`.
- After assignment, apply a match quality threshold:
  - `τ_match = 0.8` (default). Any assigned pair with `cost > τ_match` is marked unmatched.

### Unmatched Handling
- Allow unmatched on either side via `τ_match` post-check.
- Unmatched boxes are drawn in gray to clearly communicate outcomes in figures.

### Visualization
- Output per pair: stitched side-by-side image (left: frame 1, right: frame 2).
- Matched pairs share an identical color across frames; unmatched in gray.
- Optional legend overlay: pair index, class label, bbox or ID.
- Deterministic color assignment (fixed seed) for reproducibility.
- Save to `outputs/{pair_dir_basename}_stitched.png`.

### CLI / I/O
- Script accepts:
  - `--index` (default: `cv_data_hw2/index.txt`)
  - `--output_dir` (default: `outputs/`)
  - `--use_appearance {on,off}` (default: `on`)
  - `--tau_match` (default: `0.8`)
  - `--tau_centroid` (default: `0.7`)
  - `--iou_weight`, `--centroid_weight`, `--hist_weight`
  - `--max_pairs` to limit processed pairs for quick runs
- Assumes dataset already extracted; paths read directly from index.
- Logging: per-pair stats (#objs, matches, unmatched, avg cost).

### Environment / Libraries
- Python 3.x
- `numpy`, `opencv-python`, `scipy` (Hungarian), `matplotlib` (optional for legend/text rendering; OpenCV drawing by default).

### Sample Generation (5 Diverse Examples)
- Strategy to ensure diversity:
  - Sample pairs across different video sources (by directory prefix) and across varying object counts.
  - If needed, measure per-pair object counts and scene identifiers to pick 5 with variation.
- Save their stitched outputs and reference them in the report.

### Report Outline (1–2 pages)
1. Problem and dataset overview (pairs, annotation schema).
2. Method: cost components, strict class constraint, gating, Hungarian assignment, unmatched policy.
3. Implementation details: I/O, thresholds, and reproducibility.
4. Results: 5 stitched examples (captions: context and observations).
5. Assumptions and limitations; brief discussion of failure cases and improvements.

### Risks and Mitigations
- Misparsed annotations → schema validation; early assertions on 8-field lines and frame indices.
- Poor matches with geometry only → enable appearance hist by default; toggle off if slow.
- Class noise or mislabeled Unknown → strict class gating maintains precision at the expense of recall.
- Dimension variations → use actual image sizes for normalization.

### Definition of Done (Grading-Oriented Checklist)
- Parsing works for all pairs listed in `index.txt` without manual tweaks.
- Cost matrix computed with class forbidding and gating rules.
- Hungarian assignment applied; unmatched handled via `τ_match`.
- Stitched outputs saved with consistent color mapping; unmatched in gray.
- At least 5 diverse example outputs saved and suitable for the report.
- Single Python file runs end-to-end with documented CLI; deterministic results.
- Zip submission named `{netid}_{your_name}_CV_2.zip` including script, outputs, and report.

### Notes for Report Writing (to not forget)
- Explicitly state: strict class consistency (Unknown only matches Unknown) motivated by assignment guidance and to improve precision.
- Mention O(n^3) complexity of Hungarian but feasible due to small n per frame.
- Briefly justify weights and thresholds; note that appearance improves robustness across backgrounds.


