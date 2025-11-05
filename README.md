# Assignment 2: Bipartite Object Matching on VIRAT Frame Pairs

This script performs bipartite object matching between pairs of video frames using Hungarian algorithm optimization. It matches objects across two consecutive frames based on geometric features (IoU, centroid distance) and optional appearance features (HSV histogram).

## Prerequisites

### Required Python Packages

Install the following dependencies:

```bash
pip install numpy scipy opencv-python
```

Or install all at once:

```bash
pip install numpy scipy opencv-python
```

### Data Structure

The script expects the following directory structure:

```
cv_assignment2/
├── assignment2.py
├── cv_data_hw2/
│   ├── index.txt          # List of image/annotation pairs
│   └── data/
│       └── Pair_*/        # Pair directories containing PNG images and annotation files
└── README.md
```

## Basic Usage

### Run on All Data (Default)

To process all pairs listed in `cv_data_hw2/index.txt`:

```bash
cd cv_assignment2
python assignment2.py
```

This will:

- Process all pairs from the index file
- Save visualizations to `outputs/` directory
- Use default matching parameters

### Run from Parent Directory

If running from the parent directory:

```bash
python cv_assignment2/assignment2.py
```

## Command-Line Arguments

### Basic Options

| Argument       | Default                 | Description                                       |
| -------------- | ----------------------- | ------------------------------------------------- |
| `--index`      | `cv_data_hw2/index.txt` | Path to index file listing image/annotation pairs |
| `--output_dir` | `outputs`               | Directory to save visualization outputs           |
| `--verbose`    | `False`                 | Enable debug logging                              |
| `--max_pairs`  | `0` (all pairs)         | Limit number of pairs to process (0 = all)        |

### Matching Parameters

| Argument               | Default | Description                                             |
| ---------------------- | ------- | ------------------------------------------------------- |
| `--iou_weight`         | `0.6`   | Weight for (1 - IoU) in cost calculation                |
| `--centroid_weight`    | `0.4`   | Weight for normalized centroid distance                 |
| `--hist_weight`        | `0.2`   | Weight for appearance histogram distance                |
| `--use_appearance`     | `on`    | Enable/disable appearance term (`on` or `off`)          |
| `--tau_centroid`       | `0.7`   | Max normalized centroid distance threshold (gating)     |
| `--area_ratio_max`     | `5.0`   | Max area ratio between boxes (gating)                   |
| `--tau_match`          | `0.8`   | Post-assignment cost threshold (above this = unmatched) |
| `--forbid_cross_class` | `True`  | Strictly forbid cross-class matches                     |
| `--color_seed`         | `123`   | Random seed for deterministic color assignment          |

### Visualization Options

| Argument           | Default | Description                                      |
| ------------------ | ------- | ------------------------------------------------ |
| `--draw_legend`    | `False` | Draw legend with match indices and classes       |
| `--select_diverse` | `False` | Select 5 diverse pairs across sources for report |

## Examples

### Example 1: Process All Data with Default Settings

```bash
python assignment2.py
```

### Example 2: Process Only First 10 Pairs (for Testing)

```bash
python assignment2.py --max_pairs 10
```

### Example 3: Run with Verbose Logging

```bash
python assignment2.py --verbose
```

### Example 4: Custom Output Directory

```bash
python assignment2.py --output_dir my_results
```

### Example 5: Disable Appearance Matching (Faster)

```bash
python assignment2.py --use_appearance off
```

### Example 6: Select 5 Diverse Examples for Report

```bash
python assignment2.py --select_diverse --draw_legend
```

### Example 7: Custom Matching Weights

```bash
python assignment2.py --iou_weight 0.7 --centroid_weight 0.3 --hist_weight 0.1
```

### Example 8: Adjust Matching Thresholds

```bash
python assignment2.py --tau_centroid 0.5 --tau_match 0.6
```

## Output Format

### Visualizations

The script generates stitched side-by-side images for each processed pair:

- **Location**: `{output_dir}/{pair_directory_basename}_stitched.png`
- **Format**: Side-by-side visualization showing:
  - Left: Frame 1 with bounding boxes
  - Right: Frame 2 with bounding boxes
  - Matched objects: Same color across both frames
  - Unmatched objects: Gray boxes

### Logging

The script logs:

- Number of pairs processed
- Objects parsed per frame
- Number of matches found
- Number of unmatched objects per frame
- Cost matrix statistics

Use `--verbose` for detailed debug information.

## Object Classes

The script recognizes the following object classes:

- `0`: Unknown
- `1`: Person
- `2`: Car
- `3`: Other Vehicle
- `4`: Other Object
- `5`: Bike

**Note**: By default, cross-class matches are forbidden (strict class matching policy).

## Matching Algorithm

The cost matrix combines three terms:

1. **IoU (Intersection over Union)**: `w1 * (1 - IoU)`
2. **Centroid Distance**: `w2 * normalized_centroid_distance`
3. **Appearance (optional)**: `w3 * histogram_distance` (Bhattacharyya distance)

**Cost Formula** (when appearance enabled):

```
cost(i,j) = iou_weight * (1 - IoU) + centroid_weight * D_centroid_norm + hist_weight * D_hist
```

**Gating Rules** (pairs with these conditions get cost = M):

- Centroid distance > `tau_centroid`
- Area ratio > `area_ratio_max` or < `1/area_ratio_max`
- Different object classes (if `--forbid_cross_class` enabled)

**Post-assignment filtering**:

- Matches with cost > `tau_match` are marked as unmatched

## Troubleshooting

### Import Error: No module named 'cv2'

Install OpenCV:

```bash
pip install opencv-python
```

### File Not Found Error

Ensure you're running from the correct directory:

```bash
cd cv_assignment2
python assignment2.py
```

Or specify the index path explicitly:

```bash
python assignment2.py --index cv_assignment2/cv_data_hw2/index.txt
```

### Memory Issues with Large Datasets

Process fewer pairs at a time:

```bash
python assignment2.py --max_pairs 50
```

### Slow Processing

Disable appearance matching for faster processing:

```bash
python assignment2.py --use_appearance off
```