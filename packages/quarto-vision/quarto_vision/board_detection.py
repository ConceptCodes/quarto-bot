"""
Board detection using ArUco tags + templated grid projection, with optional fast
ROI circle refinement for the 16 Quarto placement holes.

Usage (image):
    uv run python packages/quarto-vision/quarto_vision/board_detection.py image assets/board.png \
        --output assets/board_markers.png --json assets/board_markers.json --refine

At runtime:
    - Detect 4 corner ArUco tags (default IDs: TL=0, TR=1, BR=2, BL=3).
    - Compute homography to a unit square.
    - Project the canonical 4x4 grid centers back to image pixels.
    - Optionally refine each center with a small-window HoughCircle for robustness.
    - Save overlay image and/or JSON with the 16 centers (pixel coordinates).

Designed to be real-time friendly: ArUco + 16 tiny ROIs only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


DEFAULT_IDS = {"tl": 0, "tr": 1, "br": 2, "bl": 3}


def detect_aruco_corners(image: np.ndarray, aruco_dict_name: str = "DICT_4X4_50"):
    """Detect ArUco markers and return (corners, ids)."""
    if not hasattr(cv2.aruco, aruco_dict_name):
        raise ValueError(f"Unknown ArUco dictionary: {aruco_dict_name}")

    dictionary = cv2.aruco.__getattribute__(aruco_dict_name)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return {}, []
    detected = {int(i): np.squeeze(c) for i, c in zip(ids.flatten(), corners)}
    return detected, ids.flatten().tolist()


def compute_homography(detected: Dict[int, np.ndarray], expected_ids: Dict[str, int]):
    """Compute homography from image plane to canonical unit square."""
    required = [expected_ids[k] for k in ("tl", "tr", "br", "bl")]
    if not all(i in detected for i in required):
        missing = [i for i in required if i not in detected]
        raise RuntimeError(f"Missing required ArUco IDs: {missing}")

    # Order: tl, tr, br, bl (use single corner per tag for stability)
    src = np.array(
        [
            detected[expected_ids["tl"]][0],
            detected[expected_ids["tr"]][1],
            detected[expected_ids["br"]][2],
            detected[expected_ids["bl"]][3],
        ],
        dtype=np.float32,
    )
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
    return H


def generate_grid_points(rows: int = 4, cols: int = 4) -> np.ndarray:
    """Return Nx2 canonical grid centers (0..1)."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) / cols
            y = (r + 0.5) / rows
            pts.append([x, y])
    return np.array(pts, dtype=np.float32)


def project_points_to_image(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project canonical points through inverse homography into image pixels."""
    pts_h = cv2.convertPointsToHomogeneous(pts).reshape(-1, 3).T  # 3xN
    H_inv = np.linalg.inv(H)
    img_pts = H_inv @ pts_h
    img_pts /= img_pts[2]
    return img_pts[:2].T  # Nx2


def refine_centers_with_hough(
    gray: np.ndarray,
    centers: np.ndarray,
    roi: int = 72,
    min_r: int = 18,
    max_r: int = 70,
    param1: int = 100,
    param2: int = 15,
) -> np.ndarray:
    """Refine center positions using small ROI HoughCircles around each predicted center."""
    refined = []
    h, w = gray.shape
    for x, y in centers:
        x = int(round(x))
        y = int(round(y))
        x0 = max(0, x - roi // 2)
        y0 = max(0, y - roi // 2)
        x1 = min(w, x + roi // 2)
        y1 = min(h, y + roi // 2)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            refined.append([x, y])
            continue
        circles = cv2.HoughCircles(
            crop,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=roi / 4,
            param1=param1,
            param2=param2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            c = np.round(circles[0, 0]).astype(int)
            refined.append([x0 + c[0], y0 + c[1]])
        else:
            refined.append([x, y])
    return np.array(refined, dtype=np.float32)


def draw_overlay(
    image: np.ndarray,
    centers: np.ndarray,
    detected: Dict[int, np.ndarray],
    save_path: Path,
):
    out = image.copy()
    # Draw ArUco boxes
    for marker_id, corners in detected.items():
        corners_int = corners.astype(int)
        cv2.polylines(out, [corners_int], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(
            out,
            str(marker_id),
            tuple(corners_int[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    # Draw centers
    for idx, (x, y) in enumerate(centers.astype(int)):
        cv2.circle(out, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            out,
            str(idx + 1),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)
    print(f"Overlay saved to {save_path}")


def save_json(centers: np.ndarray, json_path: Path):
    payload = {"centers": centers.tolist()}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Centers saved to {json_path}")


def process_image(
    image_path: Path,
    aruco_dict: str,
    refine: bool,
    roi: int,
    min_r: int,
    max_r: int,
    output: Path | None,
    json_out: Path | None,
):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Image not found: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected, ids = detect_aruco_corners(image, aruco_dict)
    if len(detected) < 4:
        raise SystemExit(f"Found {len(detected)} marker(s); need 4 corner tags (default IDs 0,1,2,3).")

    H = compute_homography(detected, DEFAULT_IDS)
    grid = generate_grid_points()
    centers = project_points_to_image(grid, H)

    if refine:
        centers = refine_centers_with_hough(gray, centers, roi=roi, min_r=min_r, max_r=max_r)

    if output:
        draw_overlay(image, centers, detected, output)
    if json_out:
        save_json(centers, json_out)

    return centers


def get_board_centers(
    frame_bgr: np.ndarray,
    aruco_dict: str = "DICT_4X4_50",
    refine: bool = False,
    roi: int = 72,
    min_r: int = 18,
    max_r: int = 70,
) -> np.ndarray:
    """
    Lightweight helper for live use: takes a BGR frame, returns 16 (x, y) centers.
    Raises RuntimeError if required tags are missing.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detected, ids = detect_aruco_corners(frame_bgr, aruco_dict)
    if len(detected) < 4:
        raise RuntimeError(f"Need 4 corner tags; found {len(detected)}.")

    H = compute_homography(detected, DEFAULT_IDS)
    grid = generate_grid_points()
    centers = project_points_to_image(grid, H)
    if refine:
        centers = refine_centers_with_hough(gray, centers, roi=roi, min_r=min_r, max_r=max_r)
    return centers


def parse_args():
    parser = argparse.ArgumentParser(description="Quarto board detection via ArUco + template grid")
    parser.add_argument("mode", choices=["image"], help="Input mode (image file).")
    parser.add_argument("source", type=Path, help="Image path.")
    parser.add_argument("--aruco-dict", default="DICT_4X4_50", help="cv2.aruco dictionary name.")
    parser.add_argument("--output", type=Path, default=Path("assets/board_markers.png"), help="Overlay output image.")
    parser.add_argument("--json", type=Path, default=Path("assets/board_markers.json"), help="JSON output for centers.")
    parser.add_argument("--refine", action="store_true", help="Run ROI Hough refinement on each predicted circle.")
    parser.add_argument("--roi", type=int, default=72, help="ROI size (pixels) for refinement window.")
    parser.add_argument("--min-r", type=int, default=18, help="Min circle radius for refinement.")
    parser.add_argument("--max-r", type=int, default=70, help="Max circle radius for refinement.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "image":
        process_image(
            args.source,
            aruco_dict=args.aruco_dict,
            refine=args.refine,
            roi=args.roi,
            min_r=args.min_r,
            max_r=args.max_r,
            output=args.output,
            json_out=args.json,
        )


if __name__ == "__main__":
    main()
