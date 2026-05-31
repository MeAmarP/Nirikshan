"""Analytics utilities for tracking and visual insights."""

from __future__ import annotations

from typing import Iterable, List, Dict, Tuple, Optional, Literal
import cv2
import numpy as np

from dataclasses import dataclass, asdict
import math
import json

from configs import AppConfig

class CountAnalytics:
    """Track simple per-class counts for detected and tracked objects."""
    def __init__(self):
        self.name = 'CountAnalytics'
        self.target_class_labels = AppConfig.detector_class_labels
        self.classwise_count_curr = {}  # stores objects to be tracked with their counts
        self.init_counts()  # initialize counts for each class label

    def __str__(self) -> str:
        return " ".join(f"{k} {v}" for k,v in self.classwise_count_curr.items())

    def init_counts(self):
        """Initialise zero counts for every configured class label."""
        for label in self.target_class_labels:
            if label not in self.classwise_count_curr:
                self.classwise_count_curr[label] = 0

    def update(self, tracked_objects, label):
        """Update the current count for a specific label.

        Args:
            tracked_objects: Iterable of tracked detections that belong to the label.
            label: Class label whose count should be updated.
        """
        # If the object is a target to be tracked...
        if label in self.target_class_labels:
            # len(None) would raise, so guard against tracker returning None
            countable = tracked_objects if tracked_objects is not None else []
            self.classwise_count_curr[label] = len(countable)

    def get(self):
        return dict(self.classwise_count_curr)

@dataclass
class Zone:
    center_x: float
    center_y: float
    count: int
    density: float
    radius: float

class CrowdDensityAnalytics:
    """
    Crowd-density and zone clustering over tracked person detections.

    Features:
    - Time-based exponential decay (FPS independent).
    - Optional DBSCAN clustering (better for unknown K).
    - Bilinear heatmap splatting for smoother, resolution-agnostic density maps.
    - Optional ground-plane homography for perspective-correct analytics.
    """

    def __init__(
        self,
        max_zones: Optional[int] = None,
        heatmap_resolution: Tuple[int, int] = (96, 54),
        decay_lambda: float = 0.,  # per-second rate; higher = faster decay
        min_detections: int = 0.25,
        cluster_algo: Literal["kmeans", "dbscan"] = "dbscan",
        dbscan_eps: float = 28.0,
        dbscan_min_samples: int = 3,
        kmeans_attempts: int = 3,
        random_state: Optional[int] = 1234,
        gaussian_sigma: float = 1.0,
        stable_norm_percentile: float = 99.5,
        homography: Optional[np.ndarray] = None,  # 3x3 H to ground plane
    ) -> None:
        self.name = "CrowdDensityAnalytics"
        self.max_zones = max_zones or 5
        self.grid_w, self.grid_h = int(heatmap_resolution[0]), int(heatmap_resolution[1])
        self.decay_lambda = float(decay_lambda)
        self.min_detections = int(min_detections)
        self.cluster_algo = cluster_algo
        self.dbscan_eps = float(dbscan_eps)
        self.dbscan_min_samples = int(dbscan_min_samples)
        self.kmeans_attempts = int(kmeans_attempts)
        self.random_state = random_state
        self.gaussian_sigma = float(gaussian_sigma)
        self.stable_norm_percentile = float(stable_norm_percentile)
        self.homography = homography.copy() if homography is not None else None

        self.heatmap = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.zones: List[Zone] = []
        self._ema_max: float = 1.0  # for stable color scaling

        if self.random_state is not None:
            cv2.setRNGSeed(int(self.random_state))

    # ---------- Public API ----------

    def update(self, tracked_objects: Iterable, frame_shape: Tuple[int, int], dt_seconds: float = 1/30.0) -> None:
        if frame_shape is None or len(frame_shape) < 2:
            raise ValueError("frame_shape must provide height and width")
        H, W = int(frame_shape[0]), int(frame_shape[1])
        if H <= 0 or W <= 0:
            raise ValueError("frame_shape must contain positive height and width")

        self._decay_heatmap(dt_seconds)

        positions = self._extract_positions(tracked_objects)
        if positions.size == 0 or positions.shape[0] < self.min_detections:
            self.zones = []
            return

        # Optional: map to ground plane for perspective correctness
        if self.homography is not None:
            positions = self._warp_points(positions, self.homography)

        # Cluster
        zones = self._cluster_to_zones(positions, W, H)
        self.zones = zones

        # Accumulate into heatmap using bilinear splatting
        self._accumulate_heatmap_bilinear(positions, (H, W))

    def get(self) -> Dict[str, List[Dict[str, float]]]:
        return {"zones": [asdict(z) for z in self.zones]}

    def get_json(self) -> str:
        return json.dumps(self.get())

    def get_heatmap_overlay(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        H, W = frame_shape[:2]
        # Stable normalization: update EMA of high-percentile
        p = np.percentile(self.heatmap, self.stable_norm_percentile)
        if not np.isnan(p) and p > 0:
            # EMA with alpha tuned for stability
            alpha = 0.2
            self._ema_max = (1 - alpha) * self._ema_max + alpha * float(p)
        denom = max(self._ema_max, 1e-6)
        normalized = np.clip(self.heatmap / denom, 0.0, 1.0)

        resized = cv2.resize(normalized, (W, H), interpolation=cv2.INTER_CUBIC)
        colored = cv2.applyColorMap((resized * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
        return colored

    # ---------- Internals ----------

    def _decay_heatmap(self, dt_seconds: float) -> None:
        # Exponential decay: exp(-lambda * dt)
        factor = math.exp(-self.decay_lambda * max(dt_seconds, 0.0))
        self.heatmap *= float(np.clip(factor, 0.0, 1.0))

    def _extract_positions(self, tracked_objects: Iterable) -> np.ndarray:
        # bottom-center points
        positions = []
        for obj in tracked_objects or []:
            tlwh = getattr(obj, "tlwh", None)
            if tlwh is None:
                continue
            arr = np.asarray(tlwh, dtype=np.float32)
            if arr.size < 4 or not np.isfinite(arr[:4]).all():
                continue
            x, y, w, h = arr[:4]
            positions.append([x + 0.5 * w, y + h])
        if not positions:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(positions, dtype=np.float32)

    def _warp_points(self, pts: np.ndarray, H: np.ndarray) -> np.ndarray:
        # pts: (N,2) -> apply homography
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        warped = (H @ pts_h.T).T
        denom = warped[:, 2:3]
        denom[denom == 0] = 1e-6
        return warped[:, :2] / denom

    def _cluster_to_zones(self, positions: np.ndarray, frame_w: int, frame_h: int) -> List[Zone]:
        total = positions.shape[0]
        if total == 0:
            return []

        if self.cluster_algo == "dbscan":
            try:
                from sklearn.cluster import DBSCAN  # optional dependency
                labels = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit_predict(positions)
                unique = [l for l in np.unique(labels) if l != -1]
                zones: List[Zone] = []
                for l in unique:
                    mask = labels == l
                    pts = positions[mask]
                    center = pts.mean(axis=0)
                    diffs = pts - center
                    radius = float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1)))) if pts.shape[0] > 1 else 15.0
                    zones.append(Zone(
                        center_x=float(center[0]),
                        center_y=float(center[1]),
                        count=int(pts.shape[0]),
                        density=float(pts.shape[0] / total),
                        radius=max(radius, 10.0),
                    ))
                return zones
            except Exception:
                # Fallback to kmeans if sklearn is unavailable
                pass

        # KMeans path (cap K by both max_zones and #points)
        K = max(1, min(self.max_zones, total))
        if K == 1:
            center = positions.mean(axis=0)
            diffs = positions - center
            radius = float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1)))) if positions.shape[0] > 1 else 15.0
            return [Zone(float(center[0]), float(center[1]), int(total), 1.0, max(radius, 10.0))]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
        compactness, labels, centers = cv2.kmeans(
            data=positions.astype(np.float32),
            K=K,
            bestLabels=None,
            criteria=criteria,
            attempts=self.kmeans_attempts,
            flags=cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.flatten()
        zones: List[Zone] = []
        for idx in range(centers.shape[0]):
            mask = labels == idx
            if not np.any(mask):
                continue
            pts = positions[mask]
            center = centers[idx]
            diffs = pts - center
            radius = float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1)))) if pts.shape[0] > 1 else 15.0
            zones.append(Zone(
                center_x=float(center[0]),
                center_y=float(center[1]),
                count=int(pts.shape[0]),
                density=float(pts.shape[0] / total),
                radius=max(radius, 10.0),
            ))
        return zones

    def _accumulate_heatmap_bilinear(self, positions: np.ndarray, frame_shape: Tuple[int, int]) -> None:
        H, W = frame_shape
        if H <= 1 or W <= 1 or positions.size == 0:
            return

        gh, gw = self.heatmap.shape  # (grid_h, grid_w)
        # Scale factors map [0, W-1] -> [0, gw-1], [0, H-1] -> [0, gh-1]
        sx = (gw - 1) / (W - 1)
        sy = (gh - 1) / (H - 1)

        # 1) Clamp pixel coords to be strictly inside the frame to avoid landing exactly on the last grid cell’s +1
        eps = 1e-6
        px = np.clip(positions[:, 0], 0.0, W - 1.0 - eps)
        py = np.clip(positions[:, 1], 0.0, H - 1.0 - eps)

        # 2) Map to grid space
        gx = px * sx
        gy = py * sy

        # 3) Base indices
        x0 = np.floor(gx).astype(np.int32)
        y0 = np.floor(gy).astype(np.int32)

        # Safety: ensure inside bounds (defensive even after eps clipping)
        x0 = np.clip(x0, 0, gw - 1)
        y0 = np.clip(y0, 0, gh - 1)

        # 4) Neighbor indices
        x1 = np.minimum(x0 + 1, gw - 1)
        y1 = np.minimum(y0 + 1, gh - 1)

        # 5) Fractional parts (use float indices for subtraction)
        fx = gx - x0.astype(np.float32)
        fy = gy - y0.astype(np.float32)

        # 6) Bilinear weights
        w00 = (1.0 - fx) * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w10 = fx * (1.0 - fy)
        w11 = fx * fy

        # 7) Accumulate (all indices are valid now)
        np.add.at(self.heatmap, (y0, x0), w00)
        np.add.at(self.heatmap, (y1, x0), w01)
        np.add.at(self.heatmap, (y0, x1), w10)
        np.add.at(self.heatmap, (y1, x1), w11)

        # Optional smoothing
        if self.gaussian_sigma > 0:
            self.heatmap = cv2.GaussianBlur(self.heatmap, ksize=(0, 0), sigmaX=self.gaussian_sigma)



    def _project_to_image_xy(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project points from the clustering space back to image pixel space.
        If homography was used during clustering, use its inverse here.
        """
        if self.homography is None:
            return xs, ys
        # inverse warp: ground -> image
        H_inv = np.linalg.inv(self.homography)
        pts = np.stack([xs, ys, np.ones_like(xs, dtype=np.float32)], axis=0)  # (3, N)
        warped = H_inv @ pts  # (3, N)
        denom = np.clip(warped[2, :], 1e-6, None)
        xi = warped[0, :] / denom
        yi = warped[1, :] / denom
        return xi.astype(np.float32), yi.astype(np.float32)

    def render_overlay(
        self,
        frame: np.ndarray,
        alpha_fg: float = 0.6,
        alpha_overlay: float = 0.4,
        density_threshold: float = 0.5,
        color_low: Tuple[int, int, int] = (0, 165, 255),   # BGR: orange
        color_high: Tuple[int, int, int] = (0, 0, 255),    # BGR: red
        font_scale: float = 2.0,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Blend heatmap and draw clustered zones onto the frame. Returns a new frame.
        Colors default to OpenCV BGR tuples; pass your own if you use a Color enum.
        """
        if frame is None or frame.size == 0:
            return frame

        H, W = frame.shape[:2]
        overlay = self.get_heatmap_overlay((H, W))
        out = cv2.addWeighted(frame, alpha_fg, overlay, alpha_overlay, 0.0)

        # Fetch zones and project centers to image pixel coordinates (if needed)
        zs = self.get().get("zones", [])
        if not zs:
            return out

        centers = np.array([[z["center_x"], z["center_y"]] for z in zs], dtype=np.float32)
        xs_i, ys_i = self._project_to_image_xy(centers[:, 0], centers[:, 1])

        # Draw
        for i, z in enumerate(zs):
            cx = int(np.clip(xs_i[i], 0, W - 1))
            cy = int(np.clip(ys_i[i], 0, H - 1))
            radius = int(max(float(z.get("radius", 10.0)), 10.0))
            density = float(z.get("density", 0.0))
            count = int(z.get("count", 0))

            color = color_low if density < density_threshold else color_high
            cv2.circle(out, (cx, cy), radius, color, 2)
            cv2.putText(
                out,
                f"{count}",
                (cx, max(0, cy - radius - 5)),
                cv2.FONT_HERSHEY_PLAIN,
                font_scale,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        return out
