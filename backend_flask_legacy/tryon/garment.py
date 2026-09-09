"""Garment assets: loading, angle resolution, mirroring.

A garment is a directory of per-angle RGBA images plus anchor JSONs::

    garments/synthetic_tee/
        garment.json              manifest (name, colour, angle index)
        synthetic_tee_000.png     RGBA
        synthetic_tee_000.json    anchors for that angle
        synthetic_tee_045.png
        ...

Anchors are keyed by MediaPipe landmark name and hold the pixel position that
*body joint* occupies in the garment image. That is the whole trick: because
both sides of the correspondence speak the same vocabulary, warping is a dict
lookup rather than a hand-maintained mapping table.

Sources differ in how many angles they can supply. Own photography gives all
eight; DeepFashion In-shop gives three (front / side / back). So nothing here
assumes a fixed angle set - ``GarmentSet`` resolves against whatever exists and
blends between the two nearest available angles.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .yaw import angular_dist, normalize_deg

#: Anchor keys whose left/right sense must swap when an asset is mirrored.
_MIRROR_SWAPS = (
    ("left_shoulder", "right_shoulder"),
    ("left_chest", "right_chest"),
    ("left_waist", "right_waist"),
    ("left_elbow", "right_elbow"),
    ("left_hip", "right_hip"),
    ("left_wrist", "right_wrist"),
)


@dataclass
class GarmentAngle:
    """One garment image at one yaw angle."""

    angle_deg: float
    view: str
    image_path: Optional[str]
    anchors: Dict[str, np.ndarray]
    fit: Dict[str, float] = field(default_factory=dict)
    mirrored_from: Optional[float] = None
    _image: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def image(self) -> np.ndarray:
        """BGRA image, loaded on first use and cached."""
        if self._image is None:
            if self.image_path is None:
                raise ValueError(f"angle {self.angle_deg} has no image and none cached")
            img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"could not read garment image: {self.image_path}")
            self._image = ensure_bgra(img)
        return self._image

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.image.shape[:2]
        return (w, h)

    def anchor(self, name: str) -> Optional[np.ndarray]:
        return self.anchors.get(name)

    def mirrored(self, angle_deg: Optional[float] = None) -> "GarmentAngle":
        """Horizontally flipped copy, with left/right anchor keys swapped.

        Halves photography: shoot 0-180 and mirror for 225/270/315. Note the key
        swap is not cosmetic - without it the warp would tie the person's left
        shoulder to what is now the garment's right, and the fit would shear.
        """
        img = cv2.flip(self.image, 1)
        width = img.shape[1]

        flipped: Dict[str, np.ndarray] = {}
        for name, pt in self.anchors.items():
            flipped[name] = np.array([width - 1 - pt[0], pt[1]], dtype=np.float64)
        for a, b in _MIRROR_SWAPS:
            if a in flipped and b in flipped:
                flipped[a], flipped[b] = flipped[b], flipped[a]

        target = normalize_deg(360.0 - self.angle_deg) if angle_deg is None else normalize_deg(angle_deg)
        return GarmentAngle(
            angle_deg=target,
            view=f"{self.view}_mirrored",
            image_path=None,
            anchors=flipped,
            fit=dict(self.fit),
            mirrored_from=self.angle_deg,
            _image=img,
        )


@dataclass
class GarmentSet:
    """All available angles for one garment."""

    garment_id: str
    name: str
    root: str
    angles: Dict[int, GarmentAngle] = field(default_factory=dict)
    dominant_color: str = "#888888"
    category: str = "upper_body"

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(cls, root: str, auto_mirror: bool = True) -> "GarmentSet":
        """Load a garment directory.

        Prefers ``garment.json`` but falls back to globbing ``*_NNN.json`` so a
        half-built asset folder still works during dataset preparation.
        """
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"garment directory not found: {root}")

        manifest_path = os.path.join(root, "garment.json")
        garment_id = os.path.basename(root)
        name = garment_id
        dominant = "#888888"
        category = "upper_body"
        entries: List[dict] = []

        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            garment_id = manifest.get("garment_id", garment_id)
            name = manifest.get("name", garment_id)
            dominant = manifest.get("dominant_color", dominant)
            category = manifest.get("category", category)
            entries = manifest.get("angles", [])

        angles: Dict[int, GarmentAngle] = {}
        if entries:
            for entry in entries:
                anchors_file = os.path.join(root, entry["anchors"])
                angle = _load_angle_json(anchors_file, root)
                if angle is not None:
                    angles[int(round(angle.angle_deg)) % 360] = angle
        else:
            for fname in sorted(os.listdir(root)):
                if not fname.endswith(".json") or fname == "garment.json":
                    continue
                angle = _load_angle_json(os.path.join(root, fname), root)
                if angle is not None:
                    angles[int(round(angle.angle_deg)) % 360] = angle

        if not angles:
            raise ValueError(f"no usable angles found in {root}")

        garment = cls(
            garment_id=garment_id,
            name=name,
            root=root,
            angles=angles,
            dominant_color=dominant,
            category=category,
        )
        if auto_mirror:
            garment.fill_mirrors()
        return garment

    def fill_mirrors(self) -> List[int]:
        """Synthesise missing right-turn angles by mirroring their left-turn twins.

        A source that only supplies 0/45/90/135/180 becomes a full 360 set. Only
        fills gaps - a real asset at 270 is never replaced by a mirror of 90.
        """
        added: List[int] = []
        for deg in sorted(self.angles.keys()):
            mirror_deg = int(normalize_deg(360.0 - deg)) % 360
            if mirror_deg == deg or mirror_deg in self.angles:
                continue
            self.angles[mirror_deg] = self.angles[deg].mirrored(mirror_deg)
            added.append(mirror_deg)
        return added

    # -- angle resolution ------------------------------------------------

    @property
    def available_angles(self) -> List[int]:
        return sorted(self.angles.keys())

    def nearest(self, yaw: float) -> GarmentAngle:
        best = min(self.available_angles, key=lambda d: angular_dist(yaw, d))
        return self.angles[best]

    def resolve(self, yaw: float) -> Tuple[GarmentAngle, GarmentAngle, float]:
        """Bracket ``yaw`` with the two nearest available angles.

        Returns ``(lo, hi, weight)`` where ``weight`` is how much the *hi* asset
        should contribute. Cross-dissolving on this is what removes the jump cut
        at bin boundaries - with only three real angles from a catalogue source,
        the dissolve is doing a lot of work.
        """
        yaw = normalize_deg(yaw)
        avail = self.available_angles
        if len(avail) == 1:
            only = self.angles[avail[0]]
            return only, only, 0.0

        lo_deg = avail[-1]
        hi_deg = avail[0]
        for i, deg in enumerate(avail):
            if deg <= yaw:
                lo_deg = deg
                hi_deg = avail[(i + 1) % len(avail)]

        span = (hi_deg - lo_deg) % 360.0
        # Do not dissolve halfway around the body when only front and back
        # photographs exist. That blend creates a translucent/twisted shirt;
        # use the closest real view until a side asset is supplied.
        if span > 90.0:
            nearest = self.nearest(yaw)
            return nearest, nearest, 0.0
        weight = 0.0 if span < 1e-9 else ((yaw - lo_deg) % 360.0) / span
        return self.angles[lo_deg], self.angles[hi_deg], float(np.clip(weight, 0.0, 1.0))

    def coverage_gaps(self, bins: Sequence[float], tolerance: float = 25.0) -> List[float]:
        """Bins with no asset within ``tolerance`` degrees. Drives the warning the
        live app shows when a garment can't actually support a full turn."""
        return [b for b in bins if all(angular_dist(b, d) > tolerance for d in self.available_angles)]


def _load_angle_json(path: str, root: str) -> Optional[GarmentAngle]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    if "anchors" not in payload or "angle_deg" not in payload:
        return None

    anchors = {
        name: np.array(xy, dtype=np.float64)
        for name, xy in payload["anchors"].items()
        if isinstance(xy, (list, tuple)) and len(xy) == 2
    }
    if not anchors:
        return None

    image_name = payload.get("image")
    image_path = os.path.join(root, image_name) if image_name else None
    return GarmentAngle(
        angle_deg=float(payload["angle_deg"]),
        view=payload.get("view", ""),
        image_path=image_path,
        anchors=anchors,
        fit={
            str(key): float(value)
            for key, value in payload.get("fit", {}).items()
            if isinstance(value, (int, float))
        },
    )


def ensure_bgra(img: np.ndarray) -> np.ndarray:
    """Coerce any loaded image to 4-channel BGRA."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def discover_garments(garments_root: str) -> List[GarmentSet]:
    """Load every garment directory under ``garments_root``, skipping bad ones.

    Used by the Flask catalogue so the mobile picker reflects what is actually
    on disk instead of the hardcoded placeholder list.
    """
    found: List[GarmentSet] = []
    if not os.path.isdir(garments_root):
        return found
    for entry in sorted(os.listdir(garments_root)):
        path = os.path.join(garments_root, entry)
        if not os.path.isdir(path):
            continue
        try:
            found.append(GarmentSet.load(path))
        except (ValueError, FileNotFoundError):
            continue
    return found
