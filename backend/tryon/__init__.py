"""Virtual try-on core package.

Modules
-------
pose       MediaPipe wrapper - landmarks in pixel and metric space
yaw        Body rotation angle from metric landmarks, plus smoothing and binning
garment    Garment asset loading, angle resolution, mirroring
warp       Correspondence building, similarity fit, thin-plate-spline refinement
compose    Alpha compositing and occlusion handling
capture    Guided 360 capture state machine
render360  Offline high-quality render pass over captured frames
"""

__all__ = [
    "pose",
    "yaw",
    "garment",
    "warp",
    "compose",
    "capture",
    "render360",
]
