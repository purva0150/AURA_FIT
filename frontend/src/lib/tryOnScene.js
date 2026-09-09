// Three.js 3D shirt renderer that follows MediaPipe world-landmarks.
// Uses an orthographic camera in NDC space so the shirt overlays the mirrored video pixel-perfectly.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

export class TryOnScene {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, premultipliedAlpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;

    this.scene = new THREE.Scene();

    // Ortho camera in NDC: viewport is [-1, 1] × [-1, 1]. Aspect handled by resize().
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
    this.camera.position.set(0, 0, 5);
    this.camera.lookAt(0, 0, 0);

    // Studio lighting for editorial vibe
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.1); key.position.set(0.8, 1.2, 2); this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xE2F13B, 0.4); rim.position.set(-1, 0.5, -1); this.scene.add(rim);
    const fill = new THREE.DirectionalLight(0xffffff, 0.35); fill.position.set(-1, -0.2, 1); this.scene.add(fill);

    this.pivot = new THREE.Group();
    this.scene.add(this.pivot);

    this.shirt = null;
    this.shirtMaterial = null;
    this.baseScale = 1;
    this.tint = new THREE.Color("#ffffff");

    // Smoothing (EMA) buffers to hide MediaPipe jitter
    this.smoothed = null;

    this.width = 1; this.height = 1;
  }

  resize(w, h) {
    this.width = w; this.height = h;
    const aspect = w / h;
    this.camera.left = -aspect; this.camera.right = aspect;
    this.camera.top = 1; this.camera.bottom = -1;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  async loadShirt(url) {
    if (this.shirt) { this.pivot.remove(this.shirt); this.shirt.traverse((n) => { n.geometry?.dispose?.(); n.material?.dispose?.(); }); }
    const gltf = await new Promise((resolve, reject) => new GLTFLoader().load(url, resolve, undefined, reject));
    const root = gltf.scene;
    // Find first mesh and center it
    let mesh = null;
    root.traverse((n) => { if (!mesh && n.isMesh) mesh = n; });
    if (!mesh) throw new Error("No mesh in GLB");

    // Reset scene material to a physically-based one we can tint
    const originalMap = mesh.material?.map || null;
    const pbr = new THREE.MeshStandardMaterial({
      color: this.tint,
      map: null, // we drop the printed logo texture so tint reads cleanly
      roughness: 0.85,
      metalness: 0.05,
      side: THREE.DoubleSide,
    });
    mesh.material = pbr;
    this.shirtMaterial = pbr;

    // Center + normalize scale so the shirt is roughly 1 unit tall
    const box = new THREE.Box3().setFromObject(root);
    const size = new THREE.Vector3(); box.getSize(size);
    const center = new THREE.Vector3(); box.getCenter(center);
    root.position.sub(center);
    // shirt looks a bit better when normalised so torso height (Y) = 1
    const heightUnit = Math.max(size.y, 1e-3);
    root.scale.setScalar(1 / heightUnit);

    // Rotate so front of shirt faces camera (+Z)
    root.rotation.y = Math.PI;

    this.shirt = root;
    this.pivot.add(this.shirt);
    // Slight upward offset so the mesh centre roughly sits at the model chest
    this.pivot.userData.baseOffsetY = 0.04;
    void originalMap;
  }

  setTint(hex) {
    this.tint.set(hex);
    if (this.shirtMaterial) this.shirtMaterial.color.copy(this.tint);
  }

  setVisible(v) {
    this.pivot.visible = !!v;
  }

  // landmarks: MediaPipe 2D normalized (0..1). worldLandmarks: 3D metres (approx).
  // videoW / videoH: intrinsic video dimensions (before CSS mirror).
  // We assume the video is already mirrored on the DOM (scaleX(-1)), so pose X → 1 - x.
  updateFromPose(landmarks, worldLandmarks, videoW, videoH, fitScale = 1.0) {
    if (!this.shirt || !landmarks) { this.setVisible(false); return; }
    const LS = landmarks[11], RS = landmarks[12], LH = landmarks[23], RH = landmarks[24];
    if (!LS || !RS || !LH || !RH) { this.setVisible(false); return; }
    // require decent visibility on shoulders + at least one hip
    const vOk = (p) => (p?.visibility ?? 1) > 0.4;
    if (!vOk(LS) || !vOk(RS) || (!vOk(LH) && !vOk(RH))) { this.setVisible(false); return; }
    this.setVisible(true);

    // Mirror x because video is CSS-mirrored
    const mx = (p) => 1 - p.x;
    // Convert normalized coords to NDC space matching our ortho camera
    // NDC: x in [-aspect, aspect], y in [-1, 1]
    const aspect = this.width / this.height;
    const toNdc = (nx, ny) => [(nx - 0.5) * 2 * aspect, -(ny - 0.5) * 2];

    const ls2 = toNdc(mx(LS), LS.y);
    const rs2 = toNdc(mx(RS), RS.y);
    const lh2 = toNdc(mx(LH), LH.y);
    const rh2 = toNdc(mx(RH), RH.y);

    // Chest anchor = midpoint of shoulders slightly lowered toward torso centre
    const shMid = [(ls2[0] + rs2[0]) / 2, (ls2[1] + rs2[1]) / 2];
    const hipMid = [(lh2[0] + rh2[0]) / 2, (lh2[1] + rh2[1]) / 2];
    const torsoCenter = [shMid[0] * 0.55 + hipMid[0] * 0.45, shMid[1] * 0.55 + hipMid[1] * 0.45];

    // Torso height (NDC) and shoulder width (NDC)
    const torsoH = Math.hypot(shMid[0] - hipMid[0], shMid[1] - hipMid[1]);
    const shoulderW = Math.hypot(ls2[0] - rs2[0], ls2[1] - rs2[1]);
    // Combined scale: shirt is ~1 unit tall in local space, we want it ≈ torso height in world
    // add small stretch so hem hangs below hips
    const scaleY = torsoH * 1.55 * fitScale;
    const scaleX = Math.max(shoulderW * 1.15, torsoH * 0.9) * fitScale;

    // In-plane roll: angle of shoulder line
    const rollZ = Math.atan2(ls2[1] - rs2[1], ls2[0] - rs2[0]);

    // Depth-based yaw / pitch from world landmarks (metres, hip-origin)
    let yaw = 0, pitch = 0;
    if (worldLandmarks?.length) {
      const wLS = worldLandmarks[11], wRS = worldLandmarks[12];
      const wLH = worldLandmarks[23], wRH = worldLandmarks[24];
      if (wLS && wRS) {
        // yaw: rotate around Y based on shoulder Z difference
        const dz = wLS.z - wRS.z;
        const dx = wLS.x - wRS.x;
        yaw = Math.atan2(dz, dx || 1e-4);
        // Because we mirrored the video, invert yaw
        yaw = -yaw;
      }
      if (wLH && wRH && wLS && wRS) {
        const shZ = (wLS.z + wRS.z) / 2;
        const hipZ = (wLH.z + wRH.z) / 2;
        const shY = (wLS.y + wRS.y) / 2;
        const hipY = (wLH.y + wRH.y) / 2;
        pitch = Math.atan2(shZ - hipZ, (shY - hipY) || 1e-4) * 0.6;
      }
    }

    // Exponential smoothing to hide MP jitter
    const alpha = 0.35;
    const tgt = { px: torsoCenter[0], py: torsoCenter[1], sx: scaleX, sy: scaleY, rz: rollZ, ry: yaw, rx: pitch };
    if (!this.smoothed) this.smoothed = { ...tgt };
    const s = this.smoothed;
    s.px = s.px + (tgt.px - s.px) * alpha;
    s.py = s.py + (tgt.py - s.py) * alpha;
    s.sx = s.sx + (tgt.sx - s.sx) * alpha;
    s.sy = s.sy + (tgt.sy - s.sy) * alpha;
    // Handle angle wrap for rotations
    const lerpAngle = (a, b, t) => a + (((b - a + Math.PI) % (Math.PI * 2)) - Math.PI) * t;
    s.rz = lerpAngle(s.rz, tgt.rz, alpha);
    s.ry = lerpAngle(s.ry, tgt.ry, alpha);
    s.rx = lerpAngle(s.rx, tgt.rx, alpha);

    // Apply to pivot
    this.pivot.position.set(s.px, s.py, 0);
    this.pivot.scale.set(s.sx, s.sy, (s.sx + s.sy) * 0.5);
    // Note: rz maps shoulder line to X-axis. Our shirt local up is +Y, so we rotate:
    // First set inner rotation on shirt for yaw/pitch, outer pivot handles roll.
    if (this.shirt) {
      this.shirt.rotation.set(s.rx, Math.PI + s.ry, 0); // Math.PI keeps front toward camera
    }
    this.pivot.rotation.set(0, 0, s.rz);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}
