import * as THREE from "three";
import { loadGarment } from "./garmentLoader";
import { ClothSimulation } from "./clothSimulation";
import { fitFromPose, angleDelta } from "./poseFit";
import { normalizeGarmentGeometry } from "./garmentGeometry";

function disposeObject(root) {
  root?.traverse((n) => {
    n.geometry?.dispose();
    const materials = Array.isArray(n.material) ? n.material : [n.material];
    materials.filter(Boolean).forEach((m) => {
      Object.values(m).forEach((v) => { if (v?.isTexture) v.dispose(); });
      m.dispose();
    });
  });
}

export class TryOnScene {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, premultipliedAlpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 20);
    this.camera.position.set(0, 0, 5);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    [[0xffffff, 1.1, 0.8, 1.2, 2], [0xe2f13b, 0.3, -1, 0.5, -1], [0xffffff, 0.35, -1, -0.2, 1]].forEach(([color, power, x, y, z]) => {
      const light = new THREE.DirectionalLight(color, power); light.position.set(x, y, z); this.scene.add(light);
    });
    this.pivot = new THREE.Group(); this.pivot.visible = false; this.scene.add(this.pivot);
    this.tint = new THREE.Color("#ffffff");
    this.simulations = [];
    this.smoothed = null;
    this.width = 1; this.height = 1;
    this.tracking = false;
    this.disposed = false;
    this.loadVersion = 0;
  }

  resize(w, h) {
    if (!w || !h) return;
    this.width = w; this.height = h;
    this.camera.left = -w / h; this.camera.right = w / h;
    this.camera.updateProjectionMatrix(); this.renderer.setSize(w, h, false);
    this.smoothed = null;
    this.simulations.forEach((s) => s.reset());
  }

  async loadShirt(url) {
    this.cancelLoad();
    const version = ++this.loadVersion;
    this.loadController = new AbortController();
    const gltf = await loadGarment(url, this.loadController.signal);
    if (this.disposed || version !== this.loadVersion) { disposeObject(gltf.scene); return; }
    gltf.scene.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const center = box.getCenter(new THREE.Vector3());
    const height = Math.max(box.getSize(new THREE.Vector3()).y, 0.001);
    const root = new THREE.Group();
    gltf.scene.traverse((n) => {
      if (!n.isMesh) return;
      // Bake centering in vertex space BEFORE scaling; retain every GLB mesh.
      const geometry = n.geometry.clone().applyMatrix4(n.matrixWorld);
      normalizeGarmentGeometry(geometry, center, height);
      const material = new THREE.MeshStandardMaterial({ color: this.tint, roughness: 0.9, metalness: 0, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geometry, material); mesh.frustumCulled = false; root.add(mesh);
    });
    disposeObject(gltf.scene);
    if (!root.children.length) throw new Error("No clothing mesh in GLB");
    this.simulations.forEach((s) => s.dispose());
    disposeObject(this.shirt); if (this.shirt) this.pivot.remove(this.shirt);
    this.shirt = root; this.pivot.add(root);
    this.simulations = root.children.map((m) => new ClothSimulation(m.geometry));
    this.smoothed = null;
  }

  setTint(hex) { this.tint.set(hex); this.shirt?.children.forEach((m) => m.material.color.copy(this.tint)); }
  setVisible(visible) {
    const wasVisible = this.pivot.visible;
    this.pivot.visible = !!visible;
    if (!visible && (wasVisible || this.tracking)) { this.tracking = false; this.smoothed = null; this.simulations.forEach((s) => s.reset()); }
  }

  updateFromPose(landmarks, world, videoW, videoH, fit = 1, dt = 1 / 60, enabled = true, view = "auto") {
    const target = this.shirt && fitFromPose(landmarks, world, videoW, videoH, this.width, this.height, fit, view);
    if (!target) { this.setVisible(false); return; }
    this.tracking = true; this.pivot.visible = true;
    if (!this.smoothed) this.smoothed = { ...target };
    const s = this.smoothed, alpha = 1 - Math.exp(-Math.min(dt, 0.1) * 14);
    Object.keys(target).forEach((key) => {
      s[key] += (key.startsWith("r") && key !== "rightArm" ? angleDelta(s[key], target[key]) : target[key] - s[key]) * alpha;
    });
    this.pivot.position.set(s.px, s.py, 0);
    this.pivot.scale.set(s.sx, s.sy, s.sx * 0.72);
    this.pivot.rotation.set(s.rx, s.ry, s.rz, "ZYX");
    this.pivot.updateMatrixWorld(true);
    this.simulations.forEach((sim) => sim.step(dt, this.pivot.matrixWorld, s.leftArm, s.rightArm, enabled));
  }

  render() { if (!this.disposed) this.renderer.render(this.scene, this.camera); }
  cancelLoad() { this.loadVersion++; this.loadController?.abort(); this.loadController = null; }
  dispose() {
    this.disposed = true; this.cancelLoad();
    this.simulations.forEach((s) => s.dispose()); this.simulations = [];
    disposeObject(this.shirt); this.scene.clear(); this.renderer.dispose(); this.renderer.forceContextLoss();
  }
}