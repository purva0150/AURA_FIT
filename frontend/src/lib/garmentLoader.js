import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// This catalog uses solid cloth tints, not the asset's printed-logo textures.
// Skip unused embedded image decoding entirely, including on cancelled navigation.
export async function loadGarment(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Garment request failed (${response.status})`);
  const buffer = await response.arrayBuffer();
  if (signal.aborted) throw new DOMException("Garment load cancelled", "AbortError");
  const loader = new GLTFLoader();
  loader.register((parser) => ({
    name: "AURA_SOLID_CLOTH",
    beforeRoot() {
      parser.json.meshes?.forEach((mesh) => mesh.primitives.forEach((p) => { delete p.material; }));
    },
  }));
  return loader.parseAsync(buffer, new URL(".", url).href);
}