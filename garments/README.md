# Garment Assets

Each garment gets its own folder with one image + anchor-point JSON per rotation angle.

```
garments/
    shirt1/
        shirt1_front.png
        shirt1_front.json
        shirt1_front45.png
        shirt1_front45.json
        shirt1_side.png
        shirt1_side.json
        shirt1_back45.png
        shirt1_back45.json
        shirt1_back.png
        shirt1_back.json
```

- Images: transparent-background PNGs (garment only, no mannequin/model).
- JSON: anchor points used for TPS warping, keyed by body landmark name.

## Example anchor JSON (`shirt1_front.json`)

```json
{
  "garment_id": "shirt1",
  "angle": "front",
  "anchors": {
    "left_shoulder": [120, 40],
    "right_shoulder": [280, 40],
    "left_hip": [110, 340],
    "right_hip": [290, 340]
  },
  "dominant_color": "#2f5fa8"
}
```

`dominant_color` feeds the Phase 11 recommendation system later — fill it in when you add the garment.

## Angle capture setup (no 3D scanner needed)

Use a mannequin or a volunteer on a rotating platform (a $20–30 turntable works) and a fixed camera height:
1. front (0°)
2. front-45 (45°)
3. side (90°)
4. back-45 (135°)
5. back (180°)

Mirror front-45/side/back-45 for the other side if you want smoother transitions, rather than shooting a 6th/7th/8th angle.
