# Dataset Setup

## VITON-HD (start here — no approval needed)

1. Go to: https://github.com/shadow2496/VITON-HD
2. Follow the dataset link in that README (it was updated 2025-04-27; ignore any older Dropbox links you find elsewhere).
3. Unzip into `dataset/viton-hd/` so you end up with:
   ```
   dataset/viton-hd/
       train/
           image/
           cloth/
           ...
       test/
           image/
           cloth/
           ...
   ```
4. License: Creative Commons BY-NC 4.0 — non-commercial use, cite their CVPR 2021 paper if you publish anything using it.

## DressCode (apply now, approval takes up to a week)

1. Go to: https://github.com/aimagelab/dress-code
2. Fill out the dataset request form using your **college/institutional email** (Gmail/personal addresses are rejected).
3. Submit the signed release agreement form as instructed (typed signatures are not accepted — needs a real signature).
4. Requests are reviewed weekly, so apply now even if you're not using it yet.
5. Once approved, unzip into `dataset/dress-code/` following the same `train/test` + `image/cloth` structure as VITON-HD.

## Notes

- Neither dataset ships multi-angle (side/back) garment shots — both are front-view only.
- For the multi-angle overlay approach, plan to supplement these with your own turntable/mannequin photography once you reach Phase 8 (Garment Dataset Creation). Store those in `garments/` using the per-angle structure documented there.
