# Training annotation format

One label file per image: `<split>/labels/<name>.txt`, same basename as the image in `<split>/images/`.
No other files (keypoints, metadata) are needed for training.

## Layout

Four lines, 13 numbers:

```
H P T
hand_cx hand_cy hand_w hand_h
tip_cx tip_cy
roi_cx roi_cy roi_w roi_h
```

| Line | Values | Meaning |
|---|---|---|
| 1 | `H P T` | Flags, each `0` or `1`. H = the wearer's hand is visible, P = the hand is pointing, T = the pointed-at target is visible |
| 2 | `cx cy w h`     | Hand box |
| 3 | `cx cy`         | Fingertip center point |
| 4 | `cx cy w h`     | Target (ROI) box |

All coordinates are normalised to `[0, 1]` of the image width and height. Boxes are centre-based, as in YOLO.

## Rules

- Only four flag combinations are valid:

  | H P T | Scene |
  |---|---|
  | `0 0 0` | no hand |
  | `1 0 0` | hand, not pointing |
  | `1 1 0` | pointing, no target |
  | `1 1 1` | pointing at a target |

- A flag that is `1` requires its line: H needs a hand box with `w, h > 0`, P needs the fingertip, T needs the target box with `w, h > 0`.
- A line whose flag is `0` is written as zeros. Never leave it out, so every file has 13 numbers.
- Label only the wearer's hand. Other people's hands are background.
- An empty file is invalid. A frame with no hand is written as `0 0 0` plus zero lines.

## Examples

Pointing at a target:
```
1 1 1
0.205357 0.669643 0.401786 0.660714
0.357143 0.383929
0.381696 0.466518 0.263393 0.227679
```

Pointing, no target:
```
1 1 0
0.205357 0.669643 0.401786 0.660714
0.357143 0.383929
0 0 0 0
```

Hand, not pointing:
```
1 0 0
0.512000 0.780000 0.300000 0.420000
0 0
0 0 0 0
```

No hand:
```
0 0 0
0 0 0 0
0 0
0 0 0 0
```

## Folder layout

```
<dataset>/
  train/  images/  labels/
  val/    images/  labels/
  test/   images/  labels/
```

`metadata/` may stay alongside for provenance, but training does not read it.

The loader (`annotation.py`) rejects any file that breaks these rules and names the file.
