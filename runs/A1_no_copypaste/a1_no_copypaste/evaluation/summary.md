# SpatialNet evaluation

## 1. Scene classification and gates (test)

3029 images. Scene accuracy 98.65% (95% CI 98.2-99.0), macro-F1 97.15%, balanced accuracy 96.46%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 96.8 | 93.4 | 90.6-95.4 | 95.1 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.9 | 99.5 | 99.1-99.7 | 99.2 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 397 | 0 | 0 | 28 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 13 | 0 | 0 | 2591 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.6 | 99.5 | 13 | 28 | 0.9987 | 0.9998 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 2. Hand localisation by scene (test)

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 73.1 | 72.5-73.6 | 76.4 | 97.5 | 91.9 | 67.9 | 55.0 |
| All hand-present | 2604 | 73.1 | 72.5-73.6 | 76.4 | 97.5 | 91.9 | 67.9 | 55.0 |

## 3. Target localisation (test, pointing at target)

Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of the image, far: beyond).

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2604 | 52.1 | 51.0-53.1 | 57.0 | 76.9 | 58.8 | 30.5 | 22.2 |
| Gap on-object | 1796 | 56.4 | 55.3-57.5 | 61.5 | 83.3 | 66.2 | 35.0 | 25.7 |
| Gap near | 599 | 46.8 | 44.9-48.8 | 48.7 | 69.8 | 47.7 | 22.9 | 16.9 |
| Gap far | 209 | 29.8 | 26.1-33.3 | 20.6 | 42.6 | 27.3 | 12.9 | 7.7 |

## 4. Fingertip and pointing-ray error by scene (test)

| Subset | n | Mean px | Mean 95% CI | Median px | P90 px | <5 px % | <10 px % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 3.80 | 3.66-3.94 | 3.31 | 5.73 | 83.6 | 97.7 |
| All pointing | 2604 | 3.80 | 3.66-3.94 | 3.31 | 5.73 | 83.6 | 97.7 |

Pixels at the 160x160 model input. Ray error for "Pointing, no target" is measured against the hand-centre-to-fingertip direction.

| Subset | n | Mean deg | Mean 95% CI | Median deg | P90 deg | <15 deg % | <30 deg % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 34.06 | 32.57-35.55 | 20.57 | 89.65 | 41.0 | 62.3 |
| All pointing | 2604 | 34.06 | 32.57-35.55 | 20.57 | 89.65 | 41.0 | 62.3 |

| Gap | n | Fingertip mean px | Ray mean deg | Missed pointing |
|---|---:|---:|---:|---:|
| on-object | 1796 | 3.74 | 38.2 | 9 |
| near | 599 | 3.83 | 22.9 | 0 |
| far | 209 | 4.23 | 30.5 | 4 |

## 5. Val-calibrated gate thresholds (test)

Highest threshold per gate with no missed positives on the 1524 validation images: hand 0.24, pointing 0.50, target 0.50.

3029 images. Scene accuracy 96.47% (95% CI 95.7-97.1), macro-F1 91.85%, balanced accuracy 87.71%. Gate thresholds hand/pointing/target = 0.24/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 99.1 | 75.5 | 71.2-79.4 | 85.7 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 96.2 | 99.9 | 99.7-100.0 | 98.0 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 321 | 0 | 0 | 104 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 3 | 0 | 0 | 2601 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.24 | 96.5 | 99.9 | 3 | 104 | 0.9987 | 0.9998 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 6. Robustness to other people's hands (test with 1-2 pasted distractor hands)

3029 images. Scene accuracy 97.42% (95% CI 96.8-97.9), macro-F1 94.45%, balanced accuracy 92.79%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 94.8 | 86.4 | 82.8-89.3 | 90.4 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 97.8 | 99.2 | 98.8-99.5 | 98.5 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 367 | 0 | 0 | 58 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 20 | 0 | 0 | 2584 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 97.4 | 99.2 | 20 | 58 | 0.9948 | 0.9991 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 71.9 | 71.3-72.5 | 75.8 | 96.3 | 90.5 | 66.5 | 52.8 |
| All hand-present | 2604 | 71.9 | 71.3-72.5 | 75.8 | 96.3 | 90.5 | 66.5 | 52.8 |

| Ownership | Value |
|---|---:|
| Hand images with distractors | 2377 |
| Ownership accuracy % (predicted hand nearer the wearer) | 98.70 |
| Ownership accuracy 95% CI | 98.2-99.1 |
| Predicted hand centre inside a distractor % | 1.22 |
| Pointing images with distractors | 2377 |
| Ownership accuracy on pointing images % | 98.70 |
| No-hand images with distractors | 425 |
| Still classified "No hand" % | 86.35 |

## 7. Headline comparison

| Metric | Test (default thresholds) | Test (val-calibrated thresholds) | Test + distractor hands |
|---|---:|---:|---:|
| Scene accuracy % | 98.6 | 96.5 | 97.4 |
| Scene macro-F1 % | 97.2 | 91.8 | 94.5 |
| Missed hand / pointing / target | 13/0/0 | 3/0/0 | 20/0/0 |
| False alarms hand / pointing / target | 28/0/0 | 104/0/0 | 58/0/0 |
| Hand IoU mean % (all hand-present) | 73.1 | 73.1 | 71.9 |
| Hand IoU>=0.5 % | 91.9 | 91.9 | 90.5 |
| Target IoU mean % | 52.1 | 52.1 | 50.5 |
| Target IoU>=0.5 % | 58.8 | 58.8 | 56.1 |
| Fingertip error mean px | 3.80 | 3.80 | 4.40 |

## 8. Efficiency

| Property | Value |
|---|---:|
| Parameters | 722,104 |
|   backbone / heads | 143,136 / 578,968 |
| Compute per image | 0.231 GFLOPs |
| Input | 160 x 160 RGB |
| Size FP32 / FP16 / INT8 | 3064 KB / 1600 KB / 2733 KB |
| Latency CPU, 1 thread, batch 1 | 29.01 ms median, 30.09 ms p90 (34 FPS) |
| Latency CPU, 4 threads, batch 1 | 30.63 ms median, 32.55 ms p90 (33 FPS) |
| Latency GPU (Tesla T4), batch 1 | 17.46 ms median, 18.18 ms p90 (57 FPS) |

CPU: x86_64. Latency is PyTorch eager mode, not an on-device runtime.
