# SpatialNet evaluation

## 1. Scene classification and gates (test)

3029 images. Scene accuracy 98.84% (95% CI 98.4-99.2), macro-F1 97.60%, balanced accuracy 97.46%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 96.2 | 95.5 | 93.1-97.1 | 95.9 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 99.3 | 99.4 | 99.0-99.6 | 99.3 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 406 | 0 | 0 | 19 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 16 | 0 | 0 | 2588 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.8 | 99.4 | 16 | 19 | 0.9985 | 0.9998 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 2. Hand localisation by scene (test)

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 76.4 | 75.7-76.9 | 80.4 | 97.4 | 92.5 | 78.7 | 68.1 |
| All hand-present | 2604 | 76.4 | 75.7-76.9 | 80.4 | 97.4 | 92.5 | 78.7 | 68.1 |

## 3. Target localisation (test, pointing at target)

Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of the image, far: beyond).

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2604 | 56.5 | 55.5-57.5 | 63.4 | 79.4 | 65.1 | 40.3 | 33.0 |
| Gap on-object | 1796 | 60.7 | 59.5-61.8 | 66.3 | 86.1 | 71.2 | 43.9 | 36.1 |
| Gap near | 599 | 52.3 | 49.9-54.6 | 59.1 | 72.0 | 57.4 | 36.2 | 29.4 |
| Gap far | 209 | 33.0 | 28.8-37.3 | 20.2 | 43.5 | 34.9 | 20.6 | 16.7 |

## 4. Fingertip and pointing-ray error by scene (test)

| Subset | n | Mean px | Mean 95% CI | Median px | P90 px | <5 px % | <10 px % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 4.24 | 4.03-4.45 | 3.55 | 6.41 | 76.6 | 97.8 |
| All pointing | 2604 | 4.24 | 4.03-4.45 | 3.55 | 6.41 | 76.6 | 97.8 |

Pixels at the 320x320 model input. Ray error for "Pointing, no target" is measured against the hand-centre-to-fingertip direction.

| Subset | n | Mean deg | Mean 95% CI | Median deg | P90 deg | <15 deg % | <30 deg % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 31.57 | 30.18-32.93 | 18.90 | 78.03 | 42.6 | 65.9 |
| All pointing | 2604 | 31.57 | 30.18-32.93 | 18.90 | 78.03 | 42.6 | 65.9 |

| Gap | n | Fingertip mean px | Ray mean deg | Missed pointing |
|---|---:|---:|---:|---:|
| on-object | 1796 | 4.07 | 35.1 | 12 |
| near | 599 | 4.30 | 21.3 | 2 |
| far | 209 | 5.51 | 30.5 | 2 |

## 5. Val-calibrated gate thresholds (test)

Highest threshold per gate with no missed positives on the 1524 validation images: hand 0.25, pointing 0.50, target 0.50.

3029 images. Scene accuracy 98.15% (95% CI 97.6-98.6), macro-F1 95.98%, balanced accuracy 93.81%. Gate thresholds hand/pointing/target = 0.25/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 98.9 | 87.8 | 84.3-90.5 | 93.0 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.0 | 99.8 | 99.6-99.9 | 98.9 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 373 | 0 | 0 | 52 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 4 | 0 | 0 | 2600 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.25 | 98.2 | 99.8 | 4 | 52 | 0.9985 | 0.9998 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 6. Robustness to other people's hands (test with 1-2 pasted distractor hands)

3029 images. Scene accuracy 98.75% (95% CI 98.3-99.1), macro-F1 97.38%, balanced accuracy 97.10%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 96.2 | 94.8 | 92.3-96.6 | 95.5 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 99.2 | 99.4 | 99.0-99.6 | 99.3 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 403 | 0 | 0 | 22 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 16 | 0 | 0 | 2588 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.7 | 99.4 | 16 | 22 | 0.9984 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 76.1 | 75.5-76.7 | 80.2 | 97.0 | 92.1 | 78.0 | 67.8 |
| All hand-present | 2604 | 76.1 | 75.5-76.7 | 80.2 | 97.0 | 92.1 | 78.0 | 67.8 |

| Ownership | Value |
|---|---:|
| Hand images with distractors | 2377 |
| Ownership accuracy % (predicted hand nearer the wearer) | 99.79 |
| Ownership accuracy 95% CI | 99.5-99.9 |
| Predicted hand centre inside a distractor % | 0.21 |
| Pointing images with distractors | 2377 |
| Ownership accuracy on pointing images % | 99.79 |
| No-hand images with distractors | 425 |
| Still classified "No hand" % | 94.82 |

## 7. Headline comparison

| Metric | Test (default thresholds) | Test (val-calibrated thresholds) | Test + distractor hands |
|---|---:|---:|---:|
| Scene accuracy % | 98.8 | 98.2 | 98.7 |
| Scene macro-F1 % | 97.6 | 96.0 | 97.4 |
| Missed hand / pointing / target | 16/0/0 | 4/0/0 | 16/0/0 |
| False alarms hand / pointing / target | 19/0/0 | 52/0/0 | 22/0/0 |
| Hand IoU mean % (all hand-present) | 76.4 | 76.4 | 76.1 |
| Hand IoU>=0.5 % | 92.5 | 92.5 | 92.1 |
| Target IoU mean % | 56.5 | 56.5 | 57.1 |
| Target IoU>=0.5 % | 65.1 | 65.1 | 66.1 |
| Fingertip error mean px | 4.24 | 4.24 | 4.46 |

## 8. Efficiency

| Property | Value |
|---|---:|
| Parameters | 722,104 |
|   backbone / heads | 143,136 / 578,968 |
| Compute per image | 0.923 GFLOPs |
| Input | 320 x 320 RGB |
| Size FP32 / FP16 / INT8 | 3064 KB / 1600 KB / 2733 KB |
| Latency CPU, 1 thread, batch 1 | 50.79 ms median, 53.89 ms p90 (20 FPS) |
| Latency CPU, 4 threads, batch 1 | 44.40 ms median, 45.33 ms p90 (23 FPS) |
| Latency GPU (Tesla T4), batch 1 | 16.14 ms median, 16.74 ms p90 (62 FPS) |

CPU: x86_64. Latency is PyTorch eager mode, not an on-device runtime.
