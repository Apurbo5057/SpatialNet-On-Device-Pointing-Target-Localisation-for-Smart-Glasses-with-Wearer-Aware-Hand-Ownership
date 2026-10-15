# SpatialNet evaluation

## 1. Scene classification and gates (test)

3029 images. Scene accuracy 98.71% (95% CI 98.2-99.1), macro-F1 97.29%, balanced accuracy 96.49%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 97.3 | 93.4 | 90.6-95.4 | 95.3 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.9 | 99.6 | 99.2-99.8 | 99.3 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 397 | 0 | 0 | 28 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 11 | 0 | 0 | 2593 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.7 | 99.6 | 11 | 28 | 0.9981 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 2. Hand localisation by scene (test)

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 73.1 | 72.5-73.6 | 76.8 | 97.4 | 91.4 | 68.7 | 55.7 |
| All hand-present | 2604 | 73.1 | 72.5-73.6 | 76.8 | 97.4 | 91.4 | 68.7 | 55.7 |

## 3. Target localisation (test, pointing at target)

Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of the image, far: beyond).

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2604 | 52.3 | 51.3-53.2 | 57.9 | 77.6 | 59.1 | 30.6 | 21.2 |
| Gap on-object | 1796 | 56.4 | 55.3-57.5 | 61.0 | 84.0 | 65.3 | 35.0 | 24.6 |
| Gap near | 599 | 47.8 | 45.8-49.7 | 50.5 | 71.3 | 51.1 | 23.0 | 15.2 |
| Gap far | 209 | 30.0 | 26.2-34.1 | 19.5 | 40.7 | 28.7 | 14.8 | 10.0 |

## 4. Fingertip and pointing-ray error by scene (test)

| Subset | n | Mean px | Mean 95% CI | Median px | P90 px | <5 px % | <10 px % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 3.90 | 3.72-4.11 | 3.28 | 5.75 | 83.4 | 97.7 |
| All pointing | 2604 | 3.90 | 3.72-4.11 | 3.28 | 5.75 | 83.4 | 97.7 |

Pixels at the 160x160 model input. Ray error for "Pointing, no target" is measured against the hand-centre-to-fingertip direction.

| Subset | n | Mean deg | Mean 95% CI | Median deg | P90 deg | <15 deg % | <30 deg % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 33.33 | 31.84-34.75 | 19.64 | 84.83 | 40.2 | 63.5 |
| All pointing | 2604 | 33.33 | 31.84-34.75 | 19.64 | 84.83 | 40.2 | 63.5 |

| Gap | n | Fingertip mean px | Ray mean deg | Missed pointing |
|---|---:|---:|---:|---:|
| on-object | 1796 | 3.87 | 36.7 | 9 |
| near | 599 | 3.81 | 23.4 | 1 |
| far | 209 | 4.38 | 32.6 | 1 |

## 5. Val-calibrated gate thresholds (test)

Highest threshold per gate with no missed positives on the 1524 validation images: hand 0.13, pointing 0.50, target 0.50.

3029 images. Scene accuracy 94.62% (95% CI 93.8-95.4), macro-F1 86.65%, balanced accuracy 80.92%. Gate thresholds hand/pointing/target = 0.13/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 99.6 | 61.9 | 57.2-66.4 | 76.3 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 94.1 | 100.0 | 99.8-100.0 | 97.0 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 263 | 0 | 0 | 162 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 1 | 0 | 0 | 2603 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.13 | 94.6 | 100.0 | 1 | 162 | 0.9981 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 6. Robustness to other people's hands (test with 1-2 pasted distractor hands)

3029 images. Scene accuracy 98.68% (95% CI 98.2-99.0), macro-F1 97.21%, balanced accuracy 96.38%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 97.3 | 93.2 | 90.4-95.2 | 95.2 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.9 | 99.6 | 99.2-99.8 | 99.2 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 396 | 0 | 0 | 29 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 11 | 0 | 0 | 2593 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.7 | 99.6 | 11 | 29 | 0.9981 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 73.0 | 72.4-73.6 | 76.6 | 97.3 | 91.4 | 68.9 | 55.6 |
| All hand-present | 2604 | 73.0 | 72.4-73.6 | 76.6 | 97.3 | 91.4 | 68.9 | 55.6 |

| Ownership | Value |
|---|---:|
| Hand images with distractors | 2377 |
| Ownership accuracy % (predicted hand nearer the wearer) | 99.71 |
| Ownership accuracy 95% CI | 99.4-99.9 |
| Predicted hand centre inside a distractor % | 0.13 |
| Pointing images with distractors | 2377 |
| Ownership accuracy on pointing images % | 99.71 |
| No-hand images with distractors | 425 |
| Still classified "No hand" % | 93.18 |

## 7. Headline comparison

| Metric | Test (default thresholds) | Test (val-calibrated thresholds) | Test + distractor hands |
|---|---:|---:|---:|
| Scene accuracy % | 98.7 | 94.6 | 98.7 |
| Scene macro-F1 % | 97.3 | 86.7 | 97.2 |
| Missed hand / pointing / target | 11/0/0 | 1/0/0 | 11/0/0 |
| False alarms hand / pointing / target | 28/0/0 | 162/0/0 | 29/0/0 |
| Hand IoU mean % (all hand-present) | 73.1 | 73.1 | 73.0 |
| Hand IoU>=0.5 % | 91.4 | 91.4 | 91.4 |
| Target IoU mean % | 52.3 | 52.3 | 52.9 |
| Target IoU>=0.5 % | 59.1 | 59.1 | 60.4 |
| Fingertip error mean px | 3.90 | 3.90 | 4.15 |

## 8. Efficiency

| Property | Value |
|---|---:|
| Parameters | 722,104 |
|   backbone / heads | 143,136 / 578,968 |
| Compute per image | 0.231 GFLOPs |
| Input | 160 x 160 RGB |
| Size FP32 / FP16 / INT8 | 3064 KB / 1600 KB / 2733 KB |
| Latency CPU, 1 thread, batch 1 | 28.19 ms median, 29.41 ms p90 (35 FPS) |
| Latency CPU, 4 threads, batch 1 | 28.13 ms median, 29.40 ms p90 (36 FPS) |
| Latency GPU (Tesla T4), batch 1 | 16.97 ms median, 17.78 ms p90 (59 FPS) |

CPU: x86_64. Latency is PyTorch eager mode, not an on-device runtime.
