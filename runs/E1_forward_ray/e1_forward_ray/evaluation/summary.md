# SpatialNet evaluation

## 1. Scene classification and gates (test)

3029 images. Scene accuracy 98.55% (95% CI 98.1-98.9), macro-F1 96.95%, balanced accuracy 96.30%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 96.4 | 93.2 | 90.4-95.2 | 94.7 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.9 | 99.4 | 99.1-99.7 | 99.2 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 396 | 0 | 0 | 29 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 15 | 0 | 0 | 2589 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.5 | 99.4 | 15 | 29 | 0.9984 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 2. Hand localisation by scene (test)

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 72.9 | 72.3-73.4 | 76.3 | 97.7 | 91.9 | 68.4 | 54.3 |
| All hand-present | 2604 | 72.9 | 72.3-73.4 | 76.3 | 97.7 | 91.9 | 68.4 | 54.3 |

## 3. Target localisation (test, pointing at target)

Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of the image, far: beyond).

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2604 | 52.7 | 51.7-53.6 | 57.5 | 77.6 | 59.2 | 32.0 | 22.4 |
| Gap on-object | 1796 | 56.8 | 55.8-57.9 | 61.9 | 83.7 | 65.6 | 36.6 | 25.2 |
| Gap near | 599 | 48.1 | 46.2-50.1 | 49.7 | 72.1 | 49.9 | 25.4 | 18.5 |
| Gap far | 209 | 30.2 | 26.2-34.2 | 19.4 | 41.6 | 30.6 | 12.0 | 9.1 |

## 4. Fingertip and pointing-ray error by scene (test)

| Subset | n | Mean px | Mean 95% CI | Median px | P90 px | <5 px % | <10 px % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 3.93 | 3.77-4.11 | 3.29 | 5.86 | 82.6 | 97.4 |
| All pointing | 2604 | 3.93 | 3.77-4.11 | 3.29 | 5.86 | 82.6 | 97.4 |

Pixels at the 160x160 model input. Ray error for "Pointing, no target" is measured against the hand-centre-to-fingertip direction.

| Subset | n | Mean deg | Mean 95% CI | Median deg | P90 deg | <15 deg % | <30 deg % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 32.67 | 31.29-33.98 | 20.19 | 79.65 | 41.1 | 64.2 |
| All pointing | 2604 | 32.67 | 31.29-33.98 | 20.19 | 79.65 | 41.1 | 64.2 |

| Gap | n | Fingertip mean px | Ray mean deg | Missed pointing |
|---|---:|---:|---:|---:|
| on-object | 1796 | 3.84 | 35.9 | 10 |
| near | 599 | 3.93 | 22.8 | 1 |
| far | 209 | 4.63 | 33.2 | 4 |

## 5. Val-calibrated gate thresholds (test)

Highest threshold per gate with no missed positives on the 1524 validation images: hand 0.20, pointing 0.50, target 0.50.

3029 images. Scene accuracy 95.71% (95% CI 94.9-96.4), macro-F1 89.78%, balanced accuracy 84.80%. Gate thresholds hand/pointing/target = 0.20/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 99.7 | 69.6 | 65.1-73.8 | 82.0 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 95.3 | 100.0 | 99.8-100.0 | 97.6 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 296 | 0 | 0 | 129 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 1 | 0 | 0 | 2603 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.20 | 95.7 | 100.0 | 1 | 129 | 0.9984 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 6. Robustness to other people's hands (test with 1-2 pasted distractor hands)

3029 images. Scene accuracy 98.45% (95% CI 97.9-98.8), macro-F1 96.74%, balanced accuracy 96.14%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 95.9 | 92.9 | 90.1-95.0 | 94.4 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.9 | 99.3 | 99.0-99.6 | 99.1 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 395 | 0 | 0 | 30 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 17 | 0 | 0 | 2587 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.4 | 99.3 | 17 | 30 | 0.9977 | 0.9996 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 72.8 | 72.2-73.4 | 76.2 | 97.7 | 91.7 | 67.5 | 54.2 |
| All hand-present | 2604 | 72.8 | 72.2-73.4 | 76.2 | 97.7 | 91.7 | 67.5 | 54.2 |

| Ownership | Value |
|---|---:|
| Hand images with distractors | 2377 |
| Ownership accuracy % (predicted hand nearer the wearer) | 99.87 |
| Ownership accuracy 95% CI | 99.6-100.0 |
| Predicted hand centre inside a distractor % | 0.08 |
| Pointing images with distractors | 2377 |
| Ownership accuracy on pointing images % | 99.87 |
| No-hand images with distractors | 425 |
| Still classified "No hand" % | 92.94 |

## 7. Headline comparison

| Metric | Test (default thresholds) | Test (val-calibrated thresholds) | Test + distractor hands |
|---|---:|---:|---:|
| Scene accuracy % | 98.5 | 95.7 | 98.4 |
| Scene macro-F1 % | 96.9 | 89.8 | 96.7 |
| Missed hand / pointing / target | 15/0/0 | 1/0/0 | 17/0/0 |
| False alarms hand / pointing / target | 29/0/0 | 129/0/0 | 30/0/0 |
| Hand IoU mean % (all hand-present) | 72.9 | 72.9 | 72.8 |
| Hand IoU>=0.5 % | 91.9 | 91.9 | 91.7 |
| Target IoU mean % | 52.7 | 52.7 | 53.4 |
| Target IoU>=0.5 % | 59.2 | 59.2 | 61.1 |
| Fingertip error mean px | 3.93 | 3.93 | 3.98 |

## 8. Efficiency

| Property | Value |
|---|---:|
| Parameters | 722,104 |
|   backbone / heads | 143,136 / 578,968 |
| Compute per image | 0.231 GFLOPs |
| Input | 160 x 160 RGB |
| Size FP32 / FP16 / INT8 | 3064 KB / 1600 KB / 2733 KB |
| Latency CPU, 1 thread, batch 1 | 28.67 ms median, 29.51 ms p90 (35 FPS) |
| Latency CPU, 4 threads, batch 1 | 29.28 ms median, 30.24 ms p90 (34 FPS) |
| Latency GPU (Tesla T4), batch 1 | 16.76 ms median, 19.06 ms p90 (60 FPS) |

CPU: x86_64. Latency is PyTorch eager mode, not an on-device runtime.
