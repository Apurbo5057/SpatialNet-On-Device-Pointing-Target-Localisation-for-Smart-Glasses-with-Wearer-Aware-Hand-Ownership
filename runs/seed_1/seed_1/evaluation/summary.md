# SpatialNet evaluation

## 1. Scene classification and gates (test)

3029 images. Scene accuracy 98.68% (95% CI 98.2-99.0), macro-F1 97.22%, balanced accuracy 96.57%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 96.8 | 93.6 | 90.9-95.6 | 95.2 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 99.0 | 99.5 | 99.1-99.7 | 99.2 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 398 | 0 | 0 | 27 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 13 | 0 | 0 | 2591 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.7 | 99.5 | 13 | 27 | 0.9987 | 0.9998 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 2. Hand localisation by scene (test)

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 74.0 | 73.4-74.5 | 77.3 | 98.0 | 92.8 | 69.9 | 57.0 |
| All hand-present | 2604 | 74.0 | 73.4-74.5 | 77.3 | 98.0 | 92.8 | 69.9 | 57.0 |

## 3. Target localisation (test, pointing at target)

Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of the image, far: beyond).

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2604 | 53.0 | 52.0-53.9 | 58.1 | 78.2 | 59.0 | 31.7 | 23.0 |
| Gap on-object | 1796 | 57.2 | 56.1-58.3 | 62.2 | 84.2 | 65.5 | 36.4 | 26.6 |
| Gap near | 599 | 48.7 | 46.7-50.8 | 50.2 | 73.6 | 50.3 | 24.4 | 17.4 |
| Gap far | 209 | 28.9 | 25.1-33.0 | 18.9 | 39.7 | 27.8 | 12.0 | 8.6 |

## 4. Fingertip and pointing-ray error by scene (test)

| Subset | n | Mean px | Mean 95% CI | Median px | P90 px | <5 px % | <10 px % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 3.80 | 3.67-3.94 | 3.28 | 5.75 | 83.2 | 97.7 |
| All pointing | 2604 | 3.80 | 3.67-3.94 | 3.28 | 5.75 | 83.2 | 97.7 |

Pixels at the 160x160 model input. Ray error for "Pointing, no target" is measured against the hand-centre-to-fingertip direction.

| Subset | n | Mean deg | Mean 95% CI | Median deg | P90 deg | <15 deg % | <30 deg % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 33.47 | 32.08-34.84 | 20.16 | 85.74 | 40.5 | 62.9 |
| All pointing | 2604 | 33.47 | 32.08-34.84 | 20.16 | 85.74 | 40.5 | 62.9 |

| Gap | n | Fingertip mean px | Ray mean deg | Missed pointing |
|---|---:|---:|---:|---:|
| on-object | 1796 | 3.76 | 37.2 | 8 |
| near | 599 | 3.71 | 22.7 | 2 |
| far | 209 | 4.42 | 32.3 | 3 |

## 5. Val-calibrated gate thresholds (test)

Highest threshold per gate with no missed positives on the 1524 validation images: hand 0.27, pointing 0.50, target 0.50.

3029 images. Scene accuracy 97.46% (95% CI 96.8-98.0), macro-F1 94.33%, balanced accuracy 91.24%. Gate thresholds hand/pointing/target = 0.27/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 99.2 | 82.6 | 78.7-85.9 | 90.1 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 97.2 | 99.9 | 99.7-100.0 | 98.5 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 351 | 0 | 0 | 74 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 3 | 0 | 0 | 2601 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.27 | 97.5 | 99.9 | 3 | 74 | 0.9987 | 0.9998 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 6. Robustness to other people's hands (test with 1-2 pasted distractor hands)

3029 images. Scene accuracy 98.51% (95% CI 98.0-98.9), macro-F1 96.87%, balanced accuracy 96.08%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 96.6 | 92.7 | 89.8-94.8 | 94.6 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.8 | 99.5 | 99.1-99.7 | 99.1 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 394 | 0 | 0 | 31 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 14 | 0 | 0 | 2590 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.5 | 99.5 | 14 | 31 | 0.9983 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 73.7 | 73.2-74.3 | 77.2 | 97.8 | 92.5 | 69.8 | 56.3 |
| All hand-present | 2604 | 73.7 | 73.2-74.3 | 77.2 | 97.8 | 92.5 | 69.8 | 56.3 |

| Ownership | Value |
|---|---:|
| Hand images with distractors | 2377 |
| Ownership accuracy % (predicted hand nearer the wearer) | 99.79 |
| Ownership accuracy 95% CI | 99.5-99.9 |
| Predicted hand centre inside a distractor % | 0.21 |
| Pointing images with distractors | 2377 |
| Ownership accuracy on pointing images % | 99.79 |
| No-hand images with distractors | 425 |
| Still classified "No hand" % | 92.71 |

## 7. Headline comparison

| Metric | Test (default thresholds) | Test (val-calibrated thresholds) | Test + distractor hands |
|---|---:|---:|---:|
| Scene accuracy % | 98.7 | 97.5 | 98.5 |
| Scene macro-F1 % | 97.2 | 94.3 | 96.9 |
| Missed hand / pointing / target | 13/0/0 | 3/0/0 | 14/0/0 |
| False alarms hand / pointing / target | 27/0/0 | 74/0/0 | 31/0/0 |
| Hand IoU mean % (all hand-present) | 74.0 | 74.0 | 73.7 |
| Hand IoU>=0.5 % | 92.8 | 92.8 | 92.5 |
| Target IoU mean % | 53.0 | 53.0 | 53.4 |
| Target IoU>=0.5 % | 59.0 | 59.0 | 59.9 |
| Fingertip error mean px | 3.80 | 3.80 | 3.98 |

## 8. Efficiency

| Property | Value |
|---|---:|
| Parameters | 722,104 |
|   backbone / heads | 143,136 / 578,968 |
| Compute per image | 0.231 GFLOPs |
| Input | 160 x 160 RGB |
| Size FP32 / FP16 / INT8 | 3064 KB / 1600 KB / 2733 KB |
| Latency CPU, 1 thread, batch 1 | 24.93 ms median, 26.58 ms p90 (40 FPS) |
| Latency CPU, 4 threads, batch 1 | 25.29 ms median, 27.99 ms p90 (40 FPS) |
| Latency GPU (Tesla T4), batch 1 | 15.99 ms median, 16.44 ms p90 (63 FPS) |

CPU: x86_64. Latency is PyTorch eager mode, not an on-device runtime.
