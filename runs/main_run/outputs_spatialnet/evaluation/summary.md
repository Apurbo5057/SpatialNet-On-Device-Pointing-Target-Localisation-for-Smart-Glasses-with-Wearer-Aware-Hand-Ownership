# SpatialNet evaluation

## 1. Scene classification and gates (test)

3029 images. Scene accuracy 98.58% (95% CI 98.1-98.9), macro-F1 97.02%, balanced accuracy 96.42%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 96.4 | 93.4 | 90.6-95.4 | 94.9 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.9 | 99.4 | 99.1-99.7 | 99.2 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 397 | 0 | 0 | 28 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 15 | 0 | 0 | 2589 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.6 | 99.4 | 15 | 28 | 0.9984 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 2. Hand localisation by scene (test)

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 73.7 | 73.1-74.2 | 77.1 | 97.7 | 92.6 | 69.5 | 56.3 |
| All hand-present | 2604 | 73.7 | 73.1-74.2 | 77.1 | 97.7 | 92.6 | 69.5 | 56.3 |

## 3. Target localisation (test, pointing at target)

Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of the image, far: beyond).

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2604 | 53.3 | 52.3-54.3 | 58.2 | 78.6 | 58.9 | 31.8 | 24.0 |
| Gap on-object | 1796 | 57.9 | 56.8-59.0 | 63.3 | 84.6 | 66.0 | 37.2 | 28.1 |
| Gap near | 599 | 47.8 | 45.8-49.8 | 49.4 | 72.8 | 49.2 | 22.5 | 16.4 |
| Gap far | 209 | 29.8 | 26.0-33.6 | 19.7 | 43.1 | 25.4 | 11.5 | 10.5 |

## 4. Fingertip and pointing-ray error by scene (test)

| Subset | n | Mean px | Mean 95% CI | Median px | P90 px | <5 px % | <10 px % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 3.89 | 3.73-4.06 | 3.32 | 5.69 | 82.8 | 97.6 |
| All pointing | 2604 | 3.89 | 3.73-4.06 | 3.32 | 5.69 | 82.8 | 97.6 |

Pixels at the 160x160 model input. Ray error for "Pointing, no target" is measured against the hand-centre-to-fingertip direction.

| Subset | n | Mean deg | Mean 95% CI | Median deg | P90 deg | <15 deg % | <30 deg % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 32.23 | 30.72-33.64 | 18.59 | 82.75 | 43.0 | 64.9 |
| All pointing | 2604 | 32.23 | 30.72-33.64 | 18.59 | 82.75 | 43.0 | 64.9 |

| Gap | n | Fingertip mean px | Ray mean deg | Missed pointing |
|---|---:|---:|---:|---:|
| on-object | 1796 | 3.79 | 36.0 | 11 |
| near | 599 | 4.09 | 22.0 | 1 |
| far | 209 | 4.16 | 29.0 | 3 |

## 5. Val-calibrated gate thresholds (test)

Highest threshold per gate with no missed positives on the 1524 validation images: hand 0.22, pointing 0.50, target 0.50.

3029 images. Scene accuracy 96.14% (95% CI 95.4-96.8), macro-F1 90.92%, balanced accuracy 86.24%. Gate thresholds hand/pointing/target = 0.22/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 100.0 | 72.5 | 68.0-76.5 | 84.0 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 95.7 | 100.0 | 99.9-100.0 | 97.8 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 308 | 0 | 0 | 117 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 0 | 0 | 0 | 2604 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.22 | 96.1 | 100.0 | 0 | 117 | 0.9984 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 6. Robustness to other people's hands (test with 1-2 pasted distractor hands)

3029 images. Scene accuracy 98.32% (95% CI 97.8-98.7), macro-F1 96.44%, balanced accuracy 95.58%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 96.1 | 91.8 | 88.8-94.0 | 93.9 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.7 | 99.4 | 99.0-99.6 | 99.0 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 390 | 0 | 0 | 35 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 16 | 0 | 0 | 2588 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.3 | 99.4 | 16 | 35 | 0.9977 | 0.9996 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 73.3 | 72.7-73.9 | 76.8 | 97.6 | 91.9 | 68.2 | 55.8 |
| All hand-present | 2604 | 73.3 | 72.7-73.9 | 76.8 | 97.6 | 91.9 | 68.2 | 55.8 |

| Ownership | Value |
|---|---:|
| Hand images with distractors | 2377 |
| Ownership accuracy % (predicted hand nearer the wearer) | 99.71 |
| Ownership accuracy 95% CI | 99.4-99.9 |
| Predicted hand centre inside a distractor % | 0.21 |
| Pointing images with distractors | 2377 |
| Ownership accuracy on pointing images % | 99.71 |
| No-hand images with distractors | 425 |
| Still classified "No hand" % | 91.76 |

## 7. Headline comparison

| Metric | Test (default thresholds) | Test (val-calibrated thresholds) | Test + distractor hands |
|---|---:|---:|---:|
| Scene accuracy % | 98.6 | 96.1 | 98.3 |
| Scene macro-F1 % | 97.0 | 90.9 | 96.4 |
| Missed hand / pointing / target | 15/0/0 | 0/0/0 | 16/0/0 |
| False alarms hand / pointing / target | 28/0/0 | 117/0/0 | 35/0/0 |
| Hand IoU mean % (all hand-present) | 73.7 | 73.7 | 73.3 |
| Hand IoU>=0.5 % | 92.6 | 92.6 | 91.9 |
| Target IoU mean % | 53.3 | 53.3 | 53.7 |
| Target IoU>=0.5 % | 58.9 | 58.9 | 60.1 |
| Fingertip error mean px | 3.89 | 3.89 | 4.02 |

## 8. Efficiency

| Property | Value |
|---|---:|
| Parameters | 722,104 |
|   backbone / heads | 143,136 / 578,968 |
| Compute per image | 0.231 GFLOPs |
| Input | 160 x 160 RGB |
| Size FP32 / FP16 / INT8 | 3064 KB / 1600 KB / 2733 KB |
| Latency CPU, 1 thread, batch 1 | 25.73 ms median, 28.03 ms p90 (39 FPS) |
| Latency CPU, 4 threads, batch 1 | 25.83 ms median, 27.01 ms p90 (39 FPS) |
| Latency GPU (Tesla T4), batch 1 | 15.90 ms median, 16.30 ms p90 (63 FPS) |

CPU: x86_64. Latency is PyTorch eager mode, not an on-device runtime.
