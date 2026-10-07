# SpatialNet evaluation

## 1. Scene classification and gates (test)

3029 images. Scene accuracy 98.65% (95% CI 98.2-99.0), macro-F1 97.12%, balanced accuracy 95.87%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 98.2 | 92.0 | 89.0-94.2 | 95.0 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.7 | 99.7 | 99.4-99.9 | 99.2 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 391 | 0 | 0 | 34 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 7 | 0 | 0 | 2597 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.6 | 99.7 | 7 | 34 | 0.9988 | 0.9998 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 2. Hand localisation by scene (test)

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 73.3 | 72.7-73.8 | 76.6 | 97.5 | 91.9 | 68.4 | 55.2 |
| All hand-present | 2604 | 73.3 | 72.7-73.8 | 76.6 | 97.5 | 91.9 | 68.4 | 55.2 |

## 3. Target localisation (test, pointing at target)

Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of the image, far: beyond).

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2604 | 52.8 | 51.8-53.7 | 58.1 | 77.3 | 60.0 | 32.4 | 23.4 |
| Gap on-object | 1796 | 57.3 | 56.2-58.4 | 62.1 | 84.3 | 66.8 | 37.0 | 26.6 |
| Gap near | 599 | 47.8 | 45.7-49.9 | 51.0 | 69.8 | 50.4 | 25.0 | 18.7 |
| Gap far | 209 | 28.4 | 24.4-32.2 | 17.0 | 38.8 | 28.7 | 13.9 | 10.0 |

## 4. Fingertip and pointing-ray error by scene (test)

| Subset | n | Mean px | Mean 95% CI | Median px | P90 px | <5 px % | <10 px % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 3.90 | 3.75-4.08 | 3.36 | 5.94 | 81.2 | 97.9 |
| All pointing | 2604 | 3.90 | 3.75-4.08 | 3.36 | 5.94 | 81.2 | 97.9 |

Pixels at the 160x160 model input. Ray error for "Pointing, no target" is measured against the hand-centre-to-fingertip direction.

| Subset | n | Mean deg | Mean 95% CI | Median deg | P90 deg | <15 deg % | <30 deg % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 33.71 | 32.40-35.11 | 20.82 | 84.33 | 39.6 | 62.4 |
| All pointing | 2604 | 33.71 | 32.40-35.11 | 20.82 | 84.33 | 39.6 | 62.4 |

| Gap | n | Fingertip mean px | Ray mean deg | Missed pointing |
|---|---:|---:|---:|---:|
| on-object | 1796 | 3.85 | 36.9 | 5 |
| near | 599 | 3.84 | 23.9 | 0 |
| far | 209 | 4.52 | 34.6 | 2 |

## 5. Val-calibrated gate thresholds (test)

Highest threshold per gate with no missed positives on the 1524 validation images: hand 0.30, pointing 0.50, target 0.50.

3029 images. Scene accuracy 97.23% (95% CI 96.6-97.8), macro-F1 93.74%, balanced accuracy 90.22%. Gate thresholds hand/pointing/target = 0.30/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 99.7 | 80.5 | 76.4-84.0 | 89.1 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 96.9 | 100.0 | 99.8-100.0 | 98.4 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 342 | 0 | 0 | 83 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 1 | 0 | 0 | 2603 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.30 | 97.2 | 100.0 | 1 | 83 | 0.9988 | 0.9998 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 6. Robustness to other people's hands (test with 1-2 pasted distractor hands)

3029 images. Scene accuracy 98.38% (95% CI 97.9-98.8), macro-F1 96.54%, balanced accuracy 95.12%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 97.7 | 90.6 | 87.4-93.0 | 94.0 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.5 | 99.7 | 99.3-99.8 | 99.1 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 385 | 0 | 0 | 40 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 9 | 0 | 0 | 2595 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.4 | 99.7 | 9 | 40 | 0.9986 | 0.9998 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 73.1 | 72.5-73.7 | 76.3 | 97.6 | 91.9 | 69.0 | 55.0 |
| All hand-present | 2604 | 73.1 | 72.5-73.7 | 76.3 | 97.6 | 91.9 | 69.0 | 55.0 |

| Ownership | Value |
|---|---:|
| Hand images with distractors | 2377 |
| Ownership accuracy % (predicted hand nearer the wearer) | 99.87 |
| Ownership accuracy 95% CI | 99.6-100.0 |
| Predicted hand centre inside a distractor % | 0.17 |
| Pointing images with distractors | 2377 |
| Ownership accuracy on pointing images % | 99.87 |
| No-hand images with distractors | 425 |
| Still classified "No hand" % | 90.59 |

## 7. Headline comparison

| Metric | Test (default thresholds) | Test (val-calibrated thresholds) | Test + distractor hands |
|---|---:|---:|---:|
| Scene accuracy % | 98.6 | 97.2 | 98.4 |
| Scene macro-F1 % | 97.1 | 93.7 | 96.5 |
| Missed hand / pointing / target | 7/0/0 | 1/0/0 | 9/0/0 |
| False alarms hand / pointing / target | 34/0/0 | 83/0/0 | 40/0/0 |
| Hand IoU mean % (all hand-present) | 73.3 | 73.3 | 73.1 |
| Hand IoU>=0.5 % | 91.9 | 91.9 | 91.9 |
| Target IoU mean % | 52.8 | 52.8 | 53.2 |
| Target IoU>=0.5 % | 60.0 | 60.0 | 60.1 |
| Fingertip error mean px | 3.90 | 3.90 | 4.10 |

## 8. Efficiency

| Property | Value |
|---|---:|
| Parameters | 722,104 |
|   backbone / heads | 143,136 / 578,968 |
| Compute per image | 0.231 GFLOPs |
| Input | 160 x 160 RGB |
| Size FP32 / FP16 / INT8 | 3064 KB / 1600 KB / 2733 KB |
| Latency CPU, 1 thread, batch 1 | 28.16 ms median, 29.19 ms p90 (36 FPS) |
| Latency CPU, 4 threads, batch 1 | 28.57 ms median, 29.79 ms p90 (35 FPS) |
| Latency GPU (Tesla T4), batch 1 | 16.43 ms median, 16.86 ms p90 (61 FPS) |

CPU: x86_64. Latency is PyTorch eager mode, not an on-device runtime.
