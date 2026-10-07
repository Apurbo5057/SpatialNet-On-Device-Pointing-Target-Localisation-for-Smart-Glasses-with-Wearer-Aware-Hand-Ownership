# SpatialNet evaluation

## 1. Scene classification and gates (test)

3029 images. Scene accuracy 98.58% (95% CI 98.1-98.9), macro-F1 97.05%, balanced accuracy 96.91%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 95.3 | 94.6 | 92.0-96.4 | 94.9 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 99.1 | 99.2 | 98.8-99.5 | 99.2 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 402 | 0 | 0 | 23 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 20 | 0 | 0 | 2584 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.6 | 99.2 | 20 | 23 | 0.9983 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 2. Hand localisation by scene (test)

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 73.4 | 72.8-74.0 | 77.0 | 97.4 | 92.0 | 69.3 | 56.2 |
| All hand-present | 2604 | 73.4 | 72.8-74.0 | 77.0 | 97.4 | 92.0 | 69.3 | 56.2 |

## 3. Target localisation (test, pointing at target)

Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of the image, far: beyond).

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2604 | 52.4 | 51.4-53.4 | 57.3 | 77.5 | 59.0 | 31.0 | 22.8 |
| Gap on-object | 1796 | 56.9 | 55.8-58.1 | 61.7 | 84.0 | 66.2 | 36.0 | 26.7 |
| Gap near | 599 | 47.3 | 45.1-49.3 | 49.1 | 71.5 | 49.1 | 22.7 | 16.4 |
| Gap far | 209 | 28.1 | 24.5-31.7 | 17.1 | 38.8 | 25.8 | 12.0 | 7.2 |

## 4. Fingertip and pointing-ray error by scene (test)

| Subset | n | Mean px | Mean 95% CI | Median px | P90 px | <5 px % | <10 px % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 3.81 | 3.67-3.98 | 3.27 | 5.78 | 82.7 | 97.8 |
| All pointing | 2604 | 3.81 | 3.67-3.98 | 3.27 | 5.78 | 82.7 | 97.8 |

Pixels at the 160x160 model input. Ray error for "Pointing, no target" is measured against the hand-centre-to-fingertip direction.

| Subset | n | Mean deg | Mean 95% CI | Median deg | P90 deg | <15 deg % | <30 deg % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 33.02 | 31.55-34.45 | 19.83 | 82.41 | 41.9 | 62.5 |
| All pointing | 2604 | 33.02 | 31.55-34.45 | 19.83 | 82.41 | 41.9 | 62.5 |

| Gap | n | Fingertip mean px | Ray mean deg | Missed pointing |
|---|---:|---:|---:|---:|
| on-object | 1796 | 3.74 | 37.0 | 15 |
| near | 599 | 3.73 | 22.3 | 1 |
| far | 209 | 4.66 | 29.6 | 4 |

## 5. Val-calibrated gate thresholds (test)

Highest threshold per gate with no missed positives on the 1524 validation images: hand 0.20, pointing 0.50, target 0.50.

3029 images. Scene accuracy 97.39% (95% CI 96.8-97.9), macro-F1 94.18%, balanced accuracy 91.10%. Gate thresholds hand/pointing/target = 0.20/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 98.9 | 82.4 | 78.4-85.7 | 89.9 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 97.2 | 99.8 | 99.6-99.9 | 98.5 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 350 | 0 | 0 | 75 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 4 | 0 | 0 | 2600 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.20 | 97.4 | 99.8 | 4 | 75 | 0.9983 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 6. Robustness to other people's hands (test with 1-2 pasted distractor hands)

3029 images. Scene accuracy 98.61% (95% CI 98.1-99.0), macro-F1 97.13%, balanced accuracy 97.13%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 95.1 | 95.1 | 92.6-96.7 | 95.1 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 99.2 | 99.2 | 98.8-99.5 | 99.2 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 404 | 0 | 0 | 21 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 21 | 0 | 0 | 2583 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.6 | 99.2 | 21 | 21 | 0.9974 | 0.9996 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 73.2 | 72.6-73.8 | 76.9 | 97.2 | 91.6 | 68.9 | 55.3 |
| All hand-present | 2604 | 73.2 | 72.6-73.8 | 76.9 | 97.2 | 91.6 | 68.9 | 55.3 |

| Ownership | Value |
|---|---:|
| Hand images with distractors | 2377 |
| Ownership accuracy % (predicted hand nearer the wearer) | 99.66 |
| Ownership accuracy 95% CI | 99.3-99.8 |
| Predicted hand centre inside a distractor % | 0.17 |
| Pointing images with distractors | 2377 |
| Ownership accuracy on pointing images % | 99.66 |
| No-hand images with distractors | 425 |
| Still classified "No hand" % | 95.06 |

## 7. Headline comparison

| Metric | Test (default thresholds) | Test (val-calibrated thresholds) | Test + distractor hands |
|---|---:|---:|---:|
| Scene accuracy % | 98.6 | 97.4 | 98.6 |
| Scene macro-F1 % | 97.0 | 94.2 | 97.1 |
| Missed hand / pointing / target | 20/0/0 | 4/0/0 | 21/0/0 |
| False alarms hand / pointing / target | 23/0/0 | 75/0/0 | 21/0/0 |
| Hand IoU mean % (all hand-present) | 73.4 | 73.4 | 73.2 |
| Hand IoU>=0.5 % | 92.0 | 92.0 | 91.6 |
| Target IoU mean % | 52.4 | 52.4 | 53.3 |
| Target IoU>=0.5 % | 59.0 | 59.0 | 60.5 |
| Fingertip error mean px | 3.81 | 3.81 | 4.02 |

## 8. Efficiency

| Property | Value |
|---|---:|
| Parameters | 722,104 |
|   backbone / heads | 143,136 / 578,968 |
| Compute per image | 0.231 GFLOPs |
| Input | 160 x 160 RGB |
| Size FP32 / FP16 / INT8 | 3064 KB / 1600 KB / 2733 KB |
| Latency CPU, 1 thread, batch 1 | 27.95 ms median, 29.20 ms p90 (36 FPS) |
| Latency CPU, 4 threads, batch 1 | 27.14 ms median, 30.11 ms p90 (37 FPS) |
| Latency GPU (Tesla T4), batch 1 | 16.77 ms median, 17.76 ms p90 (60 FPS) |

CPU: x86_64. Latency is PyTorch eager mode, not an on-device runtime.
