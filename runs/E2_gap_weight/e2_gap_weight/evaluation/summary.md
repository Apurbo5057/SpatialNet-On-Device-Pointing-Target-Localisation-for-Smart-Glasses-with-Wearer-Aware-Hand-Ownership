# SpatialNet evaluation

## 1. Scene classification and gates (test)

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
| Hand present | 3029 | 2604 | 0.50 | 98.7 | 99.4 | 16 | 22 | 0.9973 | 0.9995 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 2. Hand localisation by scene (test)

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 72.4 | 71.9-73.0 | 75.9 | 98.0 | 91.6 | 66.1 | 52.8 |
| All hand-present | 2604 | 72.4 | 71.9-73.0 | 75.9 | 98.0 | 91.6 | 66.1 | 52.8 |

## 3. Target localisation (test, pointing at target)

Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of the image, far: beyond).

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2604 | 50.9 | 50.0-51.9 | 55.8 | 76.0 | 57.1 | 28.1 | 19.6 |
| Gap on-object | 1796 | 54.3 | 53.1-55.4 | 59.2 | 81.3 | 62.2 | 31.7 | 22.4 |
| Gap near | 599 | 48.1 | 46.2-50.0 | 50.7 | 72.0 | 51.3 | 23.9 | 16.0 |
| Gap far | 209 | 30.4 | 26.8-33.9 | 21.6 | 42.6 | 29.2 | 9.1 | 5.7 |

## 4. Fingertip and pointing-ray error by scene (test)

| Subset | n | Mean px | Mean 95% CI | Median px | P90 px | <5 px % | <10 px % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 3.91 | 3.76-4.08 | 3.38 | 5.93 | 81.4 | 97.7 |
| All pointing | 2604 | 3.91 | 3.76-4.08 | 3.38 | 5.93 | 81.4 | 97.7 |

Pixels at the 160x160 model input. Ray error for "Pointing, no target" is measured against the hand-centre-to-fingertip direction.

| Subset | n | Mean deg | Mean 95% CI | Median deg | P90 deg | <15 deg % | <30 deg % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 34.69 | 33.27-36.11 | 22.22 | 85.71 | 38.5 | 61.1 |
| All pointing | 2604 | 34.69 | 33.27-36.11 | 22.22 | 85.71 | 38.5 | 61.1 |

| Gap | n | Fingertip mean px | Ray mean deg | Missed pointing |
|---|---:|---:|---:|---:|
| on-object | 1796 | 3.82 | 39.6 | 10 |
| near | 599 | 3.92 | 22.1 | 3 |
| far | 209 | 4.76 | 28.2 | 3 |

## 5. Val-calibrated gate thresholds (test)

Highest threshold per gate with no missed positives on the 1524 validation images: hand 0.20, pointing 0.50, target 0.50.

3029 images. Scene accuracy 95.05% (95% CI 94.2-95.8), macro-F1 87.92%, balanced accuracy 82.45%. Gate thresholds hand/pointing/target = 0.20/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 99.6 | 64.9 | 60.3-69.3 | 78.6 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 94.6 | 100.0 | 99.8-100.0 | 97.2 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 276 | 0 | 0 | 149 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 1 | 0 | 0 | 2603 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.20 | 95.0 | 100.0 | 1 | 149 | 0.9973 | 0.9995 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 6. Robustness to other people's hands (test with 1-2 pasted distractor hands)

3029 images. Scene accuracy 98.61% (95% CI 98.1-99.0), macro-F1 97.11%, balanced accuracy 96.93%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 95.5 | 94.6 | 92.0-96.4 | 95.0 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 99.1 | 99.3 | 98.9-99.5 | 99.2 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 402 | 0 | 0 | 23 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 19 | 0 | 0 | 2585 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.6 | 99.3 | 19 | 23 | 0.9962 | 0.9993 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 72.2 | 71.6-72.7 | 75.9 | 97.6 | 91.3 | 66.1 | 53.1 |
| All hand-present | 2604 | 72.2 | 71.6-72.7 | 75.9 | 97.6 | 91.3 | 66.1 | 53.1 |

| Ownership | Value |
|---|---:|
| Hand images with distractors | 2377 |
| Ownership accuracy % (predicted hand nearer the wearer) | 99.71 |
| Ownership accuracy 95% CI | 99.4-99.9 |
| Predicted hand centre inside a distractor % | 0.25 |
| Pointing images with distractors | 2377 |
| Ownership accuracy on pointing images % | 99.71 |
| No-hand images with distractors | 425 |
| Still classified "No hand" % | 94.59 |

## 7. Headline comparison

| Metric | Test (default thresholds) | Test (val-calibrated thresholds) | Test + distractor hands |
|---|---:|---:|---:|
| Scene accuracy % | 98.7 | 95.0 | 98.6 |
| Scene macro-F1 % | 97.4 | 87.9 | 97.1 |
| Missed hand / pointing / target | 16/0/0 | 1/0/0 | 19/0/0 |
| False alarms hand / pointing / target | 22/0/0 | 149/0/0 | 23/0/0 |
| Hand IoU mean % (all hand-present) | 72.4 | 72.4 | 72.2 |
| Hand IoU>=0.5 % | 91.6 | 91.6 | 91.3 |
| Target IoU mean % | 50.9 | 50.9 | 51.3 |
| Target IoU>=0.5 % | 57.1 | 57.1 | 57.9 |
| Fingertip error mean px | 3.91 | 3.91 | 4.09 |

## 8. Efficiency

| Property | Value |
|---|---:|
| Parameters | 722,104 |
|   backbone / heads | 143,136 / 578,968 |
| Compute per image | 0.231 GFLOPs |
| Input | 160 x 160 RGB |
| Size FP32 / FP16 / INT8 | 3064 KB / 1600 KB / 2733 KB |
| Latency CPU, 1 thread, batch 1 | 30.69 ms median, 31.65 ms p90 (33 FPS) |
| Latency CPU, 4 threads, batch 1 | 30.44 ms median, 35.03 ms p90 (33 FPS) |
| Latency GPU (Tesla T4), batch 1 | 18.57 ms median, 18.98 ms p90 (54 FPS) |

CPU: x86_64. Latency is PyTorch eager mode, not an on-device runtime.
