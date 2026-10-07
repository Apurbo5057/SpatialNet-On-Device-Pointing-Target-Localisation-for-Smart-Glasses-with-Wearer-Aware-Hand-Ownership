# SpatialNet evaluation

## 1. Scene classification and gates (test)

3029 images. Scene accuracy 98.91% (95% CI 98.5-99.2), macro-F1 97.73%, balanced accuracy 97.50%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 96.7 | 95.5 | 93.1-97.1 | 96.1 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 99.3 | 99.5 | 99.1-99.7 | 99.4 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 406 | 0 | 0 | 19 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 14 | 0 | 0 | 2590 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.9 | 99.5 | 14 | 19 | 0.9987 | 0.9998 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 2. Hand localisation by scene (test)

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 74.9 | 74.3-75.4 | 78.8 | 97.6 | 93.0 | 73.6 | 61.9 |
| All hand-present | 2604 | 74.9 | 74.3-75.4 | 78.8 | 97.6 | 93.0 | 73.6 | 61.9 |

## 3. Target localisation (test, pointing at target)

Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of the image, far: beyond).

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2604 | 54.8 | 53.7-55.7 | 60.7 | 78.8 | 62.4 | 37.1 | 27.8 |
| Gap on-object | 1796 | 58.9 | 57.7-60.0 | 65.3 | 84.8 | 68.3 | 42.0 | 32.1 |
| Gap near | 599 | 49.4 | 47.1-51.5 | 54.1 | 71.8 | 53.8 | 28.0 | 20.7 |
| Gap far | 209 | 34.5 | 30.3-38.1 | 28.5 | 46.9 | 35.9 | 20.1 | 12.0 |

## 4. Fingertip and pointing-ray error by scene (test)

| Subset | n | Mean px | Mean 95% CI | Median px | P90 px | <5 px % | <10 px % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 4.06 | 3.89-4.23 | 3.44 | 6.08 | 79.6 | 97.8 |
| All pointing | 2604 | 4.06 | 3.89-4.23 | 3.44 | 6.08 | 79.6 | 97.8 |

Pixels at the 224x224 model input. Ray error for "Pointing, no target" is measured against the hand-centre-to-fingertip direction.

| Subset | n | Mean deg | Mean 95% CI | Median deg | P90 deg | <15 deg % | <30 deg % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 31.45 | 30.06-32.81 | 19.51 | 76.64 | 42.3 | 64.1 |
| All pointing | 2604 | 31.45 | 30.06-32.81 | 19.51 | 76.64 | 42.3 | 64.1 |

| Gap | n | Fingertip mean px | Ray mean deg | Missed pointing |
|---|---:|---:|---:|---:|
| on-object | 1796 | 4.01 | 35.6 | 11 |
| near | 599 | 3.91 | 20.3 | 1 |
| far | 209 | 4.87 | 27.8 | 2 |

## 5. Val-calibrated gate thresholds (test)

Highest threshold per gate with no missed positives on the 1524 validation images: hand 0.18, pointing 0.50, target 0.50.

3029 images. Scene accuracy 96.47% (95% CI 95.7-97.1), macro-F1 91.85%, balanced accuracy 87.71%. Gate thresholds hand/pointing/target = 0.18/0.50/0.50.

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
| Hand present | 3029 | 2604 | 0.18 | 96.5 | 99.9 | 3 | 104 | 0.9987 | 0.9998 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 6. Robustness to other people's hands (test with 1-2 pasted distractor hands)

3029 images. Scene accuracy 98.84% (95% CI 98.4-99.2), macro-F1 97.59%, balanced accuracy 97.36%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 96.4 | 95.3 | 92.8-96.9 | 95.9 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 99.2 | 99.4 | 99.1-99.7 | 99.3 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 405 | 0 | 0 | 20 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 15 | 0 | 0 | 2589 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.8 | 99.4 | 15 | 20 | 0.9984 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 74.5 | 73.9-75.1 | 78.4 | 97.3 | 92.3 | 73.3 | 60.9 |
| All hand-present | 2604 | 74.5 | 73.9-75.1 | 78.4 | 97.3 | 92.3 | 73.3 | 60.9 |

| Ownership | Value |
|---|---:|
| Hand images with distractors | 2377 |
| Ownership accuracy % (predicted hand nearer the wearer) | 99.79 |
| Ownership accuracy 95% CI | 99.5-99.9 |
| Predicted hand centre inside a distractor % | 0.21 |
| Pointing images with distractors | 2377 |
| Ownership accuracy on pointing images % | 99.79 |
| No-hand images with distractors | 425 |
| Still classified "No hand" % | 95.29 |

## 7. Headline comparison

| Metric | Test (default thresholds) | Test (val-calibrated thresholds) | Test + distractor hands |
|---|---:|---:|---:|
| Scene accuracy % | 98.9 | 96.5 | 98.8 |
| Scene macro-F1 % | 97.7 | 91.8 | 97.6 |
| Missed hand / pointing / target | 14/0/0 | 3/0/0 | 15/0/0 |
| False alarms hand / pointing / target | 19/0/0 | 104/0/0 | 20/0/0 |
| Hand IoU mean % (all hand-present) | 74.9 | 74.9 | 74.5 |
| Hand IoU>=0.5 % | 93.0 | 93.0 | 92.3 |
| Target IoU mean % | 54.8 | 54.8 | 55.9 |
| Target IoU>=0.5 % | 62.4 | 62.4 | 63.9 |
| Fingertip error mean px | 4.06 | 4.06 | 4.30 |

## 8. Efficiency

| Property | Value |
|---|---:|
| Parameters | 722,104 |
|   backbone / heads | 143,136 / 578,968 |
| Compute per image | 0.453 GFLOPs |
| Input | 224 x 224 RGB |
| Size FP32 / FP16 / INT8 | 3064 KB / 1600 KB / 2733 KB |
| Latency CPU, 1 thread, batch 1 | 37.18 ms median, 38.93 ms p90 (27 FPS) |
| Latency CPU, 4 threads, batch 1 | 33.64 ms median, 35.36 ms p90 (30 FPS) |
| Latency GPU (Tesla T4), batch 1 | 16.45 ms median, 17.56 ms p90 (61 FPS) |

CPU: x86_64. Latency is PyTorch eager mode, not an on-device runtime.
