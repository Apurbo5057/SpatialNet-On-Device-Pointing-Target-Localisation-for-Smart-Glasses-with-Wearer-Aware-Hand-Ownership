# SpatialNet evaluation

## 1. Scene classification and gates (test)

3029 images. Scene accuracy 98.65% (95% CI 98.2-99.0), macro-F1 97.16%, balanced accuracy 96.65%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 96.4 | 93.9 | 91.2-95.8 | 95.1 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 99.0 | 99.4 | 99.1-99.7 | 99.2 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 399 | 0 | 0 | 26 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 15 | 0 | 0 | 2589 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.6 | 99.4 | 15 | 26 | 0.9980 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 2. Hand localisation by scene (test)

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 73.7 | 73.1-74.3 | 77.6 | 97.3 | 91.7 | 70.3 | 57.6 |
| All hand-present | 2604 | 73.7 | 73.1-74.3 | 77.6 | 97.3 | 91.7 | 70.3 | 57.6 |

## 3. Target localisation (test, pointing at target)

Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of the image, far: beyond).

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2604 | 53.5 | 52.5-54.5 | 59.0 | 78.1 | 59.6 | 33.7 | 25.0 |
| Gap on-object | 1796 | 57.9 | 56.9-59.1 | 62.8 | 84.6 | 66.1 | 38.3 | 29.1 |
| Gap near | 599 | 48.3 | 46.1-50.5 | 50.5 | 71.6 | 50.4 | 25.9 | 18.5 |
| Gap far | 209 | 30.4 | 26.5-34.0 | 19.1 | 41.1 | 30.6 | 16.7 | 9.1 |

## 4. Fingertip and pointing-ray error by scene (test)

| Subset | n | Mean px | Mean 95% CI | Median px | P90 px | <5 px % | <10 px % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 3.74 | 3.58-3.92 | 3.20 | 5.65 | 84.5 | 97.8 |
| All pointing | 2604 | 3.74 | 3.58-3.92 | 3.20 | 5.65 | 84.5 | 97.8 |

Pixels at the 160x160 model input. Ray error for "Pointing, no target" is measured against the hand-centre-to-fingertip direction.

| Subset | n | Mean deg | Mean 95% CI | Median deg | P90 deg | <15 deg % | <30 deg % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 30.89 | 29.53-32.24 | 17.17 | 82.43 | 46.1 | 65.9 |
| All pointing | 2604 | 30.89 | 29.53-32.24 | 17.17 | 82.43 | 46.1 | 65.9 |

| Gap | n | Fingertip mean px | Ray mean deg | Missed pointing |
|---|---:|---:|---:|---:|
| on-object | 1796 | 3.71 | 33.7 | 11 |
| near | 599 | 3.74 | 22.0 | 1 |
| far | 209 | 3.99 | 32.0 | 3 |

## 5. Val-calibrated gate thresholds (test)

Highest threshold per gate with no missed positives on the 1524 validation images: hand 0.12, pointing 0.50, target 0.50.

3029 images. Scene accuracy 94.75% (95% CI 93.9-95.5), macro-F1 87.11%, balanced accuracy 81.59%. Gate thresholds hand/pointing/target = 0.12/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 98.9 | 63.3 | 58.6-67.7 | 77.2 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 94.3 | 99.9 | 99.7-100.0 | 97.0 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 269 | 0 | 0 | 156 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 3 | 0 | 0 | 2601 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.12 | 94.8 | 99.9 | 3 | 156 | 0.9980 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 6. Robustness to other people's hands (test with 1-2 pasted distractor hands)

3029 images. Scene accuracy 98.42% (95% CI 97.9-98.8), macro-F1 96.68%, balanced accuracy 96.12%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 95.6 | 92.9 | 90.1-95.0 | 94.3 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.9 | 99.3 | 98.9-99.6 | 99.1 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 395 | 0 | 0 | 30 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 18 | 0 | 0 | 2586 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.4 | 99.3 | 18 | 30 | 0.9978 | 0.9997 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 73.6 | 73.0-74.2 | 77.6 | 97.2 | 91.2 | 69.7 | 57.5 |
| All hand-present | 2604 | 73.6 | 73.0-74.2 | 77.6 | 97.2 | 91.2 | 69.7 | 57.5 |

| Ownership | Value |
|---|---:|
| Hand images with distractors | 2377 |
| Ownership accuracy % (predicted hand nearer the wearer) | 99.87 |
| Ownership accuracy 95% CI | 99.6-100.0 |
| Predicted hand centre inside a distractor % | 0.13 |
| Pointing images with distractors | 2377 |
| Ownership accuracy on pointing images % | 99.87 |
| No-hand images with distractors | 425 |
| Still classified "No hand" % | 92.94 |

## 7. Headline comparison

| Metric | Test (default thresholds) | Test (val-calibrated thresholds) | Test + distractor hands |
|---|---:|---:|---:|
| Scene accuracy % | 98.6 | 94.8 | 98.4 |
| Scene macro-F1 % | 97.2 | 87.1 | 96.7 |
| Missed hand / pointing / target | 15/0/0 | 3/0/0 | 18/0/0 |
| False alarms hand / pointing / target | 26/0/0 | 156/0/0 | 30/0/0 |
| Hand IoU mean % (all hand-present) | 73.7 | 73.7 | 73.6 |
| Hand IoU>=0.5 % | 91.7 | 91.7 | 91.2 |
| Target IoU mean % | 53.5 | 53.5 | 54.1 |
| Target IoU>=0.5 % | 59.6 | 59.6 | 61.1 |
| Fingertip error mean px | 3.74 | 3.74 | 3.88 |

## 8. Efficiency

| Property | Value |
|---|---:|
| Parameters | 708,999 |
|   backbone / heads | 143,136 / 565,863 |
| Compute per image | 0.221 GFLOPs |
| Input | 160 x 160 RGB |
| Size FP32 / FP16 / INT8 | 3010 KB / 1572 KB / 2679 KB |
| Latency CPU, 1 thread, batch 1 | 28.35 ms median, 29.96 ms p90 (35 FPS) |
| Latency CPU, 4 threads, batch 1 | 29.24 ms median, 31.47 ms p90 (34 FPS) |
| Latency GPU (Tesla T4), batch 1 | 17.08 ms median, 18.17 ms p90 (59 FPS) |

CPU: x86_64. Latency is PyTorch eager mode, not an on-device runtime.
