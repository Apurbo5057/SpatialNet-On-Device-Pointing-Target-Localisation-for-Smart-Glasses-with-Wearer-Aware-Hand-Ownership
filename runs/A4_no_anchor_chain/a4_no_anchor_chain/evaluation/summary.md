# SpatialNet evaluation

## 1. Scene classification and gates (test)

3029 images. Scene accuracy 98.38% (95% CI 97.9-98.8), macro-F1 96.62%, balanced accuracy 96.30%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 95.0 | 93.4 | 90.6-95.4 | 94.2 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.9 | 99.2 | 98.8-99.5 | 99.1 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 397 | 0 | 0 | 28 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 21 | 0 | 0 | 2583 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.4 | 99.2 | 21 | 28 | 0.9969 | 0.9994 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 2. Hand localisation by scene (test)

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 71.9 | 71.2-72.4 | 75.5 | 97.7 | 91.0 | 64.5 | 51.5 |
| All hand-present | 2604 | 71.9 | 71.2-72.4 | 75.5 | 97.7 | 91.0 | 64.5 | 51.5 |

## 3. Target localisation (test, pointing at target)

Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of the image, far: beyond).

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2604 | 51.9 | 50.9-52.8 | 56.5 | 77.6 | 58.6 | 29.0 | 19.3 |
| Gap on-object | 1796 | 56.7 | 55.6-57.8 | 61.6 | 84.5 | 66.1 | 34.1 | 23.2 |
| Gap near | 599 | 45.6 | 43.7-47.4 | 47.8 | 69.8 | 47.6 | 20.4 | 12.2 |
| Gap far | 209 | 28.5 | 24.9-32.1 | 20.4 | 40.2 | 25.8 | 10.5 | 6.7 |

## 4. Fingertip and pointing-ray error by scene (test)

| Subset | n | Mean px | Mean 95% CI | Median px | P90 px | <5 px % | <10 px % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 3.89 | 3.75-4.05 | 3.27 | 5.92 | 82.3 | 97.2 |
| All pointing | 2604 | 3.89 | 3.75-4.05 | 3.27 | 5.92 | 82.3 | 97.2 |

Pixels at the 160x160 model input. Ray error for "Pointing, no target" is measured against the hand-centre-to-fingertip direction.

| Subset | n | Mean deg | Mean 95% CI | Median deg | P90 deg | <15 deg % | <30 deg % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointing, no target | 0 | - | - | - | - | - | - |
| Pointing at target | 2604 | 35.65 | 34.21-37.06 | 22.98 | 88.40 | 36.4 | 58.8 |
| All pointing | 2604 | 35.65 | 34.21-37.06 | 22.98 | 88.40 | 36.4 | 58.8 |

| Gap | n | Fingertip mean px | Ray mean deg | Missed pointing |
|---|---:|---:|---:|---:|
| on-object | 1796 | 3.82 | 40.5 | 16 |
| near | 599 | 3.90 | 23.7 | 3 |
| far | 209 | 4.45 | 28.0 | 2 |

## 5. Val-calibrated gate thresholds (test)

Highest threshold per gate with no missed positives on the 1524 validation images: hand 0.18, pointing 0.50, target 0.50.

3029 images. Scene accuracy 94.42% (95% CI 93.5-95.2), macro-F1 86.06%, balanced accuracy 80.22%. Gate thresholds hand/pointing/target = 0.18/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 99.6 | 60.5 | 55.7-65.0 | 75.3 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 93.9 | 100.0 | 99.8-100.0 | 96.9 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 257 | 0 | 0 | 168 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 1 | 0 | 0 | 2603 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.18 | 94.4 | 100.0 | 1 | 168 | 0.9969 | 0.9994 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

## 6. Robustness to other people's hands (test with 1-2 pasted distractor hands)

3029 images. Scene accuracy 97.99% (95% CI 97.4-98.4), macro-F1 95.79%, balanced accuracy 95.38%. Gate thresholds hand/pointing/target = 0.50/0.50/0.50.

| Scene (H P T) | Support | Precision % | Recall % | Recall 95% CI | F1 % |
|---|---:|---:|---:|---:|---:|
| No hand | 425 | 93.8 | 91.8 | 88.8-94.0 | 92.7 |
| Non-pointing | 0 | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - |
| Pointing at target | 2604 | 98.7 | 99.0 | 98.5-99.3 | 98.8 |

| Ground truth \ Predicted | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|
| No hand | 390 | 0 | 0 | 35 |
| Non-pointing | 0 | 0 | 0 | 0 |
| Pointing, no target | 0 | 0 | 0 | 0 |
| Pointing at target | 26 | 0 | 0 | 2578 |

| Gate | Images | Positives | Threshold | Accuracy % | Recall % | Missed | False alarms | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand present | 3029 | 2604 | 0.50 | 98.0 | 99.0 | 26 | 35 | 0.9961 | 0.9992 |
| Pointing | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |
| Target present | 2604 | 2604 | 0.50 | 100.0 | 100.0 | 0 | 0 | - | 1.0000 |

| Subset | n | Mean IoU % | Mean 95% CI | Median % | IoU>=0.3 % | IoU>=0.5 % | IoU>=0.7 % | IoU>=0.75 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-pointing | 0 | - | - | - | - | - | - | - |
| Pointing, no target | 0 | - | - | - | - | - | - | - |
| Pointing at target | 2604 | 71.6 | 71.0-72.2 | 75.2 | 97.5 | 90.7 | 64.1 | 50.7 |
| All hand-present | 2604 | 71.6 | 71.0-72.2 | 75.2 | 97.5 | 90.7 | 64.1 | 50.7 |

| Ownership | Value |
|---|---:|
| Hand images with distractors | 2377 |
| Ownership accuracy % (predicted hand nearer the wearer) | 99.79 |
| Ownership accuracy 95% CI | 99.5-99.9 |
| Predicted hand centre inside a distractor % | 0.13 |
| Pointing images with distractors | 2377 |
| Ownership accuracy on pointing images % | 99.79 |
| No-hand images with distractors | 425 |
| Still classified "No hand" % | 91.76 |

## 7. Headline comparison

| Metric | Test (default thresholds) | Test (val-calibrated thresholds) | Test + distractor hands |
|---|---:|---:|---:|
| Scene accuracy % | 98.4 | 94.4 | 98.0 |
| Scene macro-F1 % | 96.6 | 86.1 | 95.8 |
| Missed hand / pointing / target | 21/0/0 | 1/0/0 | 26/0/0 |
| False alarms hand / pointing / target | 28/0/0 | 168/0/0 | 35/0/0 |
| Hand IoU mean % (all hand-present) | 71.9 | 71.9 | 71.6 |
| Hand IoU>=0.5 % | 91.0 | 91.0 | 90.7 |
| Target IoU mean % | 51.9 | 51.9 | 52.7 |
| Target IoU>=0.5 % | 58.6 | 58.6 | 59.6 |
| Fingertip error mean px | 3.89 | 3.89 | 4.06 |

## 8. Efficiency

| Property | Value |
|---|---:|
| Parameters | 709,191 |
|   backbone / heads | 143,136 / 566,055 |
| Compute per image | 0.221 GFLOPs |
| Input | 160 x 160 RGB |
| Size FP32 / FP16 / INT8 | 3010 KB / 1572 KB / 2680 KB |
| Latency CPU, 1 thread, batch 1 | 26.30 ms median, 27.87 ms p90 (38 FPS) |
| Latency CPU, 4 threads, batch 1 | 26.26 ms median, 27.45 ms p90 (38 FPS) |
| Latency GPU (Tesla T4), batch 1 | 16.05 ms median, 17.10 ms p90 (62 FPS) |

CPU: x86_64. Latency is PyTorch eager mode, not an on-device runtime.
