# SpatialNet on the glasses: measurement request

**From:** Apurbo
**To:** Farhan Tanvir
**Reply by:** 19-10-2026. Our paper deadline is 1 November.

## What this is for

SpatialNet is the small pointing model you already run on the engineering glasses:

- **Input:** one camera frame, resized to 160 × 160 RGB.
- **Model:** a single 0.72M-parameter CNN.
- **Outputs:** the wearer's hand box, the fingertip, the pointing direction, the box of the object pointed at, and three presence probabilities.

We are submitting a paper about it, and reviewers will check the on-device numbers closely. Our earlier measurements were too thin to defend:

- **Latency:** a range of 60–80 ms per frame, without saying what was timed or over how many frames.
- **Memory:** about 100 KB of extra RAM.
- **Battery:** one one-minute reading per mode: 9% for continuous inference and 16% for streaming frames to the phone. A 9% drop per minute does not fit the device's rated 8 hours of moderate use, so readers will doubt it unless we have repeated, longer runs with an idle baseline.

**Confidentiality.** The paper does not name the device, the chip, the runtime or Samsung. It describes the unit as "an engineering unit of a pair of smart glasses" and reports only the summary numbers listed at the end of this note. We ask for the device details in section 0 only so that we describe the measurements correctly; we will not publish them.

## Priorities

| ID | Measurement | Priority | Time on the glasses |
|---|---|---|---|
| A | Deployment check: the on-device model gives the same outputs as ours | Must | 15 min |
| B | Latency, broken down per stage, over many frames | Must | 30 min |
| C | Memory | Must | 10 min |
| D | Power in four modes, repeated | Must | about 2 h, including recharging |
| E | Two small detector models on the glasses, for comparison | Nice to have | 30 min |
| F | Two extra power modes | Nice to have | about 1 h |
| G | Real frames: the wearer points at everyday objects | Must | about 1–1.5 h |
| H | Extras: non-pointing hands, short clips of natural use, a demo video | Nice to have | about 45 min |

If time is short, please do A–D and G first; E, F and H are optional.

## 0. Setup: report once, keep fixed for all runs

Please tell us:

1. **Software:** the device build or firmware version; the inference runtime and its version; the compute unit that runs the model (CPU with core type and thread count, GPU, DSP/NPU, or the camera NPU); and the precision (FP16, INT8, ...).
2. **Camera:** the capture resolution and frame rate.
3. **Settings:** any power or performance mode, or fixed CPU governor; the ambient temperature; and the Wi-Fi and Bluetooth state in each mode.
4. **Preprocessing, which matters for accuracy.** The model expects the following. Please confirm or correct each point:
   - the **whole frame squashed to 160 × 160**: no crop, no letterbox, aspect ratio not kept;
   - bilinear resizing;
   - **RGB** channel order;
   - values scaled to [0, 1], then normalised with mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).

The glasses must be **off the charger** in every run.

## A. Deployment check (must, 15 minutes)

This shows that the model on the glasses computes the same thing as our reference model.

1. Point at objects in any scene and capture **50 consecutive frames**.
2. For each frame, save:
   - the exact model input after preprocessing: a 3 × 160 × 160 array (`.npy`), or the 160 × 160 resized image before normalisation (`.png`);
   - the raw model outputs: hand box, fingertip, direction (sin, cos), target box, and the three presence values, as logits or probabilities. Please say which.
3. We run the same inputs through our PyTorch model and report the largest difference. Box coordinates should agree to about 0.01 of the image size.

**Send back:** `A_deployment/input_0001.npy … input_0050.npy` and `A_deployment/outputs.jsonl`, one JSON line per frame.

## B. Latency (must, 30 minutes)

Run SpatialNet continuously for **5 minutes** at the highest rate the pipeline allows, and log every frame. We discard the first 100 frames as warm-up.

**Per-frame log:** `B_latency_run1.csv`, with these columns:

| Column | Meaning |
|---|---|
| `frame_id` | Running frame number |
| `t_capture_ms` | Camera timestamp of the frame |
| `pre_ms` | Resize and normalisation |
| `infer_ms` | The model call only, from input ready to outputs ready |
| `post_ms` | Decoding the outputs (boxes, points, probabilities) |
| `total_ms` | From camera frame available to decoded outputs ready |
| `soc_temp_c` | Chip or skin temperature, every 10 s if available |

A second, shorter run (1 minute, `B_latency_run2.csv`) after the device has cooled down lets us separate warm and cold behaviour.

**From this we report:** the median, 10th and 90th percentiles, and mean ± SD of `infer_ms` and `total_ms`; the achieved frames per second; and whether latency drifts during the 5 minutes (thermal throttling).

## C. Memory (must, 10 minutes)

Sample the memory of the app process (PSS and RSS) **once per second** in three states, for about 2 minutes each:

1. app running, model not loaded;
2. model loaded, idle;
3. continuous inference.

If the stack exposes Android tools, `adb shell dumpsys meminfo <package>` gives PSS. Please also tell us the size of the model file on the device.

**Send back:** `C_memory.csv` with columns `t_s, state, pss_kb, rss_kb`.

## D. Power (must, about 2 hours including recharging)

**Instrument, in order of preference:**

1. An external power monitor on the battery, recording power or current at 1 kHz or faster.
2. The device's fuel-gauge counters sampled every second: charge counter in µAh, current in µA and voltage in mV. On Android: `BatteryManager` `CHARGE_COUNTER` and `CURRENT_NOW`, or `dumpsys battery`.
3. If neither is available, battery percentage only. Each run must then last at least **10 minutes**, because 1% steps are too coarse for shorter runs.

**Modes.** Keep the same Wi-Fi and Bluetooth state throughout, except where streaming needs Wi-Fi.

| Mode | Camera | Model | Radio |
|---|---|---|---|
| M0 Idle | off | off | connected, idle |
| M1 Camera only | on, same resolution and fps as M2 | off | connected, idle |
| M2 SpatialNet continuous | on | on every frame (as in the earlier measurement) | connected, idle |
| M3 Streaming | on | off | sending frames to the phone at 0.5 frames/s (as in the earlier measurement) |

**Protocol:**

- **3 runs per mode, each 5 minutes.**
- Start every run between **80% and 90%** charge, and recharge in between.
- Wait 2 minutes after a mode starts before you begin logging.
- Interleave the modes (for example M0, M2, M3, M1, M2, M0, …) so that slow drift does not favour one mode.
- Note the temperature at the start and end of each run.

**Per-run log:** `D_power_<mode>_run<k>.csv`, with columns `t_s, battery_pct, charge_counter_uah, current_ua, voltage_mv, temp_c, frames_processed`. Leave a column empty if it is not available.

**From this we report:** the mean current or power per mode (mean ± SD over the three runs), the extra cost of the model over the camera alone (M2 − M1), the energy per processed frame, and the comparison with streaming (M3).

## E. Detector models on the glasses (nice to have, 30 minutes)

We will send two small object detectors as ONNX files:

- YOLO26n, with a 128 × 128 input;
- NanoDet-Plus-m, with a 160 × 160 input.

Please run each with the latency protocol (B, one 2-minute run) and the memory protocol (C, state 3 only). If an operator is not supported on the compute unit, tell us which one and where the model fell back. This lets us compare SpatialNet with them on the device, not only on paper.

**Send back:** `E_<model>_latency.csv` and `E_<model>_memory.csv`.

## F. Extra power modes (nice to have, about 1 hour)

Use the same protocol as D, with 2 runs per mode.

- **M4 SpatialNet duty-cycled:** the model runs on 2 frames per second, closer to real use, where it only needs to run while the wearer points.
- **M5 Streaming at a matched rate:** frames sent to the phone at the rate M2 actually achieves, or at 2 frames per second if that is not feasible.

## G. Real frames from the glasses (must, about 1–1.5 hours)

All our accuracy numbers come from a public dataset of mostly landscape photos. This is the only check that the model also works on the glasses' own camera, whose portrait frames are squashed to 160 × 160. Only the wearer is needed; no other people.

**People and consent:**

- 1–3 wearers. Different people are better (hand size, skin tone, left or right hand), but one is fine if time is short. Each wearer signs a consent form for research use.
- Record where no other people are in view. If someone appears, blur the face after the model has run, not before.

**What to capture.** Capture single frames from the running pipeline: for each frame, save the frame and the outputs the model computed on that same frame.

- **G1. About 120–150 pointing frames.** In each frame, the wearer points at one everyday object. Please spread them over 4–6 places (for example a desk, a kitchen, a living room, a corridor, a shop shelf, outdoors) and vary:
  - **distance in the view:** finger touching or in front of the object, object just beyond the fingertip, object across the room;
  - **object size:** small (keys, a pen, a remote), medium (a cup, a laptop, a plant), large (a chair, a door, a car);
  - **hand position:** lower, middle and side of the view.

  Use a different object for every frame or two; at most 3 frames per object.
- **G2. About 30–40 frames with no hand in view,** in the same places.
- **G3. Optional, about 50 frames:** only if a companion is available and consents. The wearer points while the companion's hand is in view (resting, holding something, or pointing). Skip this if it is not easy.

**The object the wearer meant.** For every G1 frame, note the object the wearer was pointing at, for example by typing it into a log as the frame is captured. This is our ground truth. We draw the boxes ourselves, without looking at the model's outputs.

**For every frame, save:**

- the full camera frame (`.jpg`), or at least the 160 × 160 model input if full frames cannot leave your site;
- the on-device outputs, in the same format as in A.

**Send back:**

- `G_frames/`;
- `G_outputs.jsonl`, one line per frame;
- `G_targets.csv`, with columns `frame_id, part (G1/G2/G3), object, place, distance (touching/near/far)`;
- a short note confirming consent and how the images will be stored. No names, please.

**From this we report:** how often the hand box, the target box (IoU ≥ 0.5) and the target centre are right, and how often no-hand frames are classed as no hand, each with a 95% interval, next to the same numbers on the public dataset.

## H. Extras for a stronger paper (nice to have, about 45 minutes)

Same wearers, places and consent as G. Each part is independent.

- **H1. About 30 frames with the wearer's hand visible but not pointing,** for example holding a cup or a phone, resting on a table, or waving. Save the frames and outputs as in G. This is the first test of the model's "pointing" output, which the public dataset cannot test.
- **H2. Five clips of one minute of natural use,** in which the wearer sometimes points and otherwise does everyday things. Save every frame's outputs with its timestamp (the frames themselves are not needed). From these we measure how often the model would trigger by mistake per minute, with and without requiring several frames in a row.
- **H3. A demo video of 1–2 minutes:** the glasses' view with the model's outputs drawn live (hand box, fingertip, ray, target box), while the wearer points at a few objects. Please blur any faces. We would include it as supplementary material for the reviewers.

**Send back:** `H1_frames/`, `H1_outputs.jsonl`, `H2_clip<k>_outputs.jsonl` and `H3_demo.mp4`.

## What will be published

The paper will contain only summary numbers:

- latency: median and 10th–90th percentile;
- throughput;
- memory, in MB;
- mean current or power per mode (mean ± SD) and the extra cost of the model;
- if E is done, the same latency and memory figures for the two detectors;
- accuracy on the real frames of G, with a few example frames (faces blurred; only frames the wearers agreed to show);
- if H is done, the share of non-pointing frames read as pointing, false triggers per minute, and the demo video as supplementary material.

Raw traces, the device configuration and the runtime details stay internal.

## Folder to send back

```
spatialnet_measurements/
  0_setup.txt            device, runtime, compute unit, precision, camera mode, settings, preprocessing confirmation
  A_deployment/          input_0001.npy ... input_0050.npy, outputs.jsonl
  B_latency_run1.csv, B_latency_run2.csv
  C_memory.csv
  D_power_M0_run1.csv ... D_power_M3_run3.csv
  E_...                  (if done)
  F_...                  (if done)
  G_frames/, G_outputs.jsonl, G_targets.csv, G_consent_note.txt
  H_...                  (if done)
```

Thank you. Please contact [name, email] with any question; we can also join a call to go through the setup.
