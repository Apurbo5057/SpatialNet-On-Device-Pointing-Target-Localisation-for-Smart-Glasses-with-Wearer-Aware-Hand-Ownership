"""Small-target P@0.5 (target box under 3% of the image) for SpatialNet runs, paired with the reference run.

    python revised_paper/scripts/small_targets.py runs/main_run runs/E1_forward_ray ...
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from annotation import parse_annotation  # noqa: E402

LABELS = os.path.join(ROOT, 'EgoPoint_spatialnet', 'test', 'labels')
small = set()
for f in os.listdir(LABELS):
    roi = parse_annotation(os.path.join(LABELS, f))['roi']
    if roi is not None and roi[2] * roi[3] < 0.03:
        small.add(f[:-4])


def load(run):
    p = glob.glob(os.path.join(run, '*', 'evaluation', 'predictions', 'test.csv'))[0]
    t = pd.read_csv(p).set_index('name')
    return 100.0 * (t.loc[sorted(small & set(t.index)), 'target_iou'] >= 0.5)


ref = load(sys.argv[1])
rng = np.random.default_rng(0)
print(f'small targets: {len(ref)} images; {os.path.basename(sys.argv[1])} P@0.5 = {ref.mean():.1f}')
for run in sys.argv[2:]:
    v = load(run)
    idx = ref.index.intersection(v.index)
    d = (v.loc[idx] - ref.loc[idx]).values
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    print(f'{os.path.basename(run)}: {v.mean():.1f}  diff {d.mean():+.1f}  95% CI [{np.percentile(bs, 2.5):+.1f}, '
          f'{np.percentile(bs, 97.5):+.1f}]')
