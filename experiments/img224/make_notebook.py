"""Build the 224 px experiment notebook from the main notebook (notebooks/spatialnet_kaggle.ipynb).
The only change to the run is --img_size 224; code, data and all other settings are the main run's.
Run from the repository root: python experiments/img224/make_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'notebooks', 'spatialnet_kaggle.ipynb')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spatialnet_kaggle_224.ipynb')

nb = json.load(open(SRC, encoding='utf-8'))


def edit(i, pairs):
    text = ''.join(nb['cells'][i]['source'])
    for old, new in pairs:
        assert old in text, f'cell {i}: {old!r} not found'
        text = text.replace(old, new)
    nb['cells'][i]['source'] = text.splitlines(keepends=True)


edit(0, [
    ('# SpatialNet — Train & Evaluate on Kaggle', '# SpatialNet — 224 px input experiment'),
    ('Trains SpatialNet, evaluates it on the test split, and saves every statistic and figure for the paper.',
     'Same code, data and settings as the main run, with one change: the input is 224 x 224 instead of 160 x 160 '
     '(`--img_size 224`).\nThe model and its parameters are unchanged (722,104); the feature grid becomes 28 x 28 and '
     'compute rises from 0.23 to 0.45 GFLOPs.\nIt tests whether input resolution limits target localisation. '
     'Fingertip errors are reported in pixels at 224 x 224.'),
    ('outputs_spatialnet/                     (quick_run/ when QUICK_RUN = True)',
     'outputs_spatialnet_224/                 (quick_run_224/ when QUICK_RUN = True)'),
])
edit(2, [
    ('QUICK_RUN = True ', 'QUICK_RUN = False'),
    ('RESUME_FROM = ', 'IMG_SIZE = 224       # the only difference from the main run (160)\nRESUME_FROM = '),
    ("('quick_run' if QUICK_RUN else 'outputs_spatialnet')", "('quick_run_224' if QUICK_RUN else 'outputs_spatialnet_224')"),
])
edit(6, [
    ("'--output_dir', OUTPUT_DIR, '--quantize']",
     "'--output_dir', OUTPUT_DIR, '--quantize',\n       '--img_size', str(IMG_SIZE)]"),
])
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None
json.dump(nb, open(DST, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('wrote', DST)
for i in (2, 6):
    print(f'--- cell {i}\n' + ''.join(nb['cells'][i]['source']))
