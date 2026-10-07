"""Build a Kaggle notebook for one experiment variant (ablation, seed, ...) from the main notebook.

The variant uses the same code and data datasets as the main run; only the output folder and the
extra train flags differ. Nothing in the codebase changes.

    python experiments/make_variant.py --name A1_no_copypaste --kernel spatialnet-ablation-a1 \
        --title "SpatialNet Ablation A1" --flags "--copy_paste 0 --ownership_weight 0" \
        --desc "A1: no distractor-hand copy-paste and no ownership weighting"

Writes experiments/<name>/<notebook>.ipynb and kernel-metadata.json; push with
    kaggle kernels push -p experiments/<name>
"""
import argparse
import json
import os
import shlex

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, 'notebooks', 'spatialnet_kaggle.ipynb')
USER = 'mosharafapurbo'
DATASETS = [f'{USER}/egopoint-spatialnet', f'{USER}/spatialnet-code']

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True, help='folder under experiments/ and output folder suffix')
ap.add_argument('--kernel', required=True, help='Kaggle kernel slug, e.g. spatialnet-ablation-a1')
ap.add_argument('--title', required=True)
ap.add_argument('--flags', default='', help='extra train.py flags, as one string')
ap.add_argument('--desc', required=True, help='one line describing the variant')
args = ap.parse_args()

out_dir = os.path.join(HERE, args.name)
os.makedirs(out_dir, exist_ok=True)
nb_file = f'spatialnet_{args.name.lower()}.ipynb'
output_name = args.name.lower()
flags = shlex.split(args.flags)

nb = json.load(open(SRC, encoding='utf-8'))


def edit(i, pairs):
    text = ''.join(nb['cells'][i]['source'])
    for old, new in pairs:
        assert old in text, f'cell {i}: {old!r} not found in the main notebook'
        text = text.replace(old, new)
    nb['cells'][i]['source'] = text.splitlines(keepends=True)


edit(0, [
    ('# SpatialNet — Train & Evaluate on Kaggle', f'# SpatialNet — {args.title}'),
    ('Trains SpatialNet, evaluates it on the test split, and saves every statistic and figure for the paper.',
     f'{args.desc}.\nSame code, data and settings as the main run; the only difference is the extra train flags: '
     f'`{" ".join(flags) or "(none)"}`.'),
    ('outputs_spatialnet/                     (quick_run/ when QUICK_RUN = True)',
     f'{output_name}/'),
])
edit(2, [
    ('QUICK_RUN = True ', 'QUICK_RUN = False'),
    ('RESUME_FROM = ', f'EXTRA_FLAGS = {flags!r}   # the only difference from the main run\nRESUME_FROM = '),
    ("('quick_run' if QUICK_RUN else 'outputs_spatialnet')", f"('quick_run' if QUICK_RUN else {output_name!r})"),
])
edit(6, [("'--output_dir', OUTPUT_DIR, '--quantize']", "'--output_dir', OUTPUT_DIR, '--quantize'] + EXTRA_FLAGS")])
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None
json.dump(nb, open(os.path.join(out_dir, nb_file), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

meta = {'id': f'{USER}/{args.kernel}', 'title': args.title, 'code_file': nb_file, 'language': 'python',
        'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': False, 'dataset_sources': DATASETS, 'competition_sources': [], 'kernel_sources': []}
json.dump(meta, open(os.path.join(out_dir, 'kernel-metadata.json'), 'w', encoding='utf-8'), indent=2)
print('wrote', out_dir)
for i in (2, 6):
    print(f'--- cell {i}\n' + ''.join(nb['cells'][i]['source']))
