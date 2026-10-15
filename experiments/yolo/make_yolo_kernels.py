"""Write one Kaggle script kernel per YOLO26n mode from yolo_experiment_template.py.

    python experiments/yolo/make_yolo_kernels.py
Push each folder with  kaggle kernels push -p experiments/<folder>
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
USER = 'mosharafapurbo'
TEMPLATE = open(os.path.join(HERE, 'yolo_experiment_template.py'), encoding='utf-8').read()

RUNS = [  # folder, mode, quick, title (its slug must equal the kernel slug), gpu, training seed
    ('Y0_yolo_quick_test', 'imagenet_paste', True, 'SpatialNet Exp Y0 YOLO Quick Test', False, 42),
    ('Y1_yolo_imagenet', 'imagenet', False, 'SpatialNet Exp Y1 YOLO ImageNet', True, 42),
    ('Y2_yolo_imagenet_paste', 'imagenet_paste', False, 'SpatialNet Exp Y2 YOLO ImageNet Paste', True, 42),
    ('Y3_yolo_coco', 'coco', False, 'SpatialNet Exp Y3 YOLO COCO', True, 42),
    ('Y4_yolo_imagenet_seed1', 'imagenet', False, 'SpatialNet Exp Y4 YOLO ImageNet Seed1', True, 1),
]

for folder, mode, quick, title, gpu, seed in RUNS:
    out = os.path.join(EXP, folder)
    os.makedirs(out, exist_ok=True)
    code = f'yolo_{mode}{"_quick" if quick else ""}.py'
    with open(os.path.join(out, code), 'w', encoding='utf-8') as f:
        f.write(TEMPLATE.replace("'__MODE__'", repr(mode)).replace('__QUICK__', str(quick))
                .replace('__SEED__', str(seed)))
    slug = title.lower().replace(' ', '-')
    meta = {'id': f'{USER}/{slug}', 'title': title, 'code_file': code, 'language': 'python',
            'kernel_type': 'script', 'is_private': True, 'enable_gpu': gpu, 'enable_tpu': False,
            'enable_internet': True,
            'dataset_sources': [f'{USER}/egopoint-spatialnet', f'{USER}/spatialnet-code-exp'],
            'competition_sources': [], 'kernel_sources': []}
    with open(os.path.join(out, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('wrote', folder, meta['id'], 'gpu' if gpu else 'cpu')
