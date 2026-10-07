"""
Convert the EgoPoint dataset (img/ + json/ per split) into the SpatialNet training layout
(docs/annotation_format.md): 224x224 images, H P T labels, manifest.csv, stats.json and README.md.
The official train/val/test split is kept as is. Run from the repository root:
    python egopoint_preprocessing/prepare_egopoint.py --src EgoPoint_dataset/dataset --out EgoPoint_spatialnet \
        --doc docs/dataset_egopoint.md
"""

import os
import sys
import csv
import json
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageOps

# the label format lives with the model code (annotation.py in the repository root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from annotation import format_annotation, parse_annotation, fingertip_roi_gap, gap_bin  # noqa: E402

SPLITS = ['train', 'val', 'test']
SCENE = {(0, 0, 0): 'No hand', (1, 0, 0): 'Non-pointing', (1, 1, 0): 'Pointing, no target',
         (1, 1, 1): 'Pointing at target'}
SCENE_FLAGS = {name: ' '.join(map(str, f)) for f, name in SCENE.items()}


def _norm_box(b, w, h):
    """Pixel x1,y1,x2,y2 -> normalised cx,cy,w,h clipped to the image; None if degenerate."""
    x1, y1 = max(0.0, min(b[0], b[2])), max(0.0, min(b[1], b[3]))
    x2, y2 = min(float(w), max(b[0], b[2])), min(float(h), max(b[1], b[3]))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]


def _inside(p, b, pad=3):
    return b[0] - pad <= p[0] <= b[2] + pad and b[1] - pad <= p[1] <= b[3] + pad


def convert_record(d):
    """EgoPoint JSON -> (hand, fingertip, target, info) in normalised units, or (None, reason)."""
    w, h = d['width'], d['height']
    info = {'sample_type': d['sample_type'], 'clear_or_not': d.get('clear_or_not'),
            'fingertip_source': '', 'target_category': '', 'fix': ''}
    if d['sample_type'] == 'negative':
        return (None, None, None, info), None

    anns = d.get('annotations') or []
    hand = [a for a in anns if a.get('ann_id') == 'anno_hand']
    others = [a for a in anns if a.get('ann_id') != 'anno_hand']
    tip = d.get('fingertip')
    # a few files swap the hand and object labels: the 'anno_hand' box is an object and the real
    # hand is stored as an object with category 'hand' that contains the fingertip
    if hand and tip and hand[0].get('category_name') != 'hand':
        swap = [a for a in others if a.get('category_name') == 'hand' and _inside(tip, a['bbox'])]
        if len(swap) == 1:
            others = [a for a in others if a is not swap[0]] + [hand[0]]
            hand = swap
            info['fix'] = 'hand/object labels swapped in source'
    if len(hand) != 1:
        return None, 'positive without a hand box'
    if len(others) != 1:
        return None, f'positive with {len(others)} target boxes'
    if tip is None:
        # annotators left these unmarked: open palms, held objects, or the box on the non-pointing hand
        return None, 'positive without an annotated fingertip'
    info['fingertip_source'] = 'annotated'
    hb, tb = _norm_box(hand[0]['bbox'], w, h), _norm_box(others[0]['bbox'], w, h)
    if hb is None or tb is None:
        return None, 'degenerate box'
    ft = [min(max(tip[0] / w, 0.0), 1.0), min(max(tip[1] / h, 0.0), 1.0)]
    info['target_category'] = others[0].get('category_name', '')
    return (hb, ft, tb, info), None


def _process(job):
    """Worker: convert one record, write image + label. Returns a manifest row or a drop reason."""
    split, jpath, src_img_dir, out, size, quality = job
    with open(jpath, encoding='utf-8') as f:
        d = json.load(f)
    name = os.path.splitext(d['file_name'])[0]
    ipath = os.path.join(src_img_dir, d['file_name'])
    if not os.path.exists(ipath):
        return {'split': split, 'name': name, 'drop': 'image file missing'}
    res, reason = convert_record(d)
    if reason:
        return {'split': split, 'name': name, 'drop': reason}
    hand, tip, target, info = res
    img = Image.open(ipath)
    raw = img.size
    if raw != (d['width'], d['height']):
        img = ImageOps.exif_transpose(img)
        if img.size != (d['width'], d['height']):
            return {'split': split, 'name': name, 'drop': f'image {raw} != annotated size'}
    img.convert('RGB').resize((size, size), Image.LANCZOS).save(
        os.path.join(out, split, 'images', name + '.jpg'), quality=quality)
    with open(os.path.join(out, split, 'labels', name + '.txt'), 'w') as f:
        f.write(format_annotation(hand, tip, target))
    return {'split': split, 'name': name, 'orig_width': d['width'], 'orig_height': d['height'],
            'H': int(hand is not None), 'P': int(tip is not None), 'T': int(target is not None), **info}


def _quantiles(a):
    return {k: float(np.percentile(a, p)) for k, p in (('p10', 10), ('median', 50), ('p90', 90))} if a else {}


def scene_stats(items):
    """Statistics over (row, parsed label) pairs of one scene."""
    out = {'images': len(items), 'per_split': {s: sum(r['split'] == s for r, _ in items) for s in SPLITS},
           'orig_width': _quantiles([r['orig_width'] for r, _ in items]),
           'orig_height': _quantiles([r['orig_height'] for r, _ in items])}
    hands = [a['hand'] for _, a in items if a['hand']]
    targets = [(a['fingertip'], a['roi']) for _, a in items if a['roi']]
    tips = [a['fingertip'] for _, a in items if a['fingertip']]
    if hands:
        out['hand_box_area'] = _quantiles([h[2] * h[3] for h in hands])
        out['hand_centre_y'] = _quantiles([h[1] for h in hands])
    if tips:
        out['fingertip_y'] = _quantiles([t[1] for t in tips])
        out['fingertip_source'] = dict(Counter(r['fingertip_source'] for r, a in items if a['fingertip']))
        out['clear_or_not'] = dict(Counter(str(r['clear_or_not']) for r, _ in items))
    if targets:
        gaps = [fingertip_roi_gap(t, b) for t, b in targets]
        out['target_box_area'] = _quantiles([b[2] * b[3] for _, b in targets])
        out['gap'] = _quantiles(gaps)
        out['target_gap'] = {b: sum(gap_bin(g) == b for g in gaps) for b in ['on-object', 'near', 'far']}
        cats = Counter(r['target_category'] for r, a in items if a['roi'])
        out['target_categories'] = len(cats)
        out['top_target_categories'] = cats.most_common(10)
    return out


def collect_stats(out, rows, drops):
    st = {'splits': {}, 'dropped': drops}
    by_scene = {s: [] for s in SCENE.values()}
    for split in SPLITS:
        r = [x for x in rows if x['split'] == split]
        scenes, gaps, src, clear, cats = Counter(), Counter(), Counter(), Counter(), Counter()
        hand_area, tgt_area, gap_vals, aspect = [], [], [], []
        for x in r:
            ann = parse_annotation(os.path.join(out, split, 'labels', x['name'] + '.txt'))
            scenes[SCENE[ann['flags']]] += 1
            by_scene[SCENE[ann['flags']]].append((x, ann))
            aspect.append(x['orig_width'] / x['orig_height'])
            if ann['hand']:
                hand_area.append(ann['hand'][2] * ann['hand'][3])
            if ann['roi']:
                tgt_area.append(ann['roi'][2] * ann['roi'][3])
                g = fingertip_roi_gap(ann['fingertip'], ann['roi'])
                gap_vals.append(g)
                gaps[gap_bin(g)] += 1
                cats[x['target_category']] += 1
            if x['fingertip_source']:
                src[x['fingertip_source']] += 1
            if x['sample_type'] == 'positive':
                clear[str(x['clear_or_not'])] += 1
        q = lambda a: {k: float(np.percentile(a, p)) for k, p in (('p10', 10), ('median', 50), ('p90', 90))} if a else {}
        st['splits'][split] = {
            'images': len(r), 'scenes': {s: scenes.get(s, 0) for s in SCENE.values()},
            'target_gap': {b: gaps.get(b, 0) for b in ['on-object', 'near', 'far']},
            'fingertip_source': dict(src), 'clear_or_not': dict(clear),
            'hand_box_area': q(hand_area), 'target_box_area': q(tgt_area), 'gap': q(gap_vals),
            'orig_aspect_w_over_h': q(aspect),
            'orig_orientation': {'landscape': sum(a > 1 for a in aspect), 'square': sum(a == 1 for a in aspect),
                                 'portrait': sum(a < 1 for a in aspect)},
            'target_categories': len(cats), 'top_target_categories': cats.most_common(15)}
    st['scenes'] = {s: scene_stats(items) for s, items in by_scene.items()}
    return st


def draw_examples(out, picks, path, title):
    """Redraw the converted labels on converted images: hand box, fingertip, target box, H P T."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    cols = 4
    rows_n = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 3.55 * rows_n))
    axes = np.array(axes).reshape(-1)
    for ax, r in zip(axes, picks):
        img = Image.open(os.path.join(out, r['split'], 'images', r['name'] + '.jpg'))
        a = parse_annotation(os.path.join(out, r['split'], 'labels', r['name'] + '.txt'))
        s = img.size[0]
        ax.imshow(img)
        for b, c in ((a['hand'], '#2a78d6'), (a['roi'], '#1baf7a')):
            if b:
                ax.add_patch(patches.Rectangle(((b[0] - b[2] / 2) * s, (b[1] - b[3] / 2) * s), b[2] * s, b[3] * s,
                                               lw=2, ec=c, fc='none'))
        if a['fingertip']:
            ax.plot(a['fingertip'][0] * s, a['fingertip'][1] * s, 'o', color='#eb6834', ms=7, mec='white', mew=1)
        note = 'labels swapped back' if r['fix'] else ''
        ax.set_title(f"{r['split']}/{r['name']}   H P T = {' '.join(map(str, a['flags']))}\n"
                     f"{SCENE[a['flags']]}{': ' + r['target_category'] if r['target_category'] else ''}"
                     f"{'  (' + note + ')' if note else ''}", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(picks):]:
        ax.axis('off')
    fig.suptitle(f'{title}   blue = hand box, orange = fingertip, green = target box', fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_figures(out, rows, fig_dir, seed=1):
    """Random examples per category, and every image whose hand/object labels were swapped back."""
    import random
    rng = random.Random(seed)
    pos = [r for r in rows if str(r['T']) == '1' and not r['fix'] and r['fingertip_source'] == 'annotated']
    neg = [r for r in rows if str(r['H']) == '0']
    picks = [rng.choice([r for r in pos if r['split'] == s]) for s in SPLITS] + rng.sample(pos, 5) + \
        [rng.choice([r for r in neg if r['split'] == s]) for s in SPLITS] + rng.sample(neg, 1)
    draw_examples(out, picks, os.path.join(fig_dir, 'examples.png'), 'Random converted images per category.')
    special = [r for r in rows if r['fix']]
    if special:
        draw_examples(out, special, os.path.join(fig_dir, 'special_cases.png'),
                      'Source hand/object labels swapped back.')


def write_readme(st, args, path, fig_rel=None):
    S = st['splits']
    tot = lambda f: sum(f(S[s]) for s in SPLITS)
    L = ['# EgoPoint for SpatialNet', '',
         f'EgoPoint converted to the SpatialNet label format (`annotation_format.md`) by '
         f'`egopoint_preprocessing/prepare_egopoint.py`. '
         f'The official train / val / test split is kept unchanged. Images are resized to {args.size} x {args.size} '
         f'(JPEG quality {args.quality}); labels are normalised, so they do not depend on image size.', '',
         '## Layout', '', '```text', f'{os.path.basename(os.path.normpath(args.out))}/',
         '  train/ val/ test/', '    images/  <name>.jpg', '    labels/  <name>.txt   (H P T, hand box, fingertip, target box)',
         '  manifest.csv   one row per image: source fields and conversion notes',
         '  stats.json     every number in this file', '  README.md', '```', '',
         '## Images per split and scene', '',
         '| Split | Images | ' + ' | '.join(SCENE.values()) + ' |', '|---|---:|' + '---:|' * len(SCENE)]
    for s in SPLITS:
        L.append(f'| {s} | {S[s]["images"]} | ' + ' | '.join(str(S[s]['scenes'][k]) for k in SCENE.values()) + ' |')
    L.append(f'| total | {tot(lambda x: x["images"])} | '
             + ' | '.join(str(tot(lambda x, k=k: x['scenes'][k])) for k in SCENE.values()) + ' |')
    L += ['', 'EgoPoint has no "Non-pointing" (hand visible, not pointing) and no "Pointing, no target" images: '
              'every positive image shows a pointing hand with one target.', '',
          '## Pointing-at-target images by fingertip-target gap', '',
          'Gap = distance from the fingertip to the target box, as a fraction of image size '
          '(on-object = touching, near <= 0.08, far > 0.08).', '',
          '| Split | On-object | Near | Far | Median gap |', '|---|---:|---:|---:|---:|']
    for s in SPLITS:
        g = S[s]['target_gap']
        L.append(f'| {s} | {g["on-object"]} | {g["near"]} | {g["far"]} | {S[s]["gap"].get("median", 0):.3f} |')
    L += ['', '## Box sizes (fraction of image area)', '', '| Split | Hand p10 / median / p90 | Target p10 / median / p90 |',
          '|---|---:|---:|']
    for s in SPLITS:
        h, t = S[s]['hand_box_area'], S[s]['target_box_area']
        L.append(f'| {s} | {h["p10"]:.3f} / {h["median"]:.3f} / {h["p90"]:.3f} | '
                 f'{t["p10"]:.3f} / {t["median"]:.3f} / {t["p90"]:.3f} |')
    L += ['', '## Source properties', '', '| Split | clear_or_not true / false / missing (positives) | '
          'Original landscape / square / portrait | Target categories |', '|---|---:|---:|---:|']
    for s in SPLITS:
        x = S[s]
        c, o = x['clear_or_not'], x['orig_orientation']
        L.append(f'| {s} | {c.get("True", 0)} / {c.get("False", 0)} / {c.get("None", 0)} | '
                 f'{o["landscape"]} / {o["square"]} / {o["portrait"]} | {x["target_categories"]} |')
    L += ['', '`clear_or_not` is kept from the source in `manifest.csv`; spot checks show correct labels for both '
              'values, so it is not used to filter.', '',
          '**Most frequent target categories (train):** '
          + ', '.join(f'{k} ({v})' for k, v in S['train']['top_target_categories']) + '.', '',
          '## Conversion rules', '',
          '| EgoPoint record | Result |', '|---|---|',
          '| `sample_type` negative (no hand; its COCO object boxes are ignored) | `0 0 0` |',
          '| `sample_type` positive: `anno_hand` box + annotated `fingertip` + one target box | `1 1 1` |',
          '| `anno_hand` label on an object and the real hand labelled as an object (detected by the fingertip) '
          '| labels swapped back, then `1 1 1` |',
          '| positive without an annotated `fingertip` (all flagged `clear_or_not = false`; spot checks show open '
          'palms, held objects, or the hand box on the non-pointing hand) | dropped |',
          '| positive without an `anno_hand` box, or without a target box | dropped |', '',
          '- Source boxes are pixel `x1, y1, x2, y2`; they are clipped to the image and stored as normalised `cx cy w h`.',
          '- Fingertips are normalised and clamped to [0, 1].',
          '- Images are squashed to a square, as the model does at its 160 x 160 input; normalised labels are unaffected.', '',
          '## Dropped records', '', '| Reason | Train | Val | Test | Total |', '|---|---:|---:|---:|---:|']
    reasons = Counter(d['drop'] for d in st['dropped'])
    for reason, n in reasons.most_common():
        per = [sum(d['drop'] == reason and d['split'] == s for d in st['dropped']) for s in SPLITS]
        L.append(f'| {reason} | ' + ' | '.join(map(str, per)) + f' | {n} |')
    L += ['', f'Every dropped file name is listed in `stats.json`. Images without a JSON file are not converted '
              f'({st.get("images_without_json", 0)} found).']
    if fig_rel:
        L += ['', '## Visual check of the converted labels', '',
              'Labels redrawn from the converted `labels/*.txt` onto the converted 224 x 224 images, '
              'so these show exactly what training reads.', '',
              f'![Random converted images per category]({fig_rel}examples.png)', '',
              'The test images whose hand and object labels were swapped in the source, after the fix:', '',
              f'![Corrected cases]({fig_rel}special_cases.png)']

    sc = st['scenes']
    present = [s for s in SCENE.values() if sc[s]['images']]
    absent = [s for s in SCENE.values() if not sc[s]['images']]
    n_all = sum(sc[s]['images'] for s in SCENE.values())
    L += ['', '## Statistics per category', '',
          f'Only {len(present)} of the 4 categories occur in the converted dataset: '
          + ', '.join(f'**{s}** ({SCENE_FLAGS[s]})' for s in present) + '. '
          + (' and '.join(absent) + ' have no images.' if absent else ''), '',
          '| Category | H P T | Train | Val | Test | Total | Share |', '|---|---|---:|---:|---:|---:|---:|']
    for s in SCENE.values():
        x = sc[s]
        L.append(f'| {s} | {SCENE_FLAGS[s]} | ' + ' | '.join(str(x['per_split'][p]) for p in SPLITS)
                 + f' | {x["images"]} | {100 * x["images"] / max(n_all, 1):.1f}% |')
    fmt3 = lambda q: f'{q["p10"]:.3f} / {q["median"]:.3f} / {q["p90"]:.3f}' if q else '-'
    fmt0 = lambda q: f'{q["p10"]:.0f} / {q["median"]:.0f} / {q["p90"]:.0f}' if q else '-'
    for s in present:
        x = sc[s]
        L += ['', f'### {s} ({SCENE_FLAGS[s]})', '', '| Property (p10 / median / p90) | Value |', '|---|---:|',
              f'| Original width px | {fmt0(x["orig_width"])} |', f'| Original height px | {fmt0(x["orig_height"])} |']
        if 'hand_box_area' in x:
            L += [f'| Hand box area (fraction of image) | {fmt3(x["hand_box_area"])} |',
                  f'| Hand centre y (0 = top, 1 = bottom) | {fmt3(x["hand_centre_y"])} |',
                  f'| Fingertip y | {fmt3(x["fingertip_y"])} |']
        if 'target_box_area' in x:
            g = x['target_gap']
            L += [f'| Target box area (fraction of image) | {fmt3(x["target_box_area"])} |',
                  f'| Fingertip-target gap | {fmt3(x["gap"])} |',
                  f'| Gap on-object / near / far (images) | {g["on-object"]} / {g["near"]} / {g["far"]} |',
                  f'| clear_or_not true / false / missing (images) | {x["clear_or_not"].get("True", 0)} / '
                  f'{x["clear_or_not"].get("False", 0)} / {x["clear_or_not"].get("None", 0)} |',
                  f'| Distinct target categories | {x["target_categories"]} |',
                  '| Most frequent targets | ' + ', '.join(f'{k} ({v})' for k, v in x['top_target_categories']) + ' |']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


def write_reports(st, rows, args):
    """README.md + figures/ inside the dataset, and optionally the same description under docs/."""
    make_figures(args.out, rows, os.path.join(args.out, 'figures'))
    write_readme(st, args, os.path.join(args.out, 'README.md'), fig_rel='figures/')
    if args.doc:
        doc_fig_dir = os.path.join(os.path.dirname(args.doc) or '.', 'figures', 'dataset_egopoint')
        make_figures(args.out, rows, doc_fig_dir)
        write_readme(st, args, args.doc, fig_rel='figures/dataset_egopoint/')


def main():
    ap = argparse.ArgumentParser(description='Prepare EgoPoint for SpatialNet')
    ap.add_argument('--src', default='EgoPoint_dataset/dataset', help='folder with img/ and json/')
    ap.add_argument('--out', default='EgoPoint_spatialnet')
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--quality', type=int, default=95)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--doc', default=None, help='also write the dataset description here (e.g. docs/dataset_egopoint.md)')
    ap.add_argument('--report_only', action='store_true',
                    help='only rewrite README, --doc and figures from an existing --out')
    args = ap.parse_args()
    if args.report_only:
        with open(os.path.join(args.out, 'manifest.csv'), encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        with open(os.path.join(args.out, 'stats.json')) as f:
            st = json.load(f)
        write_reports(st, rows, args)
        return
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit(f'{args.out} exists and is not empty; choose a new --out')

    jobs, no_json = [], 0
    for split in SPLITS:
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(args.out, split, sub), exist_ok=True)
        jdir, idir = os.path.join(args.src, 'json', split), os.path.join(args.src, 'img', split)
        jsons = sorted(f for f in os.listdir(jdir) if f.endswith('.json'))
        stems = {os.path.splitext(f)[0] for f in jsons}
        no_json += sum(os.path.splitext(f)[0] not in stems for f in os.listdir(idir))
        jobs += [(split, os.path.join(jdir, f), idir, args.out, args.size, args.quality) for f in jsons]

    print(f'Converting {len(jobs)} records with {args.workers} workers...', flush=True)
    rows, drops = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(_process, jobs, chunksize=64)):
            (drops if 'drop' in r else rows).append(r)
            if (i + 1) % 2000 == 0:
                print(f'  {i + 1}/{len(jobs)}', flush=True)

    rows.sort(key=lambda r: (SPLITS.index(r['split']), r['name']))
    cols = ['split', 'name', 'H', 'P', 'T', 'sample_type', 'clear_or_not', 'fingertip_source',
            'target_category', 'orig_width', 'orig_height', 'fix']
    with open(os.path.join(args.out, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    st = collect_stats(args.out, rows, drops)
    st['images_without_json'] = no_json
    st['settings'] = {'size': args.size, 'jpeg_quality': args.quality, 'source': os.path.abspath(args.src)}
    st['fixes'] = Counter(r['fix'] for r in rows if r['fix'])
    with open(os.path.join(args.out, 'stats.json'), 'w') as f:
        json.dump(st, f, indent=2)
    write_reports(st, rows, args)

    print(f'\nDone: {len(rows)} images written, {len(drops)} records dropped, {no_json} images without JSON.')
    for s in SPLITS:
        print(f'  {s:<5} {st["splits"][s]["images"]:>6}  ' +
              '  '.join(f'{k}={v}' for k, v in st['splits'][s]['scenes'].items() if v))
    for d in drops:
        print(f'  dropped {d["split"]}/{d["name"]}: {d["drop"]}')


if __name__ == '__main__':
    main()
