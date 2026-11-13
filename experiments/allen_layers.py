"""
Anatomical reference for the laminar profiles of the region contributions (Phase 2 G): how densely each source
area's axons project into each layer of VISp, from the Allen Mouse Brain Connectivity Atlas (public API).

For every source structure of SOURCES (CCF acronyms for the region groups of experiments/region_ablation.py),
the injection experiments with that primary injection structure whose injection does not reach VISp (local label
would swamp the projection), from any Cre line but the Ai75 reporter series (pan_excitatory: wild type or
Emx1-IRES-Cre, all excitatory projection neurons; the other lines label subsets, e.g. one layer); per
experiment, the projection density (sum_projection_pixels / sum_pixels) in each VISp layer of the hemisphere
of the injection, and per compartment of summarize.region_depth (L2/3 = VISp1 + VISp2/3, L4, L5/6 = VISp5 +
VISp6a + VISp6b).

usage (from the repo root; needs internet, no session data):
    python -m experiments.allen_layers [--out tables/allen_visp_layers.csv]

Saves one row per experiment: experiment, source, group, line, pan_excitatory, hemisphere, injection_volume, the density per
layer (VISp1 ... VISp6b) and per compartment (L2/3, L4, L5/6).
"""
import argparse
import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

import paths

API = 'https://api.brain-map.org/api/v2/data/query.json?criteria='
SOURCES = {
    'visual_cortex_other': ['VISl', 'VISal', 'VISrl', 'VISam', 'VISpm', 'VISli', 'VISpl', 'VISpor'],
    'thalamus': ['LGd', 'LP', 'MG', 'POL', 'PIL', 'PoT'],
    'hippocampus': ['CA1', 'CA3', 'DG', 'ProS', 'SUB', 'POST'],
    'midbrain': ['APN', 'NOT', 'MRN', 'SCs', 'SCm', 'SGN', 'ZI'],
}
LAYERS = {'VISp1': 593, 'VISp2/3': 821, 'VISp4': 721, 'VISp5': 778, 'VISp6a': 33, 'VISp6b': 305}
LAYER_COMPARTMENTS = {'L2/3': ['VISp1', 'VISp2/3'], 'L4': ['VISp4'], 'L5/6': ['VISp5', 'VISp6a', 'VISp6b']}
PAN_EXCITATORY = ('', 'Emx1-IRES-Cre')
EXCLUDED_LINES = ('Ai75(RCL-nT)',)
MIDLINE_UM = 5700                     # CCF z of the midline: injections at smaller z are in the left hemisphere (id 1)


def query(criteria, retries=6):
    url = API + urllib.parse.quote(criteria, safe="$:,[]'=")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                out = json.load(r)
            if out.get('success'):
                return out['msg']
        except OSError:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f'Allen API query failed: {criteria}')


def experiments(source):
    """Injection experiments with source as primary injection structure, not reaching VISp."""
    rows = query(f'service::mouse_connectivity_injection_structure[injection_structures$eq{source}]'
                 '[primary_structure_only$eqtrue]')
    return [r for r in rows if (r.get('transgenic-line') or '') not in EXCLUDED_LINES
            and 'VISp' not in [s['abbreviation'] for s in r.get('injection-structures', [])]]


def layer_densities(experiment, hemisphere):
    ids = ','.join(map(str, LAYERS.values()))
    rows = query(f'model::ProjectionStructureUnionize,rma::criteria,[section_data_set_id$eq{experiment}],'
                 f'[is_injection$eqfalse],[hemisphere_id$eq{hemisphere}],[structure_id$in{ids}]')
    pixels = {r['structure_id']: (r['sum_projection_pixels'], r['sum_pixels']) for r in rows}
    out = {}
    for name, sid in LAYERS.items():
        out[name] = pixels[sid][0] / pixels[sid][1] if sid in pixels and pixels[sid][1] else np.nan
    for comp, layers in LAYER_COMPARTMENTS.items():
        num = sum(pixels[LAYERS[l]][0] for l in layers if LAYERS[l] in pixels)
        den = sum(pixels[LAYERS[l]][1] for l in layers if LAYERS[l] in pixels)
        out[comp] = num / den if den else np.nan
    return out


def run():
    rows = []
    for group, sources in SOURCES.items():
        for source in sources:
            found = experiments(source)
            print(f'{group} {source}: {len(found)} experiments')
            for e in found:
                hemisphere = 1 if e['injection-coordinates'][2] < MIDLINE_UM else 2
                rows.append(dict(experiment=e['id'], source=source, group=group, line=e.get('transgenic-line') or 'wild type',
                                 pan_excitatory=(e.get('transgenic-line') or '') in PAN_EXCITATORY,
                                 hemisphere=hemisphere, injection_volume=e.get('injection-volume'),
                                 **layer_densities(e['id'], hemisphere)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=str(paths.REPO_DIR / 'tables' / 'allen_visp_layers.csv'))
    args = parser.parse_args()
    table = run()
    table.to_csv(args.out, index=False)
    print(f'saved {len(table)} experiments to {args.out}')


if __name__ == '__main__':
    main()
