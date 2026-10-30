"""
Build tables/sessions.csv: mouse, genotype and session metadata of the analysed sessions
(cluster/sessions.txt), for grouped cross-validation and for nesting sessions within mice.

usage (from the repo root):
    python -m experiments.session_table [ecephys_sessions.csv]

Reads the VBN project metadata table from the given file, or from the public S3 bucket.
"""
import sys

import pandas as pd

import paths

METADATA_URL = ('https://visual-behavior-neuropixels-data.s3.us-west-2.amazonaws.com/'
                'visual-behavior-neuropixels/project_metadata/ecephys_sessions.csv')
COLUMNS = ['ecephys_session_id', 'behavior_session_id', 'mouse_id', 'genotype', 'sex', 'age_in_days',
           'session_number', 'experience_level', 'image_set', 'prior_exposures_to_image_set', 'equipment_name']
OUTPUT = paths.REPO_DIR / 'tables' / 'sessions.csv'


def build(metadata, session_ids):
    table = metadata.loc[metadata.ecephys_session_id.isin(session_ids), COLUMNS]
    missing = set(session_ids) - set(table.ecephys_session_id)
    if missing:
        raise ValueError(f'sessions not in the metadata table: {sorted(missing)}')
    table = table.rename(columns={'ecephys_session_id': 'session_id'})
    # Cre line of the optotagged interneurons (wt for wild-type mice)
    table.insert(4, 'cre_line', table.genotype.str.split(';').str[0].str.split('-').str[0].replace('wt/wt', 'wt'))
    return table.sort_values(['mouse_id', 'session_number']).reset_index(drop=True)


def main():
    metadata = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else METADATA_URL)
    ids = [int(line) for line in open(paths.REPO_DIR / 'cluster' / 'sessions.txt') if line.strip()]
    table = build(metadata, ids)
    table.to_csv(OUTPUT, index=False)
    print(f'{len(table)} sessions from {table.mouse_id.nunique()} mice -> {OUTPUT}')


if __name__ == '__main__':
    main()
