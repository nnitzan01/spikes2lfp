import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix

def save_relative_attribution(output_dir, session_id):
    if os.path.exists('./tables/units_info.csv'):
        print("Loading units_info.csv from the repo.")
        df = pd.read_csv('./tables/units_info.csv')
    else:
        print("units_info.csv not found in the repo.")
        exit(1)

    bands = [(0.5, 4), (4, 8), (8, 12), (12, 25), (25, 50), (50, 100), (100, 200), (200, 400)]
    output_dir_attrs = Path(output_dir / 'spikes2lfp' / 'attrs' / str(session_id))
    mean_attribution_raw = np.load(output_dir_attrs / f'attribution_scores_mean.npy')
    num_channels = mean_attribution_raw.shape[0] 
    mean_attribution_relative = np.zeros_like(mean_attribution_raw) # channels x bands+1 x neurons
    del mean_attribution_raw
    
    for bandi in range(len(bands)+1):
        for chani in range(num_channels):
            sparse_attr = np.load(output_dir_attrs / f'attribution_scores_chan{chani}_band{bandi}.npz')
            attrs = coo_matrix((sparse_attr['data'], (sparse_attr['row'], sparse_attr['col'])), shape=sparse_attr['shape']).toarray()
            total_attr = np.nansum(np.abs(attrs), axis=1, keepdims=True)
            attrs = attrs / (total_attr + 1e-10)  # avoid division by zero
            mean_attribution_relative[chani, bandi, :] = np.mean(attrs, axis=0)
            del attrs, sparse_attr, total_attr
            
    filename = Path(output_dir_attrs / f'attribution_scores_relative_mean.npy')
    np.save(filename, mean_attribution_relative)
    print("Relative attribution scores are saved in: ", output_dir_attrs)

def main(args):
    dir = Path(args.dir)
    if not dir.exists():
        print(f"Directory {dir} does not exist.")
        sys.exit(1)
    session_id = int(os.path.basename(dir))
    root_dir = dir.parent.parent.parent
    print(f"Root dir is set to: {root_dir}", flush=True)
    print(f"Session ID is set to: {session_id}", flush=True)    
    save_relative_attribution(root_dir, session_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Path to the root dir: ")
    args = parser.parse_args()
    main(args)