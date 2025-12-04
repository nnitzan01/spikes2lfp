import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix
import warnings

def save_instantaneous_predictive_influence(output_dir, session_id):
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
    mean_attribution_ipi = np.zeros_like(mean_attribution_raw) # channels x bands+1 x neurons
    del mean_attribution_raw
    
    EPSILON = 1e-10
    
    for bandi in range(len(bands)+1):
        for chani in range(num_channels):
            sparse_attr = np.load(output_dir_attrs / f'attribution_scores_chan{chani}_band{bandi}.npz')
            
            # 1. Load the full, raw attribution matrix
            attrs_raw = coo_matrix((sparse_attr['data'], (sparse_attr['row'], sparse_attr['col'])), shape=sparse_attr['shape']).toarray()
            # 2. Calculate Denominator (Total Network Load - L1 Norm)
            total_attr = np.nansum(np.abs(attrs_raw), axis=1, keepdims=True)
            # Use np.divide and np.nan_to_num to handle the division result cleanly, 
            # converting inf/nan (from 1/0) back to 0.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # Calculate the normalized matrix, adding epsilon to the denominator for safety
                attrs_normalized = np.divide(attrs_raw, total_attr, out=np.zeros_like(attrs_raw), where=total_attr > 1e-10)
                
            attrs_normalized[np.abs(attrs_raw) < EPSILON] = np.nan 
            ipi_result = np.nanmean(attrs_normalized, axis=0)
            ipi_result = np.nan_to_num(ipi_result, nan=0.0) 
            mean_attribution_ipi[chani, bandi, :] = ipi_result
    
    # Memory cleanup
    del attrs_raw, sparse_attr, total_attr, attrs_normalized, ipi_result
            
    filename = Path(output_dir_attrs / f'attribution_scores_ipi_mean.npy')
    np.save(filename, mean_attribution_ipi)
    print("Relative IPI attribution scores are saved in: ", output_dir_attrs)

def main(args):
    dir = Path(args.dir)
    if not dir.exists():
        print(f"Directory {dir} does not exist.")
        sys.exit(1)
    session_id = int(os.path.basename(dir))
    root_dir = dir.parent.parent.parent
    print(f"Root dir is set to: {root_dir}", flush=True)
    print(f"Session ID is set to: {session_id}", flush=True)    
    save_instantaneous_predictive_influence(root_dir, session_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Path to the root dir: ")
    args = parser.parse_args()
    main(args)