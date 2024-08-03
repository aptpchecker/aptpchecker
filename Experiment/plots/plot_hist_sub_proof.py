# import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse
import re
import os

TIMEOUT = None

from utils import *
            
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rc('axes', labelsize=14) 

def extract_neuralsat_results(args, names):
    results = {
        'solved': [],
        'unsolved': []              
    }
    result_dir = os.path.join(RESULT_DIR, args.verifier, args.benchmark)
    assert os.path.exists(result_dir), print(f'{result_dir=}')
    for name in names:
        res_file = os.path.join(result_dir, name)
        if not os.path.exists(res_file):
            print(f'Missing {res_file=}')
            continue
        
        parts = open(res_file).read().strip().split(',')
        # total
        if len(parts) == 5:
            proofs = extract_sub_proofs(res_file)
            runtime = float(parts[-1])
            if parts[2] == 'certified':
                if runtime <= TIMEOUT:
                    results['solved'].append(proofs)
                else:
                    results['unsolved'].append(proofs)
            else:
                assert parts[2] != 'uncertified'
                results['unsolved'].append(proofs)
        else:
            # print(res_file)
            pass
            
    return results


def extract_abcrown_results(args, names):
    results = {
        'solved': [],
        'unsolved': []              
    }
    result_dir = os.path.join(RESULT_DIR, args.verifier, args.benchmark)
    assert os.path.exists(result_dir), print(f'{result_dir=}')
    for name in names:
        res_file = os.path.join(result_dir, name)
        if not os.path.exists(res_file):
            print(f'Missing {res_file=}')
            continue
        
        parts = open(res_file).read().strip().split(',')
        # total
        proofs = extract_sub_proofs(res_file)
        if len(parts) == 4:
            runtime = float(parts[-1])
            if parts[1] == 'certified':
                if runtime <= TIMEOUT:
                    results['solved'].append(proofs)
                else:
                    results['unsolved'].append(proofs)
                    # print(res_file)
            else:
                assert parts[2] != 'uncertified'
                results['unsolved'].append(proofs)
        else:
            results['unsolved'].append(proofs)
            pass
            
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verifier', type=str, required=True)
    parser.add_argument('--csv_name', type=str, default='instances.csv')
    parser.add_argument('--format', type=str, default='png')
    parser.add_argument('--n_bins', type=int, default=10)
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--timeout', type=int, default=1000)
    args = parser.parse_args()   
    
    TIMEOUT = args.timeout
    
    # extract result file basename
    results = {'cnn': {'solved': [], 'unsolved': []}, 'fnn': {'solved': [], 'unsolved': []}}

    for benchmark in BENCHMARK_NAMES:
        print(f'\t- {benchmark=}')
        args.benchmark = benchmark
        basenames = get_result_file(args)
        
        if 'neuralsat' in args.verifier:
            res = extract_neuralsat_results(args, basenames)
        elif 'abcrown' in args.verifier:
            res = extract_abcrown_results(args, basenames)
        else:
            raise ValueError(args.verifier)
        
        
        for key in results:
            if benchmark.startswith(key):
                results[key]['solved'].extend(res['solved'])
                results[key]['unsolved'].extend(res['unsolved'])
                

    
    x1 = np.array(results['fnn']['solved'])
    x2 = np.array(results['fnn']['unsolved'])
    x3 = np.array(results['cnn']['solved'])
    x4 = np.array(results['cnn']['unsolved'])
    xs = [x1, x2, x3, x4]

    xs_debug = np.concatenate(xs)
    print(xs_debug.shape)
    print(f'{args.verifier}: mean={np.mean(xs_debug):.02f}, std={np.std(xs_debug):.02f}, median={np.median(xs_debug):.02f}, min={np.min(xs_debug)}, max={np.max(xs_debug)}')

    labels = ['FNN (solved)', 'FNN (unsolved)', 'CNN (solved)', 'CNN (unsolved)']
    colors = ['r', 'r', 'b', 'b']
    hatch_patterns = ['', 'xxxxx', '', 'xxxxx'] 
    
    if args.compact:
        fig, ax = plt.subplots(1, 1, figsize=(2.3, 6))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
    s_idx = 0
    e_idx = 4
    _, _, patches = ax.hist(
        xs[s_idx:e_idx], 
        args.n_bins, 
        stacked=True, 
        density=False, 
        histtype='bar', 
        color=colors[s_idx:e_idx], 
        alpha=0.5, 
        label=labels[s_idx:e_idx],
    )

    for i, patch_set in enumerate(patches):
        for patch in patch_set:
            patch_copy = plt.Rectangle(patch.get_xy(), patch.get_width(), patch.get_height(), 
                                    hatch=hatch_patterns[i], fill=False, edgecolor='k', linewidth=0.5)
            ax.add_patch(patch_copy)
            
    if not args.compact:
        legend_patches = [Patch(facecolor=colors[i], edgecolor='k', alpha=0.5, hatch=hatch_patterns[i], label=labels[i]) for i in range(len(colors))]
        ax.legend(handles=legend_patches, handleheight=1, handlelength=3, loc='upper right')

    plt.yscale('log')
    plt.ylim(top=1e3)
    if args.compact:
        plt.xlim(-100, 2000)
    plt.tight_layout()
    plt.title(NAMES[args.verifier].split(' + ')[1])
    
    plt.xlabel('Number of sub-proofs')
    plt.ylabel('Counts')
    
    folder = os.path.join(FIGURE_DIR, args.verifier)
    os.makedirs(folder, exist_ok=True)
    
    filename = f'{folder}/SUB_PROOFS.{args.format}'
    plt.savefig(filename, format=args.format, bbox_inches="tight")
    print(f'{filename=}')
    plt.clf()