# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse
import os

from utils import *

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rc('axes', labelsize=14) 

        
def extract_neuralsat_results(args, verifier, names):
    verify_res, total_res = {}, {}
    result_dir = os.path.join(RESULT_DIR, verifier, args.benchmark)
    assert os.path.exists(result_dir), f'{result_dir=}'
    for name in names:
        res_file = os.path.join(result_dir, name)
        if not os.path.exists(res_file):
            print(f'Missing {res_file=}')
            continue
        parts = open(res_file).read().strip().split(',')
        # total
        if len(parts) == 5 and parts[2] == 'certified':
            if float(parts[-1]) <= args.timeout:
                total_res[name] = float(parts[-1])
            else:
                print(f"Uncerified {res_file=}")

        # verify
        if len(parts) == 5 and parts[0] == 'unsat':
            if float(parts[1]) <= args.timeout:
                verify_res[name] = float(parts[1])

    return verify_res, total_res

def extract_abcrown_results(args, names):
    verify_res, total_res = {}, {}
    result_dir = os.path.join(RESULT_DIR, verifier, args.benchmark)
    for name in names:
        res_file = os.path.join(result_dir, name)
        if not os.path.exists(res_file):
            continue
        parts = open(res_file).read().strip().split(',')
        # print(f'{verifier=} {parts=}')
        
        # total
        if len(parts) == 4 and parts[1] == 'certified':
            if float(parts[-1]) <= args.timeout:
                total_res[name] = float(parts[-1])
                
        # verify
        if len(parts) == 4 and parts[0] == 'unsat':
            if float(parts[-1]) - float(parts[-2]) <= args.timeout:
                verify_res[name] = float(parts[-1]) - float(parts[-2])

    return verify_res, total_res
        
def extract_results(args, verifier, names):
    if 'neuralsat' in verifier:
        return extract_neuralsat_results(args, verifier, names)
    if 'abcrown' in verifier:
        return extract_abcrown_results(args, names)
    raise NotImplementedError(verifier)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--csv_name', type=str, default='instances.csv')
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--format', type=str, default='png')
    parser.add_argument('--include_verify', action='store_true')
    args = parser.parse_args()   
        
        
    COLORS = {
        'abcrown_default': 'blue',
        'abcrown_babsr': 'magenta',
        'neuralsat_SX': 'green',
        'neuralsat_S_SX': 'red',
    }


    # extract result file basename
    select_verifiers = [
        'abcrown_default',
        'abcrown_babsr',
        'neuralsat_SX',
        'neuralsat_S_SX',
    ]
    
    # extract runtimes
    all_results = {
        v: {
            'verify': [],
            'total': [],
        } for v in select_verifiers
    }

    for benchmark in BENCHMARK_NAMES:
        if not benchmark.startswith(args.type):
            continue
        print(f'{benchmark=}')
        args.benchmark = benchmark
        basenames = get_result_file(args)
        
        for verifier in select_verifiers:
            verify_res, total_res = extract_results(args, verifier, basenames)
            print(f'\t- {verifier=}, {len(verify_res)=}, {len(total_res)=}')
            all_results[verifier]['verify'].extend(list(verify_res.values()))
            all_results[verifier]['total'].extend(list(total_res.values()))
        
    for v, res in all_results.items():
        print(v, len(res['verify']), len(res['total']))
    
    # plot
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    for verifier, results in all_results.items():
        plt.plot(list(sorted(results['total'])), linestyle=STYLES['total'], color=COLORS[verifier], linewidth=2.0, label=f'{NAMES[verifier]}', alpha=0.6 if COLORS[verifier]!='black' else 1.0)
        plt.plot(list(sorted(results['verify'])), linestyle=STYLES['verify'], color=COLORS[verifier], linewidth=2.0, label=f'{NAMES[verifier].split(" + ")[0]}', alpha=0.6 if COLORS[verifier]!='black' else 1.0)

        
    plt.xlabel('Solved problems')
    plt.ylabel('Runtimes (s)')
    plt.title(args.type.upper())
    
    ax.set_yticks(range(0, args.timeout+1, 100))

    ax.legend(
        loc="upper left",
        fancybox=True,
    )
    plt.tight_layout()
    prefix = args.type if len(args.type) else "ablation"
    filename = f"{FIGURE_DIR}/{prefix}.{args.format}"
    plt.savefig(filename, format=args.format, bbox_inches="tight")
    print(f'{filename=}')
    plt.clf()