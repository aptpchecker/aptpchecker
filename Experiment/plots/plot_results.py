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

EXPORT_CSV = False
        
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
            runtime = float(parts[-1])
            if runtime <= args.timeout:
                total_res[name] = runtime
        # verify
        if len(parts) == 5 and parts[0] == 'unsat':
            runtime = float(parts[1])
            if runtime <= args.timeout:
                verify_res[name] = runtime
    return verify_res, total_res

def extract_marabou_results(args, names):
    verify_res, total_res = {}, {}
    result_dir = os.path.join(RESULT_DIR, 'marabou', args.benchmark)
    # assert os.path.exists(result_dir), print(f'{result_dir=}')
    for name in names:
        # total
        res_file = os.path.join(result_dir, name)
        if os.path.exists(res_file):
            parts = open(res_file).read().strip().split(' ')
            assert len(parts) == 4
            if parts[0] == 'unsat':
                print(res_file)
                runtime = float(parts[1])
                if runtime <= args.timeout:
                    total_res[name] = runtime
        else:
            error_file = res_file[:-4] + '.error'
            if not os.path.exists(error_file):
                # print(f'Missing {error_file=}')
                pass
            
        # verify
        res_file = res_file.replace('/marabou/', '/marabou_verify/')
        if os.path.exists(res_file):
            parts = open(res_file).read().strip().split(' ')
            assert len(parts) == 4
            if parts[0] == 'unsat':
                runtime = float(parts[1])
                if runtime <= args.timeout:
                    verify_res[name] = runtime
        else:
            error_file = res_file[:-4] + '.error'
            if not os.path.exists(error_file):
                # print(f'Missing {error_file=}')
                pass
            
        
    return verify_res, total_res
        
def extract_results(args, verifier, names):
    if 'neuralsat' in verifier:
        return extract_neuralsat_results(args, verifier, names)
    if verifier == 'marabou':
        return extract_marabou_results(args, names)
    raise NotImplementedError(verifier)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--csv_name', type=str, default='instances.csv')
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--format', type=str, default='png')
    parser.add_argument('--include_verify', action='store_true')
    args = parser.parse_args()   
    
    select_verifiers = ['marabou', 'neuralsat_default', 'neuralsat_X', 'neuralsat_S', 'neuralsat_SX']
    
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
        kind = 'total'
        value = results[kind]
        postfix = ''
        suffix = NAMES[verifier]
        sorted_value = list(sorted(value))
        if len(sorted_value) == 1:
            plt.scatter([0], sorted_value, color=COLORS[verifier], label=f'{suffix}{postfix}', alpha=0.6 if COLORS[verifier] != 'black' else 1.0, marker='x')
        else:
            plt.plot(sorted_value, linestyle=STYLES[kind], color=COLORS[verifier], linewidth=2.0, label=f'{suffix}{postfix}', alpha=0.6 if COLORS[verifier]!='black' else 1.0)

    if args.include_verify:
        if len(all_results['marabou']['verify']) == 1:
            plt.scatter([0], list(sorted(all_results['marabou']['verify'])), color='k', label=f'Marabou', marker='+')
        else:
            plt.plot(list(sorted(all_results['marabou']['verify'])), linestyle='--', color='k', linewidth=2.0, label='Marabou', alpha=1.0)
        plt.plot(list(sorted(all_results['neuralsat_SX']['verify'])), linestyle='--', color='r', linewidth=2.0, label='NeuralSAT', alpha=1.0)
    

    plt.xlabel('Solved problems')
    plt.ylabel('Runtimes (s)')
    
    ax.set_xlim(-5, 201)
    ax.set_yticks(range(0, args.timeout+1, 100))

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, 0.8),
        fancybox=True,
    )
    plt.tight_layout()
    filename = f"{FIGURE_DIR}/{args.type}.{args.format}"
    plt.savefig(filename, format=args.format, bbox_inches="tight")
    print(f'{filename=}')
    plt.clf()