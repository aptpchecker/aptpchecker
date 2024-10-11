import numpy as np
import os
import re

CHECKER = 'APTPchecker'

abcrown = r'$\alpha\beta$-CROWN'

NAMES = {
    'marabou': 'Marabou + MarabouChecker',
    'neuralsat_default': f'NeuralSAT + {CHECKER}',
    'neuralsat_X': f'NeuralSAT + {CHECKER} (X)',
    'neuralsat_S': f'NeuralSAT + {CHECKER} (S)',
    'neuralsat_SX': f'NeuralSAT + {CHECKER} (S+X)',
    'neuralsat_S_SX': f'NeuralSAT (S) + {CHECKER} (S+X)',
    'abcrown_default': f'{abcrown} + {CHECKER} (S+X)',
    'abcrown_babsr': f'{abcrown} (BaBSR) + {CHECKER} (S+X)',
}

ERROR_PLOT_SHAPE = [
    's', # square marker
    'o', # circle marker
    'D', # diamond marker
    'v', # triangle_down marker
    'p', # pentagon marker
    '.', # point marker
]

COLORS = {
    'neuralsat_default': 'green',
    'neuralsat_X': 'orange',
    'neuralsat_S': 'blue',
    'neuralsat_SX': 'red',
    'marabou': 'black',
}


STYLES = {
    'total': '-',
    'verify': '--'
}

BENCHMARK_NAMES = {
    'fnn_medium': 'MNIST_FNN_MEDIUM',
    'fnn_small': 'MNIST_FNN_SMALL',
    'cnn_small': 'MNIST_CNN_SMALL',
    'cnn_medium': 'MNIST_CNN_MEDIUM',
}

ROOT_DIR = os.path.dirname(__file__)
RESULT_DIR = os.path.join(os.path.dirname(ROOT_DIR), 'result')
BENCHMARK_DIR = os.path.join(os.path.dirname(ROOT_DIR), 'benchmark')
FIGURE_DIR = os.path.join(ROOT_DIR, 'figure')
os.makedirs(FIGURE_DIR, exist_ok=True)

def convert_to_scientific_notation(x):
    if x == 0:
        return '0'
    return r'$10^{%d}$' % int(np.log10(x))


def get_result_file(args):
    # benchmark 
    benchmark_path = os.path.join(BENCHMARK_DIR, args.benchmark)
    assert os.path.exists(benchmark_path), print(f'{benchmark_path=}')
    
    instances = []
    for line in open(f'{benchmark_path}/{args.csv_name}').read().strip().split('\n'):
        onnx_path, vnnlib_path, _ = line.split(',')
        onnx_path = os.path.join(benchmark_path, onnx_path)
        vnnlib_path = os.path.join(benchmark_path, vnnlib_path)
        assert os.path.exists(onnx_path)
        assert os.path.exists(vnnlib_path)
        onnx_name = os.path.splitext(os.path.basename(onnx_path))[0]
        vnnlib_name = os.path.splitext(os.path.basename(vnnlib_path))[0]
        instances.append(f'net_{onnx_name}_spec_{vnnlib_name}.res')
    return instances
        
def extract_net_name(res_name):
    # print(f'{res_name=}')
    net_name = os.path.basename(res_name).split('_spec_')[0].split('_')[-1]
    # print(f'{net_name=}')
    # exit()
    return net_name
        
def compute_box_plot(data):
    if not len(data):
        return 0., 0.
    
    median = np.median(data)
    # Compute quartiles
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    return median, q1, q3

def filter_outliers(xs):
    data = np.array(xs)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    inlier_indices = np.where((data >= lower_bound) & (data <= upper_bound))[0]
    return inlier_indices


def extract_visited(res_name):
    log_file = res_name[:-4] + '.log'
    assert os.path.exists(log_file), f'Missing {log_file=}'
    visited = re.findall(r"Visited: (\d+)", open(log_file).read())
    visited = int(visited[-1])
    return visited


def extract_sub_proofs(res_name):
    log_file = res_name[:-4] + '.log'
    assert os.path.exists(log_file), f'Missing {log_file=}'
    leaves = len(re.findall(r"Solved leaf:", open(log_file).read()))
    interiors = len(re.findall(r"Solved internal:", open(log_file).read()))
    return leaves + interiors

def extract_tree_size(res_name):
    log_file = res_name[:-4] + '.log'
    assert os.path.exists(log_file), f'Missing {log_file=}'
    leaves = re.findall(r"len\(current_proof_queue\)=(\d+)", open(log_file).read())
    if not len(leaves):
        leaves = re.findall(r"len\(proof_tree\)=(\d+)", open(log_file).read())
    # assert len(leaves), log_file
    leaves = sum([int(_) for _ in leaves])
    return leaves

def extract_constraints(res_name):
    log_file = res_name[:-4] + '.log'
    assert os.path.exists(log_file), f'Missing {log_file=}'
    pattern = r"(\d+)\s+constrs,?\s+(\d+)\s+vars"
    matches = re.search(pattern, open(log_file).read())
    if matches:
        constrs = int(matches.group(1))
        vars = int(matches.group(2))
        return constrs, vars
    return None, None

def extract_variable(res_name, return_raw=False):
    log_file = res_name[:-4] + '.log'
    assert os.path.exists(log_file), f'Missing {log_file=}'
    # visited = re.findall(r"Visited: (\d+)", open(log_file).read())
    # visited = max([int(_) for _ in visited])
    pattern = r"constrained=(\d+), free=(\d+), total=(\d+)"
    matches = re.findall(pattern, open(log_file).read())
    all_constrained, all_free, all_total = 0, 0, 0
    for match in matches:
        # print(match)
        constrained, free, total = match
        all_constrained += int(constrained)
        all_free += int(free)
        all_total += int(total)
    assert all_constrained + all_free == all_total
    assert all_total > 0, f'{res_name} {all_total=}'
    if return_raw:
        return all_constrained, all_free
    return float(all_constrained / all_total * 100), float(all_free / all_total * 100)


def extract_milp_complexity(res_name, return_raw=False):
    log_file = res_name[:-4] + '.log'
    assert os.path.exists(log_file), f'Missing {log_file=}'
    log = open(log_file).read()
    # visited = re.findall(r"Visited: (\d+)", open(log_file).read())
    # visited = max([int(_) for _ in visited])
    pattern = r"constrained=(\d+), free=(\d+), total=(\d+)"
    matches = re.findall(pattern, log)
    all_a_vars = 0
    for match in matches:
        all_a_vars += int(match[0])
    # Regular expression to match lists of numbers

    pattern = re.compile(r'Node\([^)]*, \[([-?\d, ]+)\]\)')
    matches = pattern.findall(log)
    # Convert matches to lists of integers
    paths = [list(map(int, match.split(', '))) for match in matches]

    # print(f'{all_a_vars=} {log_file=}')
    # print(len(paths))
    complexities = [all_a_vars - len(_) for _ in paths]
    # assert all([c >= 0 for c in complexities]), f'{log_file=} {all_a_vars=}'
    # print(f'{complexities=}')
    # exit()
    return complexities


def extract_milp_complexity_with_time(res_name, return_raw=False):
    log_file = res_name[:-4] + '.log'
    assert os.path.exists(log_file), f'Missing {log_file=}'
    log = open(log_file).read()
    # visited = re.findall(r"Visited: (\d+)", open(log_file).read())
    # visited = max([int(_) for _ in visited])
    pattern = r"constrained=(\d+), free=(\d+), total=(\d+)"
    matches = re.findall(pattern, log)
    all_a_vars = 0
    for match in matches:
        all_a_vars += int(match[0])
    # Regular expression to match lists of numbers

    # pattern = re.compile(r'Node\([^)]*, \[([-?\d, ]+)\]\)')
    # Single regex pattern to match Node contents and runtime
    # pattern = r"Node\([^)]*, \[([-\d, ]+)\]\) in ([\d.]+) seconds"
    # pattern = r"Node\(\w+, \[([-\d, ]+)\]\) in ([\d.]+) seconds"
    pattern = r"Node\([^,]+, \[([- \d,]+)\]\) in ([\d.]+) seconds"

    matches = re.findall(pattern, log)
    paired_data = [(list(map(int, nums.split(','))), float(runtime)) for nums, runtime in matches]
    
    complexities = []
    runtimes = []
    for pair in paired_data:
        complexities.append(all_a_vars - len(pair[0]))
        runtimes.append(pair[1])
    return complexities, runtimes



def extract_leaf_solving_time_with_variable(res_name):
    unstable, stable = extract_variable(res_name, return_raw=True)
    total = unstable + stable
    log_file = res_name[:-4] + '.log'
    # print(f'{log_file=}')
    unstable_data, stable_data = [], []
    pattern = r"Node\(.*?, \[([-\d, ]+)\]\) in (\d+\.\d+) seconds"
    for line in open(log_file).read().strip().split('\n'):
        if not line.startswith('Solved leaf: can_node'):
            continue
        # print(line)
        match = re.findall(pattern, line)
        if not match:
            continue
        branch, runtime = match[0]
            
        # print(branch.split(', '), float(runtime))
        # break
        unstable_data.append(
            (
                (unstable - len(branch.split(', '))) / total * 100, 
                float(runtime)
            )
        )
        stable_data.append(
            (
                (stable + len(branch.split(', '))) / total * 100, 
                float(runtime)
            )
        )
    return unstable_data, stable_data
    


def extract_leaf_solving_time(res_name):
    log_file = res_name[:-4] + '.log'
    assert os.path.exists(log_file), f'Missing {log_file=}'
    pattern = r"Solved leaf:.*in (\d+\.\d+) seconds"
    runtimes = re.findall(pattern, open(log_file).read())
    runtimes = [float(_) for _ in runtimes]
    return runtimes




def extract_internal_solving_time(res_name):
    log_file = res_name[:-4] + '.log'
    assert os.path.exists(log_file), f'Missing {log_file=}'
    pattern = r"Solved internal:.*in (\d+\.\d+) seconds"
    runtimes = re.findall(pattern, open(log_file).read())
    runtimes = [float(_) for _ in runtimes]
    return runtimes


def extract_stabilization_time(res_name):
    log_file = res_name[:-4] + '.log'
    assert os.path.exists(log_file), f'Missing {log_file=}'
    pattern = r"build_solver_time=(\d+.\d+)"
    runtimes = re.findall(pattern, open(log_file).read())
    runtimes = [float(_) for _ in runtimes]
    return runtimes


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)
            
            
# for file in recursive_walk('../result/neuralsat_stabilize_True_expand_1/mnist_medium_mixed_bak/'):
#     if file.endswith(".log") and 'Added' in open(file).read():
#         filename = os.path.splitext(file)[0]
#         print(filename)
#         cmd = f'cp -r {filename}.* ../result/neuralsat_stabilize_True_expand_1/mnist_medium_mixed/'
#         # os.system(f'cp -r "{filename}.*" "../result/neuralsat_stabilize_True_expand_1/mnist_medium_mixed/"')
#         os.system(cmd)
# exit()
