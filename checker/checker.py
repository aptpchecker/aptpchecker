from beartype import beartype
import gurobipy as grb
import multiprocessing
from tqdm import tqdm
import typing
import copy
import time
import os

from util.data.proof import Node, ProofTree, ProofReturnStatus
from milp.milp_solver import build_milp_solver
from util.data.objective import Objective

MULTIPROCESS_MODEL = None

ALLOWED_GUROBI_STATUS_CODES = [
    grb.GRB.OPTIMAL, 
    grb.GRB.INFEASIBLE, 
    grb.GRB.USER_OBJ_LIMIT, 
    grb.GRB.TIME_LIMIT
]

@beartype
def _solve_mip(candidate: tuple[Node, dict, float]) -> float:
    can_node, name_dict, _ = candidate
    start_solve_time = time.time()
    can_model = MULTIPROCESS_MODEL.copy()
    assert can_model.ModelSense == grb.GRB.MINIMIZE
    assert can_model.Params.BestBdStop > 0
    can_model.update()
    
    # add split constraints
    for literal in can_node.history:
        assert literal != 0
        relu_name, pre_relu_name, neuron_idx = name_dict[abs(literal)]
        pre_var = can_model.getVarByName(f"lay{pre_relu_name}_{neuron_idx}")
        relu_var = can_model.getVarByName(f"ReLU{relu_name}_{neuron_idx}")
        # print(f'\t- {pre_relu_name=}, {neuron_idx=}, {relu_name=}')
        # print(f'\t- {literal=} {pre_var=}, {relu_var=} {pre_var.lb=} {pre_var.ub=}')
        # print()
        assert pre_var is not None
        if relu_var is None: # var is None if relu is stabilized
            assert pre_var.lb * pre_var.ub >= 0, print('[!] Missing constraints')
            if (literal < 0 and pre_var.lb > 0) or (literal > 0 and pre_var.ub <= 0):
                # always unsat
                return float('inf')
        else:
            if literal > 0: # active
                can_model.addConstr(pre_var == relu_var)
            else: # inactive
                relu_var.lb = 0
                relu_var.ub = 0
        # TODO: remove all other relu_var relevant constraints
    can_model.update()
    can_model.optimize()

    print(f'Solved leaf: {can_node = } in {time.time() - start_solve_time} seconds, {can_model.NumVars=}, {can_model.NumConstrs=}')
        
    assert can_model.status in ALLOWED_GUROBI_STATUS_CODES, print(f'[!] Error: {can_model=} {can_model.status=} {can_node.history=}')
    if can_model.status == grb.GRB.USER_OBJ_LIMIT: # early stop
        return 1e-5
    if can_model.status == grb.GRB.INFEASIBLE: # infeasible
        return float('inf')
    if can_model.status == grb.GRB.TIME_LIMIT: # timeout
        return can_model.ObjBound
    return can_model.objval
    
@beartype
def mip_worker(candidate: tuple[Node, dict, float]) -> Node | None:
    node = candidate[0]
    assert node is not None
    obj_val = _solve_mip(candidate)
    if obj_val > 0:
        return node
    return None
    

class ProofChecker:
    
    @beartype
    def __init__(self, 
                 net: typing.Any, 
                 input_shape: tuple, 
                 objective: Objective, 
                 verbose: bool = False) -> None:
        
        self.net = net
        self.objective = copy.deepcopy(objective)
        self.input_shape = input_shape
        self.verbose = verbose
        self.device = 'cpu'


    @beartype
    @property
    def var_mapping(self) -> dict:
        if not hasattr(self, '_var_mapping'):
            self._var_mapping = {}
            count = 1
            for (pre_act_name, layer_size), act_name in zip(self.pre_relu_names, self.relu_names):
                for nid in range(layer_size):
                    self._var_mapping[count] = (act_name, pre_act_name, nid)
                    count += 1
        return self._var_mapping
    
    @beartype
    def _initialize_mip(self, 
                        objective: Objective, 
                        timeout: float | int = 15.0, 
                        refine: bool = False) -> grb.Model:
        
        input_lower = objective.lower_bound.view(*self.input_shape[1:])
        input_upper = objective.upper_bound.view(*self.input_shape[1:])
        
        # property: c @ output <= rhs
        c_to_use = objective.cs[None]
        rhs_to_use = objective.rhs[0]
        
        assert c_to_use.shape[1] == 1, f'Unsupported shape {c_to_use.shape=}'

        tic = time.time()
        # add MIP constraints
        solver, solver_vars, self.pre_relu_names, self.relu_names = build_milp_solver(
            net=self.net,
            input_lower=input_lower,
            input_upper=input_upper,
            c=c_to_use,
            timeout=timeout,
            name='APTPchecker',
            refine=refine,
        )    
        build_solver_time = time.time() - tic
        print(f'{build_solver_time=}')
        assert len(objective.cs) == len(solver_vars[-1]) == 1
        
        # setup objective
        solver.update()
        objective_var = solver.getVarByName(solver_vars[-1][0].varName) - rhs_to_use
        solver.setObjective(objective_var, grb.GRB.MINIMIZE)
        solver.update()
        return solver
    
    
    @beartype
    def build_mip(self, 
                  objective: Objective, 
                  timeout: float | int, 
                  timeout_per_node: float | int, 
                  refine: bool) -> grb.Model:
        
        # step 1: build core model without specific objective
        print(f'\n############ Build MIP ############\n')
        mip_model = self._initialize_mip(
            objective=objective, 
            timeout=timeout_per_node, 
            refine=refine,
        )
        mip_model.setParam('TimeLimit', timeout)
        mip_model.setParam('OutputFlag', self.verbose)
        mip_model.update()
        print(f'{mip_model=}')

        # step 2: set specific objective
        return mip_model
    
    
    @beartype
    def prove_nodes(self, 
                    proof: list[list], 
                    batch: int, 
                    timeout: float | int, 
                    expand: bool = False) -> str:
        
        print(f'\n############ Check Proof ############\n')
        # step 1: proof tree
        proof_tree = ProofTree(proofs=proof)
        expand_factor = 2.0 if expand else 1.0
        
        # step 2: prove nodes
        progress_bar = tqdm(total=len(proof_tree), desc=f"Processing proof")
        while len(proof_tree):
            if time.time() - self.start_time > timeout:
                return ProofReturnStatus.TIMEOUT 
            
            # get nodes to be proved
            processing_nodes = proof_tree.get(batch)
            
            # gather necessary information
            candidates = [(node, self.var_mapping, expand_factor) for node in processing_nodes]
            print(f'Proving {len(candidates)=}')
            
            # run proofs in parallel
            max_worker = min(len(candidates), os.cpu_count() // 2)
            if max_worker > 1:
                with multiprocessing.Pool(max_worker) as pool:
                    results = pool.map(mip_worker, candidates, chunksize=1)
            else:
                results = [mip_worker(c) for c in candidates]

            # filter proved nodes 
            processed = len(proof_tree)
            for solved_node in results:
                if solved_node is not None:
                    # remove proved leaf
                    proof_tree.filter(solved_node)
                else:
                    # cannot prove a leaf
                    return ProofReturnStatus.UNCERTIFIED # unproved
                
            # print(f'\t- Remaining: {len(proof_tree)}')
            processed -= len(proof_tree)
            progress_bar.update(processed)
        
        return ProofReturnStatus.CERTIFIED # proved
    
    
    @beartype
    def prove(self, 
              proof: list[list], 
              batch: int = 1, 
              timeout: float | int = 3600.0, 
              timeout_per_node: float | int = 15.0,
              refine: bool = False, 
              expand: bool = False) -> str:
        
        print(f"Settings: {refine=} {expand=} {batch=}")

        self.start_time = time.time()
        global MULTIPROCESS_MODEL
        
        # step 1: build mip
        mip_model = self.build_mip(
            objective=self.objective, 
            timeout=timeout, 
            timeout_per_node=timeout_per_node, 
            refine=refine,
        )

        # check timeout
        if time.time() - self.start_time > timeout:
            return ProofReturnStatus.TIMEOUT 
        
        MULTIPROCESS_MODEL = mip_model
        
        # step 2: prove nodes
        status = self.prove_nodes(
            proof=proof,
            batch=batch, 
            timeout=timeout,
            expand=expand,
        )
        
        MULTIPROCESS_MODEL = None
        
        return status
        