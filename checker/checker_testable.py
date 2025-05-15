from typing import Any, List
import copy

class ProofReturnStatus:

    CERTIFIED = 'certified'
    UNCERTIFIED = 'uncertified'

class Node:
    
    def __init__(self, history, name='node') -> None:
        self.history = history
        self.name = name
        
    def __len__(self):
        return len(self.history)
    
class ProofTree:
    
    def __init__(self, proofs: list) -> None:
        histories = proofs if len(proofs) else [[]]
        self.queue = [Node(history=h, name=f'node_{i}') for i, h in enumerate(histories)]
        
    def get(self, batch):
        indices = range(min(len(self), batch))
        return [self.queue[idx] for idx in indices]
    
    def add(self, node: Node):
        self.queue.append(node)
    
    def filter(self, node: Node):
        "Filter out solved nodes"
        new_queue = [n for n in self.queue if not n == node]
        self.queue = new_queue
    
    def __len__(self):
        return len(self.queue)


def mip_worker(candidate: tuple[Node, dict, float]) -> Node | None:
    """
    Mock a realistic MIP feasibility check.
    A node is verified if the sum of its history is non-negative.
    """
    node = candidate[0]
    assert node is not None
    if sum(node.history) >= 0:
        return node
    return None

class ProofChecker:

    def __init__(self, net: Any, input_shape: Any, objective: Any, verbose: bool = False) -> None:
        self.net = net
        self.objective = copy.deepcopy(objective)
        self.input_shape = input_shape
        self.verbose = verbose
        self.device = 'cpu'


    def build_mip(self) -> Any:
        mip_model = "mock_mip_solver"
        return mip_model

    def prove_nodes(self, proof: List[List[int]], batch: int) -> str:
        
        # step 1: proof tree
        proof_tree = ProofTree(proofs=proof)
        # print(f'\t- {proof=}')
        # step 2: prove nodes
        while len(proof_tree):
            # get nodes to be proved
            processing_nodes = proof_tree.get(batch)
            
            # gather necessary information
            candidates = [(node, {}, 1.0) for node in processing_nodes]
            
            # run proofs
            results = [mip_worker(c) for c in candidates]
            for solved_node in results:
                if solved_node is not None:
                    # remove proved leaf
                    proof_tree.filter(solved_node)
                else:
                    # cannot prove a leaf
                    return ProofReturnStatus.UNCERTIFIED
            
        return ProofReturnStatus.CERTIFIED

    def prove(self, proof: List[List[int]]) -> str:
        """
        pre: isinstance(proof, list)
        pre: all(isinstance(p, list) for p in proof)
        post: _ in {ProofReturnStatus.CERTIFIED, ProofReturnStatus.UNCERTIFIED}
        post: (_ == ProofReturnStatus.UNCERTIFIED) == (any(sum(n) < 0 for n in proof))
        post: (_ == ProofReturnStatus.CERTIFIED) == (all(sum(n) >= 0 for n in proof))
        """
        
        # step 1: build mip
        mip_model = self.build_mip()
        
        # step 2: prove nodes
        status = self.prove_nodes(proof=proof, batch=1)
        
        return status

if __name__ == "__main__":
    checker = ProofChecker(None, None, None)
    print(checker.prove([[1], [2, -1]]))  # CERTIFIED
    print(checker.prove([[], [-1]]))  # UNCERTIFIED
    print(checker.prove([[1, 2, -10], [3, -3, -2]])) # UNCERTIFIED (sum is -7 and -2)
    