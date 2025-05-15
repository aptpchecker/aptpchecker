
class ProofReturnStatus:

    UNKNOWN = 'unknown'
    TIMEOUT = 'timeout'
    CERTIFIED = 'certified'
    UNCERTIFIED = 'uncertified'


class Node:
    
    def __init__(self, history, name='node') -> None:
        self.history = history
        self.name = name
        
    def __len__(self):
        return len(self.history)
    
    def __repr__(self):
        return f'Node({self.name}, {self.history})'


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

    def __repr__(self):
        lists = []
        if len(self) > 10:
            lists += [str(n) for n in self.queue[:5]]
            lists += ['...']
            lists += [str(n) for n in self.queue[-5:]]
        else:
            lists += [str(n) for n in self.queue]
        return '\nQueue(\n\t' + '\n\t'.join(lists) + '\n)'
            
            