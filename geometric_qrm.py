import itertools
import numpy as np


#####################
# CODE CONSTRUCTION #
#####################
# -> _logical_qubits (list of logical qubits)
# -> indexed_logical_qubits (qb generator -> int)
# -> _logical_circuit (list of CnZs)
# -> compute_logic (_logical_circuit but with indices)

def _logical_qubits(K: set, q: int, r: int):
    """Given a hypercube with basis $K$, enumerate all possible logical qubits

    Args:
        K (set): Hypercube basis; may be a restriction within a larger hypercube
        q (int): Codimension of the X stabilizers
        r (int): r+1 is the dimension of the Z stabilizers
    """
    assert r >= q, "Does not form a valid CSS code"
    
    # Generate index set of logical operators restricted to K
    logical_operators = set()
    for Ksize in range(q + 1, r + 1):
        logs = itertools.combinations(K, Ksize)
        logical_operators.update(logs)
    
    return logical_operators

def indexed_logical_qubits(m: int, q: int, r: int, sort=True):
    assert r >= q, "Does not form a valid CSS code"
    all_logicals = _logical_qubits(set(range(1, m + 1)), q, r)
    
    if sort:
        all_logicals = sorted(all_logicals, key=lambda x: (len(x), x))
    
    logical_dict = dict(zip(all_logicals, range(1, len(all_logicals) + 1)))
    
    return logical_dict


def _logical_circuit(k: int, K: set, q: int, r: int):
    """Describes the logical circuit affected by a $k$th level physical operator applied to a hypercube of basis $K$ on a QRM with parameters $q$, $r$

    Args:
        k (int): Clifford level of physical operator applied; 2 for T gates
        K (set): Set of bases, e.g. {1, 2, 3}
        q (int): q parameter 
        r (int): r parameter

    Returns:
        set: Set of C^n Z circuits that are applied
    """
    assert r >= q, "Does not form a valid CSS code"
    
    if len(K) < q + k * r + 1:
        return False
    
    assert len(K) >= q + k * r + 1, "Will not be a logical operator"
    
    K = set(list(K))
    
    if len(K) > (k + 1) * r:
        # logical identity
        return set()
    
    logical_operators = _logical_qubits(K, q, r)
    
    # Check which of these combinations of combinations are valid
    circuit = set()
    all_combinations = list(itertools.combinations(logical_operators, k + 1))
    
    for combo in all_combinations:
        # get union of all sets and check if its a covering
        all_idxs = set().union(*combo)
        if all_idxs == K:
            circuit.add(combo)
    
    return circuit


def compute_logic(k, K, m, q, r):
    assert m > 0 
    assert q < r
    assert len(K) <= m
    # K need to be a set, k is an int
    
    map_logqubit_int = indexed_logical_qubits(m, q, r)
    logical_circuit = _logical_circuit(k, K, q, r)
    
    if not logical_circuit:
        return False
    
    # map these logical circuits now to indexed form
    
    out = []
    for targs in logical_circuit:
        mapped_targs = [map_logqubit_int[qid] for qid in targs]
        out.append(mapped_targs.copy())
    
    return out

######################
# CODE VISUALIZATION #
######################
# -> mat_to_ir (converts a matrix to a Qcirc IR)
# -> to_qcirc (prints the Qcirc IR)
# -> make_code (combines the above to make a code)

def mat_to_ir(mat):
    # input: len(circuit) x num qubits
    # output: len(circuit) x num qubits
    # we want an IR that allows us to go to Qcirc easily
    # out_{ij} = 0 --> just use \qw
    # out_{ij} = k --> \ctrl{k}
    # out_{ij} = -1 --> \ctrl{0}
    
    out = np.zeros_like(mat)
    for r_idx, row in enumerate(mat):
        # get all the nonzero idxs
        nonzero_idxs = np.nonzero(row)[0]
        
        assert len(nonzero_idxs) > 1, "No CZs"
        
        for c_idx, val in enumerate(nonzero_idxs[:-1]):
            out[r_idx, val] = nonzero_idxs[c_idx + 1] - val
        
        out[r_idx, nonzero_idxs[-1]] = -1
    
    return out

def to_qcirc(mat, qb_labels=None):
    out = ""
    qb_x_time = mat.T
    
    all_qbs = list(qb_labels.items())
    
    out += r"""\documentclass{standalone}
    \usepackage{adjustbox}
    \usepackage{quantikz}
    %\usetikzlibrary{...}% tikz package already loaded by 'tikz' option
    \begin{document}
    """
    out += r"\begin{adjustbox}{width=0.8 \paperheight} \begin{quantikz}"
    # [font=\tiny]
    
    for (qb_idx, qb) in enumerate(qb_x_time):
        # add qubit annotation
        if qb_labels:
            out += "\\lstick{$\\overline{%s}$} = %s &" %(qb_idx + 1, all_qbs[qb_idx][0])
        
        for (time_idx, time) in enumerate(qb):
            if time == 0:
                out += "\\qw &"
            elif time == -1:
                out += "\\ctrl{} &"
            else:
                out += "\\ctrl{%d} &" % time
            out += " "
        out += "\\\\ \n"
    
    out += "\\end{quantikz} \\end{adjustbox}\n"
    out += r"\end{document}"
    return out

def make_code(k, KSize, m, q, r,):
    K = range(1, KSize + 1)
    circ = compute_logic(k, K, m, q, r)
    if not circ:
        # catches cases where the logical circuit is either trivial or invalid
        return False
    
    # if len(circ) == 0:
    #     return False
    
    qubits = indexed_logical_qubits(m, q, r)
    num_qubits = len(qubits)
    
    circ_mat = np.zeros((len(circ), num_qubits))
    for rowidx, row in enumerate(circ):
        for idx in row:
            circ_mat[rowidx, idx - 1] = 1
    
    # sort
    sorted_circ_mat = circ_mat.tolist()
    sorted_circ_mat.sort(key=lambda x: tuple(x))
    sorted_circ_mat = sorted_circ_mat[::-1]
    
    qcirc_mat = mat_to_ir(sorted_circ_mat)
    
    return to_qcirc(qcirc_mat, qb_labels=qubits)