import numpy as np

def compute_usefulness(action, item):
    score_values = item["coin_values"]
    score = score_values[action]
    trajectory = item["trajectories"][action]
    max_score_per_trajectory = [min(i for i in score_values if i > 0),max(score_values)]
    return score//max_score_per_trajectory[trajectory]

def compute_entropy(p):
    '''Computes the Shannon entropy of a two state system, where state 0 will 
    be chosen with probability p, and 1 with probability 1-p. 
    
    Accepts any shape numpy array or other numerical type such as float or int
    Uses safe_log2 to handle values of 0 and 1 without numerical issues. 
    These result in entropy of 0, which is the correct limit.'''
    return - p*safe_log2(p) - (1-p)*safe_log2(1-p)


def safe_log2(p):
    '''Reproduces behavior of np.log2, but for zeros returns -1e6 instead of -np.inf'''
    # Handle lists
    if type(p) is list:
        p = np.array(p)
    # Handle scalar types
    if type(p) in [int, float, np.float16, np.float32, np.float64]:
        if p == 0:
            return -1e6
        else:
            return np.log2(p)
    # Handle arrays
    elif type(p) is np.ndarray:
        p_new = np.empty_like(p)
        p_new[p!=0] = np.log2(p[p!=0])
        p_new[p==0] = -1e6
        return p_new
    # Unknown type
    else:
        raise ValueError('Must be numeric type: int, float, list, numpy.ndarray')