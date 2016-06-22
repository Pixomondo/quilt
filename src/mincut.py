
import numpy as np


def mincut(matrix, direction=None):
    """
    Computes the minimum cut from one edge of the matrix to the other
    
    Inputs:
       matrix:  Evaluations of 2d function to be cut along local minima
       direction: 0 = vertical cut, 1 = horizontal cut
    
    Outputs:
       cut: Matrix containing entries indicating side
           -1: left (or top) side of cut
            0: along the cut
            1: right (or bottom) side of cut 
    """
    if direction:
        matrix = np.transpose(matrix)

    # 1) define COST
    # Allocate the current cost array, initialize it with matrix values
    cost = matrix
    # Starting with the second array, compute the path costs until the end
    for i in range(1, cost.shape[0]):
        for j in range(cost.shape[1]):
            if j == 0:
                cost[i, j] += min(cost[i-1, 0], cost[i-1, 1])
            elif j == cost.shape[1]-1:
                cost[i, j] += min(cost[i-1, -2], cost[i-1, -1])
            else:
                cost[i, j] += min([cost[i-1, j-1], cost[i-1, j], cost[i-1, j+1]])

    # 2) find CUT
    # Backtrace to find the cut
    cut = np.zeros(matrix.shape)

    idx = np.argmin(cost[-1, :])
    cut[-1, :idx] = -1
    cut[-1, idx+1:] = +1

    # compare values next to the idx to decide if to shift the idx one step
    for i in range(cost.shape[0]-2, -1, -1):
        for j in range(cost.shape[1]):

            idx_max = cost.shape[1]-1
            window = range(max(idx-1, 0), min(idx+1, idx_max)+1)

            # compare [idx-1] with [idx-1:idx+1]
            if idx > 0 and cost[i, idx-1] == np.min(cost[i, window]):
                idx -= 1
            # compare [idx+1] with [idx-1:idx+1]
            elif idx < idx_max and cost[i, idx+1] == np.min(cost[i, window]):
                idx += 1

            cut[i, :idx] = -1
            cut[i, idx] = 0
            cut[i, idx+1:] = +1

    if direction:
        cut = np.transpose(cut)
    
    return cut



