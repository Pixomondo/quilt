#! /usr/bin/env python
"""
Minimum cut finder
"""

# import 3rd party modules
from numpy import transpose, ones, any, uint8
from maxflow import Graph


def mincut(matrix, direction=None):
    """
    Computes the minimum cut from one edge of the matrix to the other

    Inputs:
       matrix:  Evaluations of 2d function to be cut along local minima
       direction: 0 = vertical cut, 1 = horizontal cut

    Outputs:
       cut: Matrix containing entries indicating side
            0: left (or top) side of cut
            1: right (or bottom) side of cut
    """

    # the images are the same: the mask is just ones
    if not any(matrix):
        return ones(matrix.shape)

    if direction:
        matrix = transpose(matrix)

    g = Graph[float]()
    # Add the nodes. nodeids has the identifiers of the nodes in the grid.
    nodeids = g.add_grid_nodes(matrix.shape)

    # Add non-terminal edges with the same capacity.
    g.add_grid_edges(nodeids, weights=matrix)

    # Add the terminal edges.
    # internal node are not connected to source nor target
    g.add_grid_tedges(nodeids, 0, 0)
    # left column: connected to source
    g.add_grid_tedges(nodeids[:, 0], matrix[:, 0], 0)
    # right column: connected to target
    g.add_grid_tedges(nodeids[:, -1], 0, matrix[:, -1])

    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    sgm = uint8(g.get_grid_segments(nodeids))

    if direction:
        sgm = transpose(sgm)

    return sgm
