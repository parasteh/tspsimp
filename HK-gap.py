import itertools
import pickle
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from math import sqrt

import numpy as np


def held_karp(dists):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.
    Parameters:
        dists: distance matrix
    Returns:
        A tuple, (cost, path).
    """
    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list(reversed(path))


def generate_distances(n):

    dists = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dists[i][j] = dists[j][i] = random.randint(1, 99)

    return dists


def read_distances(filename):
    dists = []
    with open(filename, 'rb') as f:
        for line in f:
            # Skip comments
            if line[0] == '#':
                continue

            dists.append(map(int, map(str.strip, line.split(','))))

    return dists

def get_distances(tsp):
    length = len(tsp)

    distance_matrix  = np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            dist = get_distance(tsp[i], tsp[j])
            distance_matrix[i,j]  = dist
    return distance_matrix

def get_distance(x,y):
    return sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 )

from tqdm import tqdm
if __name__ == '__main__':
    pbar = tqdm(total=10000)
    with open('datasets/tsp_20_10000.pkl','rb') as input_file:
        test_set = pickle.load(input_file)
    costs = []
    size  = 20
    for i in test_set:
        dists = get_distances(i)
        cost = held_karp(dists)
        costs.append(cost)
        pbar.update(1)
    print( "mean average cost is : " + str(np.mean(costs)))
    # # Pretty-print the distance matrix
    # for row in dists:
    #     print(''.join([str(n).rjust(3, ' ') for n in row]))
    executor = ThreadPoolExecutor(max_workers=1)
    print('')
    print("HK")
    print(held_karp(dists))