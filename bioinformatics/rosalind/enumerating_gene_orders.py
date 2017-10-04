# -*- coding: utf-8 -*-
"""
A permutation of length nn is an ordering of the positive integers {1,2,…,n}. 
For example, π=(5,3,2,1,4) is a permutation of length 5.

Given: A positive integer n ≤ 7.

Return: The total number of permutations of length n, followed by a list 
of all such permutations (in any order).
"""

import math
from itertools import permutations

from franklin import clock

digits = '123456'

@clock
def main():
    num_perms = math.factorial(len(digits))
    for perm in permutations(digits, len(digits)):
        print(' '.join(perm))
    return num_perms


if __name__ == '__main__':
    main()