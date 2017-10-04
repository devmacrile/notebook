# -*- coding: utf-8 -*-
"""
For a collection of strings and a positive integer k, the overlap graph 
for the strings is a directed graph Ok in which each string is represented 
by a node, and string s is connected to string t with a directed edge when 
there is a length k suffix of s that matches a length k prefix of t, as long 
as s≠t; we demand s≠t to prevent directed loops in the overlap graph 
(although directed cycles may be present).

Given: A collection of DNA strings in FASTA format having total length at most 10 kbp.

Return: The adjacency list corresponding to O3. You may return edges in any order.
"""

from collections import defaultdict

from franklin import clock, load_fasta


@clock
def main():
    k = 3
    adjacency = defaultdict(list)
    dnas = load_fasta('rosalind_grph.txt')
    for sid, s in dnas:
        for tid, t in dnas:
            if sid == tid:
                continue
            if s[-k:] == t[:k]:
                print(' '.join([sid, tid]))  # solution needs output
                adjacency[sid].append(tid)
    return adjacency

if __name__ == '__main__':
    main()