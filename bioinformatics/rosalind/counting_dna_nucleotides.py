# -*- coding: utf-8 -*-
"""
An example of a length 21 DNA string (whose alphabet contains the 
symbols 'A', 'C', 'G', and 'T') is "ATGCTTCAGAAAGGTCTTACG."

Given: A DNA string s of length at most 1000 nt.

Return: Four integers (separated by spaces) counting the respective 
number of times that the symbols 'A', 'C', 'G', and 'T' occur in s.
"""

from collections import Counter

from franklin import clock


def load_data():
    with open('data/rosalind_dna.txt') as f:
        dna_string = f.readline()
    return dna_string

@clock
def main():
    dna_string = load_data()
    ntide_counts = Counter(dna_string)
    return [ntide_counts[k] for k in ['A', 'C', 'G', 'T']]

if __name__ == '__main__':
    main()