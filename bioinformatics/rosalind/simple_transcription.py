# -*- coding: utf-8 -*-
"""
Given a DNA string tt corresponding to a coding strand, its transcribed RNA 
string u is formed by replacing all occurrences of 'T' in t with 'U' in u.

Given: A DNA string t having length at most 1000 nt.

Return: The transcribed RNA string of t.
"""

from franklin import clock


def load_data():
    with open('data/rosalind_rna.txt') as f:
        dna_string = ''.join(f.read().splitlines())
    return dna_string

@clock
def main():
    dna = load_data()
    return dna.replace('T', 'U')

if __name__ == '__main__':
    main()
