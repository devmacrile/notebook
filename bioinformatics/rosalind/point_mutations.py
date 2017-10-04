# -*- coding: utf-8 -*-
"""
Given two strings s and t of equal length, the Hamming distance 
between s and t, denoted dH(s,t), is the number of corresponding 
symbols that differ in s and t.

Given: Two DNA strings s and t of equal length (not exceeding 1 kbp).

Return: The Hamming distance dH(s,t).
"""

from franklin import clock


def load_data():
    with open('data/rosalind_hamm.txt') as f:
        dna_strings = f.readlines()
    return [d.strip() for d in dna_strings]


def hamming_distance(s1, s2):
    diffs = 0
    for i, s in enumerate(s1):
        if s != s2[i]:
            diffs += 1
    return diffs


@clock
def main():
    dnas = load_data()
    return hamming_distance(dnas[0], dnas[1])


if __name__ == '__main__':
    main()