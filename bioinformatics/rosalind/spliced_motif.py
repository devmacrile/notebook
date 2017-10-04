# -*- coding: utf-8 -*-
"""
As a substring can have multiple locations, a subsequence can have multiple 
collections of indices, and the same index can be reused in more than one 
appearance of the subsequence; for example, ACG is a subsequence of AACCGGTT 
in 8 different ways.

Given: Two DNA strings s and t (each of length at most 1 kbp) in FASTA format.

Return: One collection of indices of s in which the symbols of t appear as a 
subsequence of s. If multiple solutions exist, you may return any one.
"""

from franklin import clock, load_fasta


@clock
def main():
    fasta = load_fasta('rosalind_sseq.txt')
    skey = max(fasta, key=lambda k: len(fasta[k]))
    tkey = min(fasta, key=lambda k: len(fasta[k]))
    s, t = fasta[skey], fasta[tkey]
    i = 0
    indices = []
    for v in t:
        index = s[i:].find(v)
        indices.append(i + (index + 1))
        i += index + 1
    print ' '.join(map(str, indices))
    return indices


if __name__ == '__main__':
    # ' '.join(map(str, main()))
    main()