# -*- coding: utf-8 -*-
"""
A common substring of a collection of strings is a substring of every member 
of the collection. We say that a common substring is a longest common substring 
if there does not exist a longer common substring. For example, "CG" is a common 
substring of "ACGTACGT" and "AACCGTATA", but it is not as long as possible; in 
this case, "CGTA" is a longest common substring of "ACGTACGT" and "AACCGTATA".

Note that the longest common substring is not necessarily unique; for a simple 
example, "AA" and "CC" are both longest common substrings of "AACC" and "CCAA".

Given: A collection of k (k ≤ 100) DNA strings of length at most 1 kbp each 
in FASTA format.

Return: A longest common substring of the collection. (If multiple solutions exist, 
you may return any single solution.)
"""

from franklin import clock, load_fasta, longest_common_substring


@clock
def main():
    fasta = load_fasta('rosalind_lcsm.txt')
    sequences = [fasta[k] for k in fasta.keys()]
    n = len(sequences)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s, t = sequences[i], sequences[j]
            lcs, k = longest_common_substring(s, t)
            if lcs and all([lcs in seq for seq in sequences]):
                return lcs
    return None


if __name__ == '__main__':
    main()
