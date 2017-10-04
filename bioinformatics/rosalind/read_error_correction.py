# -*- coding: utf-8 -*-
"""
As is the case with point mutations, the most common type of sequencing 
error occurs when a single nucleotide from a read is interpreted incorrectly.

Given: A collection of up to 1000 reads of equal length (at most 50 bp) in 
FASTA format. Some of these reads were generated with a single-nucleotide error. 
For each read s in the dataset, one of the following applies:

    * s was correctly sequenced and appears in the dataset at 
        least twice (possibly as a reverse complement);
    * s is incorrect, it appears in the dataset exactly once, 
        and its Hamming distance is 1 with respect to exactly 
        one correct read in the dataset (or its reverse complement).

Return: A list of all corrections in the form "[old read]->[new read]". 
(Each correction must be a single symbol substitution, and you may return 
the corrections in any order.)
"""

from franklin import clock, load_fasta

"""
Keep list sorted, only check...
set?
"""

@clock
def main():
    fastas = load_fasta('test.txt')
    return fastas

if __name__ == '__main__':
    main()