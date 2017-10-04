# -*- coding: utf-8 -*-
"""
For a collection of strings, a larger string containing every one of the smaller 
strings as a substring is called a superstring.

By the assumption of parsimony, a shortest possible superstring over a collection 
of reads serves as a candidate chromosome.

Given: At most 50 DNA strings of equal length, not exceeding 1 kbp, in FASTA format 
(which represent reads deriving from the same strand of a single linear chromosome).

The dataset is guaranteed to satisfy the following condition: there exists a unique 
way to reconstruct the entire chromosome from these reads by gluing together pairs of 
reads that overlap by more than half their length.

Return: A shortest superstring containing all the given strings (thus corresponding to a 
reconstructed chromosome).
"""

from franklin import clock, load_fasta


def max_overlap(strings):
    """
    Returns the indices of the strings the maximum overlap and
    the overlap amount as a tuple (i, j, k), where i is the source
    string index (suffix), j in the sink string index (prefix)
    and k is the overlap between the two strings.
    """
    max_overlap = 0
    index_tuple = (None, None, None)
    for i, s in enumerate(strings):
        for j, t in enumerate(strings):
            if i == j:
                continue
            for k in reversed(range(len(s))):
                if k < max_overlap:
                    break
                elif k == max_overlap:
                    multiple_maxes = True
                    candidates = [index_tuple, (i, j, k)]
                if s[-k:] == t[:k]:
                    max_overlap = k
                    index_tuple = (i, j, k)
    
    return index_tuple


@clock
def main():
    fastas = load_fasta('rosalind_long.txt')
    dnas = [fastas[k] for k in fastas.keys()]
    
    # greedily compute the max overlap in remaining
    # set of strings, and merge strings with max overlap
    # until we have one single superstring
    while len(dnas) > 1:
        source_index, sink_index, k = max_overlap(dnas)
        source = dnas[source_index]
        source_prefix = source[:(len(source) - k)]
        dnas[sink_index] = source_prefix + dnas[sink_index]
        del dnas[source_index]
    assert len(dnas) == 1
    return dnas[0]


if __name__ == '__main__':
    main()
