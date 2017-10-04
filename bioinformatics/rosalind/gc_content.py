"""
The GC-content of a DNA string is given by the percentage of symbols in 
the string that are 'C' or 'G'. For example, the GC-content of "AGCTATAG" 
is 37.5%. Note that the reverse complement of any DNA string has the same GC-content.

DNA strings must be labeled when they are consolidated into a database. A 
commonly used method of string labeling is called FASTA format. In this format, 
the string is introduced by a line that begins with '>', followed by some labeling 
information. Subsequent lines contain the string itself; the first line to begin 
with '>' indicates the label of the next string.

In Rosalind's implementation, a string in FASTA format will be labeled by the 
ID "Rosalind_xxxx", where "xxxx" denotes a four-digit code between 0000 and 9999.

Given: At most 10 DNA strings in FASTA format (of length at most 1 kbp each).

Return: The ID of the string having the highest GC-content, followed by the 
GC-content of that string. Rosalind allows for a default error of 0.001 in all 
decimal answers unless otherwise stated; please see the note on absolute error below.
"""

from collections import Counter

from franklin import clock


def read_fasta():
    with open('data/rosalind_gc.txt') as f:
        dnas = f.read()
    parse = lambda x: (x.split('\n')[0], ''.join(x.split('\n')[1:]))
    return [parse(d) for d in dnas.split('>')[1:]]

@clock
def main():
    fasta_id, max_gc = None, 0.0
    for id_, dna in read_fasta():
        basecounts = Counter(dna)
        gc = float(basecounts['G'] + basecounts['C']) / len(dna)
        if gc > max_gc:
            fasta_id = id_
            max_gc = gc
    return fasta_id, 100 * max_gc

if __name__ == '__main__':
    main()