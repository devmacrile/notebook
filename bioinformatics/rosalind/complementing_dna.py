"""
In DNA strings, symbols 'A' and 'T' are complements of each other, as are 'C' and 'G'.

The reverse complement of a DNA string s is the string scsc formed by reversing 
the symbols of ss, then taking the complement of each symbol (e.g., the reverse 
complement of "GTCA" is "TGAC").

Given: A DNA string s of length at most 1000 bp.

Return: The reverse complement s^c of s.
"""

from franklin import clock

pairs = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


def load_data():
    with open('data/rosalind_revc.txt') as f:
        dna_string = ''.join(f.read().splitlines())
    return dna_string

@clock
def main():
    dna = load_data()
    return ''.join(reversed([pairs[k] for k in dna]))

if __name__ == '__main__':
    main()