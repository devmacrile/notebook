# -*- coding: utf-8 -*-
"""
The 20 commonly occurring amino acids are abbreviated by using 20 letters from 
the English alphabet (all letters except for B, J, O, U, X, and Z). Protein strings 
are constructed from these 20 symbols. Henceforth, the term genetic string will 
incorporate protein strings along with DNA strings and RNA strings.

The RNA codon table dictates the details regarding the encoding of specific codons 
into the amino acid alphabet.

Given: An RNA string ss corresponding to a strand of mRNA (of length at most 10 kbp).

Return: The protein string encoded by s.
"""

from franklin import clock, load_rna_translations


def read_string():
    with open('data/rosalind_prot.txt') as f:
        rna = f.readline().strip()
    return rna


@clock
def main():
    s = read_string()
    translations = load_rna_translations()
    amino_acids = []
    for i in range(len(s) // 3):
        codon = s[(i * 3):(i * 3 + 3)]
        translation = translations[codon]
        if translation == 'Stop':
            break
        amino_acids.append(translation)
    return ''.join(amino_acids)


if __name__ == '__main__':
    main()