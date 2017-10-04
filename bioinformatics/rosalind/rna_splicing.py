# -*- coding: utf-8 -*-
"""
After identifying the exons and introns of an RNA string, we only need 
to delete the introns and concatenate the exons to form a new string ready 
for translation.

Given: A DNA string s (of length at most 1 kbp) and a collection of substrings 
of s acting as introns. All strings are given in FASTA format.

Return: A protein string resulting from transcribing and translating the exons 
of s. (Note: Only one solution will exist for the dataset provided.)
"""

from franklin import clock, load_fasta, load_rna_translations


def translate(s):
    translations = load_rna_translations()
    amino_acids = []
    for i in range(len(s) // 3):
        codon = s[(i * 3):(i * 3 + 3)]
        translation = translations[codon]
        if translation == 'Stop':
            break
        amino_acids.append(translation)
    return ''.join(amino_acids)


@clock
def main():
    fasta = load_fasta('rosalind_splc.txt')
    
    # assumption based on sample input data?
    # nope; another example where fasta could be better
    # as a list than a dict. should find maxlen string here
    mainkey = 'Rosalind_4192'  # sorted(fasta.keys())[0]
    dna = fasta[mainkey]
    substrings = [fasta[k] for k in sorted(fasta.keys()) if k != mainkey]
    
    # remove introns from our string
    for substring in substrings:
        dna = dna.replace(substring, '')

    # transcribe dna to rna, then translate
    rna = dna.replace('T', 'U')
    return translate(rna)


if __name__ == '__main__':
    main()