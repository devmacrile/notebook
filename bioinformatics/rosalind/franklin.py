"""
Just some utility functions accumulated while solving some of these problems.
"""

import time
import difflib
import functools


def clock(func):
    """
    Simple utility for clocking function call and displaying
    return value.
    """
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result
    return clocked


def load_fasta(fname):
    with open('data/%s' % fname) as f:
        contents = f.read()
    return parse_fasta(contents)


def parse_fasta(fasta):
    parseid = lambda x: x.split('\n')[0]
    parsedna = lambda x: ''.join(x.split('\n')[1:])
    return {parseid(s): parsedna(s) for s in fasta.split('>')[1:]}


def load_rna_translations():
    rna_codons = {}
    with open('data/rna_codon_table.txt') as f:
        lines = f.readlines()
    for line in lines:
        translations = [s for s in line.strip().split('  ') if s]
        for translation in translations:
            codon, acid = translation.split()
            rna_codons[codon] = acid
    return rna_codons


def longest_common_substring(s, t):
    """
    TODO only returns from starting indices?
    See goo.gl/Cz98dX
    """
    matcher = difflib.SequenceMatcher(None, s, t, autojunk=False)
    match = matcher.find_longest_match(0, len(s), 0, len(t))
    #print(match)
    #print(matcher.get_matching_blocks())
    return s[match.a:(match.a + match.size)], match.size


def lcs(s, t):
    """
    Computes the longest common substring of s and t.
    Use longest_common_substring ^ instead.
    """
    m = [[0] * (1 + len(t)) for i in xrange(1 + len(s))]
    longest, x_longest = 0, 0
    for x in xrange(1, 1 + len(s)):
       for y in xrange(1, 1 + len(t)):
           if s[x - 1] == t[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
    return s[x_longest - longest: x_longest], longest
