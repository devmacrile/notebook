# -*- coding: utf-8 -*-
"""
The location of a substring s[j:k] is its beginning position j; 
note that tt will have multiple locations in ss if it occurs more than 
once as a substring of s.

Given: Two DNA strings s and t (each of length at most 1 kbp).

Return: All locations of t as a substring of s.
"""

import re

from franklin import clock


def load_data():
    with open('data/rosalind_subs.txt') as f:
        dnas = f.readlines()
    return [d.strip() for d in dnas]


@clock
def main():
    bigstring, substring = load_data()
    return [match.start() + 1 for match in re.finditer('(?=%s)' % substring, bigstring)]

if __name__ == '__main__':
    print(' '.join(map(str, main())))