#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the whole pipeline of generating and fitting light curves
for a transmission analysis.
"""

from glob import glob
import numpy as np

delimiter_string = '***********'

print( delimiter_string,'Starting run using the following parameters:',delimiter_string,'\n')
with open('parameters.py', 'r') as f:
    print(f.read())
    f.close()

files2run = ['e1', 'e2', 'e3', 'e4']
# files2run = ['e7', 'e8']

for file2run in files2run:
    f = glob( file2run + '*.py')[0]
    print( '\n', delimiter_string, 'Running', f, delimiter_string,'\n')
    exec(open(f).read())

print( '\n', delimiter_string, 'Run finished', delimiter_string,'\n' )