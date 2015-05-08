#!/usr/bin/env python
import os
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--home', help='install in home directory',action="store_true")
args = parser.parse_args()

if args.home:
    cmd='python setup.py install --home=~'
else:
    cmd='python setup.py install'

print cmd
os.system(cmd)
