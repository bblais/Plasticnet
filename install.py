#!/usr/bin/env python
import os
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--clean', help='clean install',action="store_true")
args = parser.parse_args()

if args.clean:
    cmd="python clean.py"
    print(cmd)
    os.system(cmd)

cmd="cp setup_plasticnet.py setup.py"
print(cmd)
os.system(cmd)

cmd="pip install ."
print(cmd)
os.system(cmd)

cmd="cp setup_splikes.py setup.py"
print(cmd)
os.system(cmd)

cmd="pip install ."
print(cmd)
os.system(cmd)

