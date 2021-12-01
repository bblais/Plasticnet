#!/usr/bin/env python
import os
import sys

def cleanc(dir):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            base,ext=os.path.splitext(path)
            cpath=base+'.c'
            if os.path.isfile(cpath):
                os.remove(cpath)
                print("~~",cpath)
            cpath=base+'.so'
            if os.path.isfile(cpath):
                os.remove(cpath)
                print("~~",cpath)
        elif os.path.isdir(path):
            cleanc(path)

cleanc("plasticnet")
cleanc("splikes")

