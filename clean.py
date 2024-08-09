#!/usr/bin/env python
import os
import sys
from glob import glob

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

for S in ['plasticnet.egg-info',
             'splikes.egg-info',
             'build','dist',
             '/opt/anaconda3/envs/work/lib/python3.11/site-packages/splikes*',
             '/opt/anaconda3/envs/work/lib/python3.11/site-packages/plasticnet*'
            ]:

    dirs=glob(S)
    for dir in dirs:
        if os.path.exists(dir):
            cmd=f'rm -rf {dir}'
            print(cmd)
            os.system(cmd)
        else:
            print(f"{dir} doesn't exist.")
