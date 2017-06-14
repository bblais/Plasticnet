#!/usr/bin/env python

import glob
import os

file_lines=[]

except_file=['isi_distributions.pyx','process.pyx']

found=[]
for fname in glob.glob('*.pyx'):
    if fname in except_file:
        continue
        
    with open(fname) as fid:
        lines=fid.readlines()
    found.append( ([line.strip() for line in lines if line.startswith('cdef class')],fname) )
    
    

for lines,fname in found:    
    base,ext=os.path.splitext(fname)
    for line in lines:
        obj=line.split('cdef class')[1].split('(')[0].strip()
        file_lines.append("from .%s import %s" % (base,obj))
        

file_lines.append("from . import process")

if os.path.exists("__init__.py"):
    os.rename("__init__.py","backup__init__.py")
    
s="\n".join(file_lines)
print(s)

with open("__init__.py","w") as fid:
    fid.write(s)


