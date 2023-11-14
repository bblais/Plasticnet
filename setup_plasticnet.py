# this is from https://github.com/cython/cython/wiki/PackageHierarchy

import sys, os, stat, subprocess
from distutils.core import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension

# we'd better have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org and install it")
    sys.exit(1)

import numpy

def get_version(package):
    
    d={}
    version_line=''
    with open('%s/__init__.py' % (package)) as fid:
        for line in fid:
            if line.startswith('version='):
                version_line=line
    print(version_line)
    exec(version_line,d)
    return d['version']



# scan the  directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files

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

# generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    folder=extName.split(".")[0]

    files=[extPath]
    
    files.append('splikes/randomkit.c')

    return Extension(
        extName,
        files,
        include_dirs = [numpy.get_include(), ".", "%s/" % folder],   # adding the '.' to include_dirs is CRUCIAL!!
        extra_compile_args = ["-O3", ],
        extra_link_args = ['-g'],
        )

# get the list of extensions
extNames = scandir("plasticnet")

#cleanc("plasticnet")

# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]
# finally, we can pass all this to distutils
setup(
  name="plasticnet",
  version=get_version('plasticnet'),
  description="Plasticity in Rate-Based Neurons",
  author="Brian Blais",
  packages=["plasticnet", "plasticnet.neurons", "plasticnet.connections", "plasticnet.monitors"],
  ext_modules=extensions,
  cmdclass = {'build_ext': build_ext},
)

# # get the list of extensions
# extNames = scandir("splikes")
# #cleanc("splikes")

# # and build up the set of Extension objects
# extensions = [makeExtension(name) for name in extNames]
# # finally, we can pass all this to distutils
# setup(
#   name="splikes",
#   version=get_version('splikes'),
#   description="Plasticity in Spike-Based Neurons",
#   author="Brian Blais",
#   packages=["splikes", "splikes.neurons", "splikes.connections", "splikes.monitors"],
#   ext_modules=extensions,
#   cmdclass = {'build_ext': build_ext},
# )