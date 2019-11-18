from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='varfloat',
      ext_modules=[CUDAExtension('varfloat', ['varfloat_ext.cpp',
                                              'varfloat.cu'
                                              ])],
      cmdclass={'build_ext': BuildExtension})


# for installation execute:
# > python build.py install
# record list of all installed files:
# > python build.py install --record files.txt
