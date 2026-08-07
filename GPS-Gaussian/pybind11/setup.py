import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))

if sys.platform == 'win32':
    glfw_root = os.path.join(here, 'glfw-3.4.bin.WIN64')
    libraries = ['glfw3', 'opengl32', 'gdi32', 'user32', 'shell32']
    library_dirs = [os.path.join(glfw_root, 'lib-vc2022')]
    include_dirs = [os.path.join(here, 'include'), os.path.join(glfw_root, 'include')]
    cxx_args = ['/O2', '/std:c++17']
else:
    libraries = ['glfw', 'GL', 'dl']
    library_dirs = []
    include_dirs = [os.path.join(here, 'include')]
    cxx_args = ['-O3', '-D__STDC_LIMIT_MACROS', '-D__STDC_CONSTANT_MACROS', 
                '-U_FORTIFY_SOURCE', '-D_FORTIFY_SOURCE=0']

setup(
    name='CudaRuntime1',
    ext_modules=[
        CUDAExtension(
            name='CudaRuntime1',
            sources=['viewer.cpp', 'src/glad.c'],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': ['-O3'],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)