from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension
from torch.utils import cpp_extension

setup(name='custom_kernels',
      ext_modules=[
        CUDAExtension('cuda_kernel', [
            'gather_kernel.cu',
            'gather_cuda.cpp',
        ]),
        cpp_extension.CppExtension('c_kernel', ['gather.cpp'])
    ],
  cmdclass={'build_ext': cpp_extension.BuildExtension})
