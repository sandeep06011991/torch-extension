from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension
from torch.utils import cpp_extension

setup(name='custom_kernels',
      ext_modules=[
        CUDAExtension('lltm_cpp', [
            'lltm_cuda_kernel.cu',
            'lltm_cuda.cpp',
        ]),
        cpp_extension.CppExtension('custom_kernels', ['gather.cpp'])

      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
