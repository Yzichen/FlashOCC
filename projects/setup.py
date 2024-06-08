from setuptools import find_packages, setup

import os
import shutil
import sys
import torch
import warnings
from os import path as osp
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    setup(
        name='flashocc_plugin',
        description=("OpenMMLab's next-generation platform"
                     'for general 3D object detection.'),
        long_description_content_type='text/markdown',
        author='MMDetection3D Contributors',
        author_email='zwwdev@gmail.com',
        keywords='computer vision, 3D object detection',
        url='https://github.com/open-mmlab/mmdetection3d',
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        license='Apache License 2.0',
        ext_modules=[
            make_cuda_ext(
                name="bev_pool_ext",
                module="mmdet3d_plugin.ops.bev_pool",
                sources=[
                    "src/bev_pooling.cpp",
                    "src/bev_sum_pool.cpp",
                    "src/bev_sum_pool_cuda.cu",
                    "src/bev_max_pool.cpp",
                    "src/bev_max_pool_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="bev_pool_v2_ext",
                module="mmdet3d_plugin.ops.bev_pool_v2",
                sources=[
                    "src/bev_pool.cpp",
                    "src/bev_pool_cuda.cu"
                ],
            ),
            make_cuda_ext(
                name="nearest_assign_ext",
                module="mmdet3d_plugin.ops.nearest_assign",
                sources=[
                    "src/nearest_assign.cpp",
                    "src/nearest_assign_cuda.cu"
                ],
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
