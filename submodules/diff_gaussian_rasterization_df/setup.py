#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
import os

_here = os.path.dirname(os.path.abspath(__file__))

ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    have_torch = True
except Exception:
    have_torch = False

if have_torch and os.environ.get('CUDA_HOME'):
    ext_modules = [
        CUDAExtension(
            name="diff_gaussian_rasterization_df._C",
            sources=[
                os.path.join(_here, "cuda_rasterizer/rasterizer_impl.cu"),
                os.path.join(_here, "cuda_rasterizer/forward.cu"),
                os.path.join(_here, "cuda_rasterizer/backward.cu"),
                os.path.join(_here, "rasterize_points.cu"),
                os.path.join(_here, "ext.cpp")],
            extra_compile_args={"nvcc": ["-I" + os.path.join(_here, "third_party/glm/")]})
    ]
    cmdclass = {'build_ext': BuildExtension}
else:
    print("Warning: CUDA extension build skipped. Ensure 'torch' is installed and 'CUDA_HOME' is set to build the native extensions.")

setup(
    name="diff_gaussian_rasterization_df",
    packages=['diff_gaussian_rasterization_df'],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
