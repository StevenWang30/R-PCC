import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
try:
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    from setuptools import Extension as Pybind11Extension
from pybind11.setup_helpers import build_ext as Pybind11BuildExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def make_cpp_ext(name, module, sources):
    pybind11_modules = Pybind11Extension(
            name='%s.%s' % (module, name),
            sources=[os.path.join(*module.split('.'), src) for src in sources],
    )
    return pybind11_modules


if __name__ == '__main__':
    setup(
        name='r-pcc',
        description='A baseline for range image-based point cloud compression. (CUDA extension)',
        install_requires=[
            'numpy',
            'torch>=1.7',
            'tensorboardX',
        ],
        author='Sukai Wang',
        author_email='swangcy@connect.com',
        license='Apache License 2.0',
        packages=find_packages(),
        cmdclass={'build_ext': BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name='furthest_point_sampling_cuda',
                module='ops.fps',
                sources=[
                    'src/fps_api.cpp',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',
                ],
            ),
        ],
    )
    setup(
        name='r-pcc',
        description='A baseline for range image-based point cloud compression. (c++ extension)',
        install_requires=[
            'easydict',
            'pyyaml',
            'lz4==0.7.0',
            'ninja'
        ],
        cmdclass={'build_ext': Pybind11BuildExtension},
        ext_modules=[
            make_cpp_ext(
                name='feature_extractor_cpp',
                module='ops.cpp_modules',
                sources=[
                    'src/cpp_modules.cpp',
                ],
            ),
            make_cpp_ext(
                name='segment_utils_cpp',
                module='ops.cpp_modules',
                sources=[
                    'src/cpp_modules.cpp',
                ],
            ),
            make_cpp_ext(
                name='quantization_utils_cpp',
                module='ops.cpp_modules',
                sources=[
                    'src/cpp_modules.cpp',
                ],
            ),
            make_cpp_ext(
                name='dataset_utils_cpp',
                module='ops.cpp_modules',
                sources=[
                    'src/cpp_modules.cpp',
                ],
            ),
            make_cpp_ext(
                name='contour_utils_cpp',
                module='ops.cpp_modules',
                sources=[
                    'src/cpp_modules.cpp',
                ],
            ),
        ],
    )


