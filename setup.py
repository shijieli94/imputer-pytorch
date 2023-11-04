from setuptools import find_packages, setup
from torch.utils import cpp_extension

extensions = [
    cpp_extension.CUDAExtension(
        "torch_imputer",
        sources=[
            "torch_imputer/best_alignment_kernel.cu",
            "torch_imputer/imputer_kernel.cu",
            "torch_imputer/imputer.cpp",
        ],
    )
]

setup(
    name="torch_imputer",
    version="0.1.0",
    description="Implementation of Imputer: Sequence Modelling via Imputation and Dynamic Programming in PyTorch ",
    url="https://github.com/rosinality/imputer-pytorch",
    author="Kim Seonghyeon",
    author_email="kim.seonghyeon@navercorp.com",
    license="MIT",
    python_requires=">=3.6",
    packages=find_packages(exclude=["example"]),
    ext_modules=extensions,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
