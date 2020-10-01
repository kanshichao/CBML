
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension


requirements = ["torch", "torchvision"]

setup(
    name="cbml_benchmark",
    version="0.1",
    author="Anonymous",
    url="https://github.com/Anonymous",
    description="CBML",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=requirements,
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
