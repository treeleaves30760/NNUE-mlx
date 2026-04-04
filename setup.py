"""Build the accelerated NNUE inference C extension.

Usage:
    python setup.py build_ext --inplace
    # or via Makefile:
    make build-accel
"""

import platform
from setuptools import setup, Extension, find_packages

ext_modules = []

if platform.system() == "Darwin" and platform.machine() == "arm64":
    ext = Extension(
        "src.accel._nnue_accel",
        sources=["src/accel/_nnue_accel.c"],
        extra_compile_args=["-O3", "-march=native", "-flto"],
        extra_link_args=["-framework", "Accelerate"],
        define_macros=[("USE_NEON", "1"), ("USE_ACCELERATE", "1")],
    )
    ext_modules.append(ext)

setup(
    ext_modules=ext_modules,
    packages=find_packages(),
)
