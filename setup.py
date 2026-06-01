from pathlib import Path

import numpy as np
from setuptools import Extension, setup


WIGNERD_C_DIR = Path("b1like") / "c"

wignerd_ext = Extension(
    "b1like.cwignerd",
    sources=[str(WIGNERD_C_DIR / "cwignerd.c"), str(WIGNERD_C_DIR / "wignerd.c")],
    include_dirs=[str(WIGNERD_C_DIR), np.get_include()],
    define_macros=[("PyInit_wignerd", "PyInit_cwignerd")],
    extra_compile_args=["-O3"],
)

setup(ext_modules=[wignerd_ext])
