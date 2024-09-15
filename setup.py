#!/usr/bin/env python

from setuptools import find_packages, setup
import os
import glob
import torch
from torch.utils.cpp_extension import CppExtension

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "king/dimensions/subject_consistency/yolox", "layers", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "king/dimensions/subject_consistency/yolox._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

def fetch_readme():
    with open('README.md', encoding='utf-8') as f:
        text = f.read()
    return text

def fetch_requirements():
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')
    with open(filename, 'r') as f:
        envs = [line.rstrip('\n') for line in f.readlines() if '@' not in line]
    return envs

install_requires = fetch_requirements()

setup(name='king',
      version='0.1.0',
      description='Evaluating the smooth tracking and consistent generation of entities in motion within diffused videos.',
      long_description=fetch_readme(),
      long_description_content_type='text/markdown',
      project_urls={
          'Source': 'https://github.com/fan23j/KING',
      },
      entry_points={
          'console_scripts': ['king=king.cli.king:main']
      },
      install_requires=install_requires,
      ext_modules=get_extensions(),
      packages=find_packages(),
      classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
      cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
      include_package_data=True,
      license='Apache Software License 2.0',
)
