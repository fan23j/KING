#!/usr/bin/env python

from setuptools import find_packages, setup
import os

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
      packages=find_packages(),
      include_package_data=True,
      license='Apache Software License 2.0',
)
