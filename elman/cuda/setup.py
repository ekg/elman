# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


VERSION = '0.5.0-rc0'
DESCRIPTION = 'Hasty: CUDA kernels for Elman Ladder RNNs.'
AUTHOR = 'LMNT, Inc.'
AUTHOR_EMAIL = 'haste@lmnt.com'
URL = 'https://github.com/lmnt-com/haste'
LICENSE = 'Apache 2.0'
CLASSIFIERS = [
  'Development Status :: 4 - Beta',
  'Intended Audience :: Developers',
  'Intended Audience :: Education',
  'Intended Audience :: Science/Research',
  'License :: OSI Approved :: Apache Software License',
  'Programming Language :: Python :: 3.6',
  'Programming Language :: Python :: 3.7',
  'Programming Language :: Python :: 3.8',
  'Topic :: Scientific/Engineering :: Mathematics',
  'Topic :: Software Development :: Libraries :: Python Modules',
  'Topic :: Software Development :: Libraries',
]


import os
import sys

from glob import glob
from platform import platform
from torch.utils import cpp_extension
from setuptools import setup
from setuptools.dist import Distribution


class BuildHashy(cpp_extension.BuildExtension):
  def run(self):
    # Use parallel make with 12 jobs
    os.system('make -j12 hasty')
    super().run()


base_path = os.path.dirname(os.path.realpath(__file__))
if 'Windows' in platform():
  CUDA_HOME = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH'))
  extra_args = []
else:
  CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
  extra_args = ['-Wno-sign-compare']

with open(f'pytorch/_version.py', 'wt') as f:
  f.write(f'__version__ = "{VERSION}"')

# Only include the sources we actually need for Elman Ladder
ladder_sources = [
    'pytorch/elman_ladder.cc',
    'pytorch/support.cc',
]
extension = cpp_extension.CUDAExtension(
    'hasty_pytorch_lib',
    sources = ladder_sources,
    extra_compile_args = extra_args,
    include_dirs = [os.path.join(base_path, 'lib'), os.path.join(CUDA_HOME, 'include')],
    libraries = ['hasty'],
    library_dirs = ['.'])

# Try to read README if it exists
long_description = 'Elman Ladder: Log-space RNN research framework'
readme_path = os.path.join(base_path, 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(name = 'hasty_pytorch',
    version = VERSION,
    description = DESCRIPTION,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    url = URL,
    license = LICENSE,
    keywords = 'pytorch machine learning rnn lstm gru custom op',
    packages = ['hasty_pytorch'],
    package_dir = { 'hasty_pytorch': 'pytorch' },
    install_requires = [],
    ext_modules = [extension],
    cmdclass = { 'build_ext': BuildHashy },
    classifiers = CLASSIFIERS)
