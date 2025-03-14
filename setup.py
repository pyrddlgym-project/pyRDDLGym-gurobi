# This file is part of pyRDDLGym.

# pyRDDLGym is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation.

# pyRDDLGym is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.

# You should have received a copy of the MIT License
# along with pyRDDLGym. If not, see <https://opensource.org/licenses/MIT>.

from setuptools import setup, find_packages

from pathlib import Path
long_description = (Path(__file__).parent / "README.md").read_text()

setup(
      name='pyRDDLGym-gurobi',
      version='0.2',
      author="Michael Gimelfarb, Ayal Taitler, Scott Sanner",
      author_email="mike.gimelfarb@mail.utoronto.ca, ataitler@gmail.com, ssanner@mie.utoronto.ca",
      description="pyRDDLGym-gurobi: Gurobi compilation of RDDL description files, and optimization tools.",
      long_description=long_description,
      long_description_content_type='text/markdown',
      license="MIT License",
      url="https://github.com/pyrddlgym-project/pyRDDLGym-gurobi",
      packages=find_packages(),
      install_requires=[
          'pyRDDLGym>=2.0',
          'gurobipy>=10.0.0' 
        ],
      python_requires=">=3.8",
      package_data={'': ['*.cfg']},
      include_package_data=True,
      classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
