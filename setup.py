from setuptools import setup, find_packages

with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
  name="leakage_free_openfe",
  version="0.1",
  author="original by: Tianping Zhang, modified by: Ram Seshadri",
  author_email="",
  description="Leakage-Free-OpenFE: automated feature generation without data leakage",
  long_description=long_description,
  long_description_content_type="text/markdown",
  python_requires='>=3.6',
  install_requires=[
      "numpy>=1.19.3",
      "pandas>=1.1.5",
      "scikit_learn>=0.24.2",
      "lightgbm>=3.3.2",
      # "xgboost>=1.5.2",
      "scipy>=1.5.4",
      "tqdm",
      "pyarrow",
  ],
  url="https://github.com/AutoViML/Leakage_Free_OpenFE",
  packages=find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)