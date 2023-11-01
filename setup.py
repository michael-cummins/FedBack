from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='admm',
    version='0.0.1',
    author='Michael Cummins',
    description='Federated Learning via ADMM',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url='https://github.com/michael-cummins/ADMM',
    install_requires=[
        # 'numpy>=1.25.2',
        # 'torch>=1.0',
        # 'torchvision>=1.0',
        # 'hydra-core',
        # 'ray==1.13'
    ],
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)