from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='fedback',
    version='0.0.1',
    author='Michael Cummins',
    description='Controlling Participation in Federated Learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url='https://github.com/michael-cummins/ADMM',
    install_requires=[
        'torch>=2',
        'torchvision',
        'matplotlib',
        'seaborn',
        'natsort',
        'tqdm',
    ],
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)