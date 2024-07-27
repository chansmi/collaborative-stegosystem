from setuptools import setup, find_packages

setup(
    name="collaborative-stegosystem",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "trl==0.4.7",
        "transformers==4.30.2",
        "torch==2.0.1",
        "pyyaml==6.0",
    ],
    author="Chandler",
    author_email="smith.18.chandler@gmail.com",
    description="An experiment in collaborative steganography using reinforcement learning with language models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)