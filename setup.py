import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nzgmdb",
    version="1.0.0",
    author="Quakecore",
    description="Package for executing the pipeline for the NZGMDB",
    url="https://github.com/ucgmsim/nzgmdb",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={"nzgmdb": ["data/*"]},
)
