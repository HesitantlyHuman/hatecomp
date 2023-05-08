import os
from setuptools import setup, find_packages

PACKAGE_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = open(os.path.join(PACKAGE_ROOT, "README.md"), "r").read()

if __name__ == "__main__":
    setup(
        name="hatecomp",
        version="0.4.0",
        description="Collection of pytorch datasets for hate speech and toxic internet discourse",
        long_description=README_FILE,
        long_description_content_type="text/markdown",
        url="https://github.com/HesitantlyHuman/hatecomp",
        author="HesitantlyHuman",
        author_email="tannersims@hesitantlyhuman.com",
        license="MIT",
        package_data={"hatecomp": ["models/model_registry.json"]},
        packages=find_packages(),
        install_requires=[
            "aiometer >=0.3.0, <1.0.0",
            "httpx >=0.21.1, <1.0.0",
            "requests >=2.0.0, <3.0.0",
            "torch >=1.7.0, <2.0.0",
            "tqdm >=4.1.0, <5.0.0",
            "tweepy >=4.0.0, <5.0.0",
            "transformers >=4.16.0, <5.0.0",
        ],
    )
