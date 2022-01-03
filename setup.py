import os
from setuptools import setup, find_packages

PACKAGE_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = open(os.path.join(PACKAGE_ROOT, 'README.md'), 'r').read()

if __name__ == '__main__':
    setup(
        name = 'hatecomp',
        version = '0.1.0',
        description = "Collection of pytorch datasets for hate speech and toxic internet discourse",
        long_description = README_FILE,
        long_description_content_type = "text/markdown",
        url = 'https://github.com/GenerallyIntelligent/hatecomp',
        author = 'GenerallyIntelligent',
        author_email = 'tannersims@generallyintelligent.me',
        license = 'MIT',
        packages = find_packages(),
        install_requires = [
            'aiometer>=0.3.0',
            'httpx>=0.21.1',
            'requests>=2.26.0',
            'torch>=1.10.1',
            'tqdm>=4.62.3',
            'tweepy>=4.4.0'
        ]
    )