from setuptools import setup, find_namespace_packages

if __name__ == '__main__':
    setup(
        name = 'hatecomp.HASOC',
        version = '0.0.1',
        package_dir = {'' : 'src'},
        packages = find_namespace_packages(
            where = 'src'
        )
    )