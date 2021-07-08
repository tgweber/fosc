from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='fosc',
    version='0.0.2',
    descripion='Field Of Study Classifier (FOSC)',
    long_description=readme,
    author='Tobias Weber',
    author_email='mail@tgweber.de',
    url='https://github.com/tgweber/fospred',
    license=license,
    packages=find_packages(exclude=('tests')),
    install_requires=[
        "tensorflow==2.5.0",
        "keras",
        "pandas",
        "scikit-learn==0.21.3"
    ]
)
