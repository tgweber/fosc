from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='fosc',
    version='0.0.3',
    description='Field Of Study Classifier (FOSC)',
    long_description=readme,
    author='Tobias Weber',
    author_email='mail@tgweber.de',
    url='https://github.com/tgweber/fosc',
    license=license,
    packages=find_packages(exclude=('tests')),
    install_requires=[
        "tensorflow",
        "keras",
        "numpy",
        "pandas",
        "scikit-learn"
    ]
)
