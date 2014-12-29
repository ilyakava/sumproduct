import os

from setuptools import setup, find_packages

def read(*paths):
    """Build a file path from *paths* and return the contents."""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

setup(
    name='sumproduct',
    version='0.0.2',
    description='The sum-product algorithm. Belief propagation (message passing) for factor graphs',
    long_description=(read('README.rst')),
    url='http://github.com/ilyakava/sumproduct/',
    license='MIT',
    author='Ilya Kavalerov',
    author_email='ilyakavalerov@gmail.com',
    packages=find_packages(exclude=['tests*']),
    py_modules=['sumproduct'],
    include_package_data=True,
    classifiers=[
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
