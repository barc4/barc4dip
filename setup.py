from setuptools import find_packages, setup

setup(
    name='barc4dip',
    version='2025.01.09',
    author='Rafael Celestre',
    author_email='rafael.celestre@synchrotron-soleil.fr',
    description='A Python package for digital image processing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/barc4/barc4dip',
    license='CC BY-NC-SA 4.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: CC BY-NC-SA 4.0',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
