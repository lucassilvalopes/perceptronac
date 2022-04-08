
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    'torch==1.7.1',
    'numpy==1.19.5',
    'Pillow==8.2.0',
    'netpbmfile==2021.6.6',
    'pandas==1.1.5',
    'matplotlib==3.3.4',
    'tqdm==4.60.0',
    'open3d==0.13.0',
    'scikit-learn==0.24.1',
    'numba==0.53.1',
    'tika==1.24',
    'pdf2image==1.16.0',
]

unit_tests_require = [
    'pytest',
]

setuptools.setup(
    name='perceptronac',
    version='0.1.0',
    description='Functions useful for (multi-layer) perceptron arithmetic coding.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Digital Image and Video Processing (DIVP) Group',
    license='COPYRIGHT',
    author_email='divp@divp.org',
    url='https://github.com/lucassilvalopes/perceptronac',
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
         'unit': unit_tests_require,
    },
    packages=setuptools.find_packages(exclude=['scripts','tests']),
    python_requires='>=3.6'
)