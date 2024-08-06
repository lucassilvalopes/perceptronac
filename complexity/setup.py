
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    'ax-platform',
    'numpy',
    'pandas',
    'matplotlib',
    # 'PyQt5'
]

unit_tests_require = [
    'pytest',
]

setuptools.setup(
    name='complexity',
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