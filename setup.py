import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lingualytics',  
    version='0.1.5',
    author="Rohan Rajpal",
    author_email="rohan46000@gmail.com",
    description="A multilingual text analytics package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohanrajpal/lingualytics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    keywords=['lingualytics', 'nlp', 'texthero', 'torch']
 )