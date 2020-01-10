import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NNS-RPY2-ScikitLearn",
    version="0.0.1",
    author="Fred Viole, Roberto Spadim",
    author_email="fredviole@gmail.com, roberto@spadim.com.br",
    description="NNS Regressor and Classifier, using Scikit Learn interface and RPY2 bridge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rspadim/NNS-RPY2-ScikitLearn",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "rpy2",
        "scikit_learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
