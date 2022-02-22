"""Setup file for prototype template."""

import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="prototype_template",
    description="Repository for a quantum prototype",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
)
