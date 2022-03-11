"""Setup file for Qiskit-Braket plugin."""

import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="qiskit_braket_plugin",
    description="Qiskit-Braket plugin to execute Qiskit "
    "programs on AWS quantum computing "
    "hardware devices through Amazon Braket.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
    version="0.0.1",
)
