# Qiskit-Braket Installation Guide

## Installing Depencencies

```shell
pip install -r requirements.txt
```

## Installing Optional Dependencies

```shell
pip install -r requirements-dev.txt
```
## Installing Qiskit-Braket plugin

```shell
pip install .
```

## Testing the Installation

```shell
tox -epy39
tox -elint
tox -ecoverage
```
