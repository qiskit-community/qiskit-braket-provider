"""
=========================================================
Provider module (:mod:`qiskit_braket_provider.providers`)
=========================================================

.. currentmodule:: qiskit_braket_provider.providers

Provider module contains classes and functions to connect
AWS Braket abstraction to Qiskit architecture.

Provider classes and functions
==============================

.. autosummary::
    :toctree: ../stubs/

    AWSBraketBackend
    BraketLocalBackend
    AWSBraketProvider
    AmazonBraketTask
"""

from .adapter import to_braket as to_braket
from .adapter import to_qiskit as to_qiskit
from .braket_backend import AWSBraketBackend as AWSBraketBackend
from .braket_backend import BraketAwsBackend as BraketAwsBackend
from .braket_backend import BraketLocalBackend as BraketLocalBackend
from .braket_job import AmazonBraketTask as AmazonBraketTask
from .braket_job import AWSBraketJob as AWSBraketJob
from .braket_provider import AWSBraketProvider as AWSBraketProvider
from .braket_provider import BraketProvider as BraketProvider
from .braket_quantum_task import BraketQuantumTask as BraketQuantumTask
