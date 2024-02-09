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

from .adapter import to_braket, to_qiskit
from .braket_backend import AWSBraketBackend, BraketLocalBackend
from .braket_provider import AWSBraketProvider
from .braket_job import AmazonBraketTask
from .braket_job import AWSBraketJob
