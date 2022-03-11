"""Tests for AWS Braket backends."""


from unittest import TestCase

from qiskit.transpiler import Target

from qiskit_braket_plugin.providers import AWSBraketDeviceBackend, AWSBraketLocalBackend


class TestAWSBraketBackend(TestCase):
    """Tests AWSBraketBackend."""

    def test_device_backend(self):
        """Tests device backend."""
        backend = AWSBraketDeviceBackend()
        self.assertTrue(backend)
        self.assertIsInstance(backend.target, Target)
        self.assertIsNone(backend.max_circuits)
        self.assertIsNone(backend.meas_map)

    def test_local_backend(self):
        """Tests local backend."""
        backend = AWSBraketLocalBackend()
        self.assertTrue(backend)
        self.assertIsInstance(backend.target, Target)
        self.assertIsNone(backend.max_circuits)
        self.assertIsNone(backend.meas_map)
