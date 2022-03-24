"""Tests for AWS Braket backends."""


from unittest import TestCase
from unittest.mock import Mock

from qiskit.transpiler import Target

from qiskit_braket_plugin.providers import AWSBraketDeviceBackend, AWSBraketLocalBackend


class TestAWSBraketBackend(TestCase):
    """Tests AWSBraketBackend."""

    def test_device_backend(self):
        """Tests device backend."""
        device = Mock()
        backend = AWSBraketDeviceBackend(device)
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
