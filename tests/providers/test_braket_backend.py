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
        backend = AWSBraketLocalBackend(name="default")
        self.assertTrue(backend)
        self.assertIsInstance(backend.target, Target)
        self.assertIsNone(backend.max_circuits)
        with self.assertRaises(NotImplementedError):
            backend.drive_channel(0)
        with self.assertRaises(NotImplementedError):
            backend.acquire_channel(0)
        with self.assertRaises(NotImplementedError):
            backend.measure_channel(0)
        with self.assertRaises(NotImplementedError):
            backend.control_channel([0, 1])
