"""Tests for AWS Braket backends."""


from unittest import TestCase
from unittest.mock import Mock

from qiskit.transpiler import Target

from qiskit_braket_plugin.providers import AWSBraketDeviceBackend, AWSBraketLocalBackend
from qiskit_braket_plugin.providers.utils import aws_device_to_target
from tests.providers.test_braket_provider import MOCK_GATE_MODEL_QPU_CAPABILITIES_1


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


class TestAWSBackendTarget(TestCase):
    """Tests target for AWS Braket backend."""

    def test_target(self):
        """Tests target."""
        mock_device = Mock()
        mock_device.properties = MOCK_GATE_MODEL_QPU_CAPABILITIES_1

        target = aws_device_to_target(mock_device)
        self.assertEqual(target.num_qubits, 30)
        self.assertEqual(len(target.operations), 1)
        self.assertEqual(len(target.instructions), 30)
        self.assertIn("Target for AWS Device", target.description)
