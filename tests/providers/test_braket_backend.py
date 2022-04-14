"""Tests for AWS Braket backends."""


from unittest import TestCase
from unittest.mock import Mock

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import Target

from qiskit_braket_plugin.providers import AWSBraketBackend, BraketLocalBackend
from qiskit_braket_plugin.providers.utils import aws_device_to_target
from tests.providers.mocks import RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES


class TestAWSBraketBackend(TestCase):
    """Tests BraketBackend."""

    def test_device_backend(self):
        """Tests device backend."""
        device = Mock()
        device.properties = RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES
        backend = AWSBraketBackend(device)
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

    def test_local_backend(self):
        """Tests local backend."""
        backend = BraketLocalBackend(name="default")
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

    def test_local_backend_circuit(self):
        """Tests local backend with circuit."""
        backend = BraketLocalBackend(name="default")
        circuits = []

        # Circuit 0
        q_c = QuantumCircuit(2)
        q_c.x(0)
        q_c.cx(0, 1)
        circuits.append(q_c)

        # Circuit 1
        q_c = QuantumCircuit(2)
        q_c.h(0)
        q_c.cx(0, 1)
        circuits.append(q_c)

        results = []
        for circuit in circuits:
            results.append(backend.run(circuit).result())

        # Result 0
        self.assertEqual(results[0].get_counts(), {"11": 1024})
        # Result 1
        _00 = results[1].get_counts()["00"]
        _11 = results[1].get_counts()["11"]
        self.assertEqual(_00 + _11, 1024)

    def test_random_circuits(self):
        """Tests with random circuits."""
        backend = BraketLocalBackend(name="default")

        for i in range(10):
            circuit = random_circuit(i + 1, 5, seed=42)
            transpiled_circuit = transpile(circuit, backend=backend)
            result = backend.run(transpiled_circuit).result().get_counts()
            self.assertIsInstance(result, dict)


class TestAWSBackendTarget(TestCase):
    """Tests target for AWS Braket backend."""

    def test_target(self):
        """Tests target."""
        mock_device = Mock()
        mock_device.properties = RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES

        target = aws_device_to_target(mock_device)
        self.assertEqual(target.num_qubits, 30)
        self.assertEqual(len(target.operations), 1)
        self.assertEqual(len(target.instructions), 30)
        self.assertIn("Target for AWS Device", target.description)
