"""Tests for AWS Braket backends."""
import unittest
from unittest import TestCase
from unittest.mock import Mock
import pkg_resources

from qiskit import QuantumCircuit, transpile, BasicAer
from qiskit.algorithms import VQE, VQEResult
from qiskit.algorithms.optimizers import (
    SLSQP,
)
from qiskit.circuit.library import TwoLocal
from qiskit.circuit.random import random_circuit
from qiskit.opflow import (
    I,
    X,
    Z,
)
from qiskit.result import Result
from qiskit.transpiler import Target
from qiskit.utils import QuantumInstance

from qiskit_braket_provider import AWSBraketProvider
from qiskit_braket_provider.providers import AWSBraketBackend, BraketLocalBackend
from qiskit_braket_provider.providers.adapter import aws_device_to_target
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
        user_agent = (
            f"QiskitBraketProvider/"
            f"{pkg_resources.get_distribution('qiskit-braket-provider').version}"
        )
        device.aws_session.add_braket_user_agent.assert_called_with(user_agent)
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

    def test_vqe(self):
        """Tests VQE."""
        local_simulator = BraketLocalBackend(name="default")

        h2_op = (
            (-1.052373245772859 * I ^ I)
            + (0.39793742484318045 * I ^ Z)
            + (-0.39793742484318045 * Z ^ I)
            + (-0.01128010425623538 * Z ^ Z)
            + (0.18093119978423156 * X ^ X)
        )

        quantum_instance = QuantumInstance(
            local_simulator, seed_transpiler=42, seed_simulator=42
        )
        ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
        slsqp = SLSQP(maxiter=1)

        vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=quantum_instance)

        result = vqe.compute_minimum_eigenvalue(h2_op)

        self.assertIsInstance(result, VQEResult)
        self.assertEqual(len(result.optimal_parameters), 8)
        self.assertEqual(len(list(result.optimal_point)), 8)

    def test_random_circuits(self):
        """Tests with random circuits."""
        backend = BraketLocalBackend(name="braket_sv")
        aer_backend = BasicAer.get_backend("statevector_simulator")

        for i in range(1, 10):
            with self.subTest(f"Random circuit with {i} qubits."):
                circuit = random_circuit(i, 5, seed=42)
                braket_transpiled_circuit = transpile(
                    circuit, backend=backend, seed_transpiler=42
                )
                braket_result = (
                    backend.run(braket_transpiled_circuit, shots=1000)
                    .result()
                    .get_counts()
                )

                transpiled_aer_circuit = transpile(
                    circuit, backend=aer_backend, seed_transpiler=42
                )
                aer_result = (
                    aer_backend.run(transpiled_aer_circuit, shots=1000)
                    .result()
                    .get_counts()
                )

                self.assertEqual(
                    sorted([k for k, v in braket_result.items() if v > 50]),
                    sorted([k for k, v in aer_result.items() if v > 0.05]),
                )
                self.assertIsInstance(braket_result, dict)

    @unittest.skip("Call to external resources.")
    def test_retrieve_job(self):
        """Tests retrieve job by id."""
        backend = AWSBraketProvider().get_backend("SV1")
        circuits = [
            transpile(
                random_circuit(3, 2, seed=seed), backend=backend, seed_transpiler=42
            )
            for seed in range(3)
        ]
        job = backend.run(circuits, shots=10)
        job_id = job.job_id()
        retrieved_job = backend.retrieve_job(job_id)

        job_result: Result = job.result()
        retrieved_job_result: Result = retrieved_job.result()

        self.assertEqual(job_result.job_id, retrieved_job_result.job_id)
        self.assertEqual(job_result.status, retrieved_job_result.status)
        self.assertEqual(
            job_result.backend_version, retrieved_job_result.backend_version
        )
        self.assertEqual(job_result.backend_name, retrieved_job_result.backend_name)


class TestAWSBackendTarget(TestCase):
    """Tests target for AWS Braket backend."""

    def test_target(self):
        """Tests target."""
        mock_device = Mock()
        mock_device.properties = RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES

        target = aws_device_to_target(mock_device)
        self.assertEqual(target.num_qubits, 30)
        self.assertEqual(len(target.operations), 2)
        self.assertEqual(len(target.instructions), 31)
        self.assertIn("Target for AWS Device", target.description)
