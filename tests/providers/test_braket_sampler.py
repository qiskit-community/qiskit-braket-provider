"""Tests for BraketSampler."""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.providers import JobStatus

from qiskit_braket_provider.providers import BraketLocalBackend
from qiskit_braket_provider.providers.braket_sampler import BraketSampler
from qiskit_braket_provider.providers.braket_sampler_job import BraketSamplerJob, _JobMetadata


class TestBraketSampler(TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.backend = BraketLocalBackend()
        self.sampler = BraketSampler(self.backend)

    def test_initialization(self):
        """Test sampler initialization."""
        self.assertIsInstance(self.sampler, BraketSampler)
        self.assertEqual(self.sampler._backend, self.backend)

    def test_different_precisions_raises_error(self):
        """Test that pubs with different shots raise an error."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        with self.assertRaises(ValueError) as context:
            self.sampler.run([(qc, [0, 1, 2, 3], 100), (qc, [0, 1, 2, 3], 200)])

        self.assertIn("same shots", str(context.exception))

    def test_job_status_methods(self):
        """Test job status methods."""
        mock_task = Mock()
        mock_task.id = "test-task-id"
        mock_task.state.return_value = "RUNNING"

        job = BraketSamplerJob(mock_task, _JobMetadata(pubs=[], parameter_indices=[], shots=10000))

        # Test status methods
        self.assertEqual(job.status(), JobStatus.RUNNING)
        self.assertTrue(job.running())
        self.assertFalse(job.done())
        self.assertFalse(job.cancelled())
        self.assertFalse(job.in_final_state())

        # Test completed state
        mock_task.state.return_value = "COMPLETED"
        self.assertEqual(job.status(), JobStatus.DONE)
        self.assertFalse(job.running())
        self.assertTrue(job.done())
        self.assertFalse(job.cancelled())
        self.assertTrue(job.in_final_state())

        # Test cancelled state
        mock_task.state.return_value = "CANCELLED"
        self.assertEqual(job.status(), JobStatus.CANCELLED)
        self.assertFalse(job.running())
        self.assertFalse(job.done())
        self.assertTrue(job.cancelled())
        self.assertTrue(job.in_final_state())

        # Test cancel method
        job.cancel()
        mock_task.cancel.assert_called_once()

        # Test job_id
        self.assertEqual(job.job_id(), "test-task-id")

    def test_run_local(self):
        """Tests that correct results are returned for circuits with multiple registers"""
        theta = Parameter("θ")

        qreg_a = QuantumRegister(9, "qreg_a")
        qreg_b = QuantumRegister(3, "qreg_b")
        creg_a = ClassicalRegister(2, "creg_a")
        creg_b = ClassicalRegister(10, "creg_b")

        chsh_circuit = QuantumCircuit(qreg_a, qreg_b, creg_a, creg_b)
        chsh_circuit.h(0)
        for i in range(11):
            chsh_circuit.cx(i, i + 1)
        chsh_circuit.cx(0, 1)
        chsh_circuit.ry(theta, 0)
        chsh_circuit.measure_all(add_bits=False)
        parameter_values = np.array(  # shape (3, 6)
            [np.linspace(0, 2 * np.pi, 6), np.linspace(0, np.pi, 6), np.linspace(0, np.pi / 2, 6)]
        )
        pub = SamplerPub.coerce((chsh_circuit, parameter_values))

        data = self.sampler.run([pub]).result()[0].data
        for index in np.ndindex(pub.shape):
            results_a = data.creg_a[index]
            shots_a = results_a.num_shots
            for v in results_a.get_int_counts().values():
                self.assertTrue(np.isclose(v / shots_a, 0.5, rtol=0.3, atol=0.2))
            results_b = data.creg_b[index]
            shots_b = results_b.num_shots
            for v in results_b.get_int_counts().values():
                self.assertTrue(np.isclose(v / shots_b, 0.5, rtol=0.3, atol=0.2))
