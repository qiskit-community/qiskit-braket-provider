"""Tests for BraketEstimator."""

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimatorV2, BasePrimitiveJob
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.providers import JobStatus
from qiskit.quantum_info import SparsePauliOp

from braket.program_sets import ProgramSet
from qiskit_braket_provider.providers import BraketLocalBackend
from qiskit_braket_provider.providers.braket_estimator import BraketEstimator
from qiskit_braket_provider.providers.braket_estimator_job import BraketEstimatorJob, _JobMetadata


class TestBraketEstimator(TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.backend = BraketLocalBackend()
        self.estimator = BraketEstimator(self.backend)

    def test_initialization(self):
        """Test estimator initialization."""
        self.assertIsInstance(self.estimator, BraketEstimator)
        self.assertEqual(self.estimator._backend, self.backend)

    def test_simple_pub(self):
        """Test a simple pub with no broadcasting."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        observable = SparsePauliOp(["ZZ"])
        pub = (qc, observable)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            self.assertIsInstance(call_args[0][0], ProgramSet)
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_parameterized_circuit(self):
        """Test with a parameterized circuit."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        observable = SparsePauliOp(["Z"])
        param_values = np.array([[0.0], [np.pi / 4], [np.pi / 2]])
        pub = (qc, observable, param_values)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_multiple_observables(self):
        """Test with multiple observables."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        observables = [SparsePauliOp(["ZZ"]), SparsePauliOp(["XX"])]
        pub = (qc, observables)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            program_set = mock_run.call_args[0][0]
            self.assertIsInstance(program_set, ProgramSet)

    def test_multiple_pubs(self):
        """Test running multiple pubs."""
        qc1 = QuantumCircuit(1)
        qc1.h(0)

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)

        obs1 = SparsePauliOp(["Z"])
        obs2 = SparsePauliOp(["ZZ"])

        pub1 = (qc1, obs1)
        pub2 = (qc2, obs2)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub1, pub2], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_default_precision(self):
        """Test that default precision is used when not specified."""
        qc = QuantumCircuit(1)
        qc.h(0)
        observable = SparsePauliOp(["Z"])
        pub = (qc, observable)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub])

            self.assertIsInstance(job, BasePrimitiveJob)

    def test_custom_precision(self):
        """Test using custom precision."""
        qc = QuantumCircuit(1)
        qc.h(0)
        observable = SparsePauliOp(["Z"])
        pub = (qc, observable)

        custom_precision = 0.05

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=custom_precision)

            self.assertIsInstance(job, BasePrimitiveJob)

    def test_empty_pub_list(self):
        """Test running with empty pub list."""
        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_complex_broadcasting(self):
        """Test with complex broadcasting shapes (2, 3, 6)."""
        theta = Parameter("θ")
        phi = Parameter("φ")
        qc = QuantumCircuit(2)
        qc.ry(theta, 0)
        qc.rx(phi, 1)
        qc.cx(0, 1)

        # Parameter values with shape (3, 6)
        np.random.seed(42)
        param_data = {
            "θ": np.random.uniform(0, 2 * np.pi, size=(3, 6)),
            "φ": np.random.uniform(0, 2 * np.pi, size=(3, 6)),
        }
        parameter_values = BindingsArray(param_data)

        # Observables with shape (2, 3, 1)
        observables = [
            [[SparsePauliOp(["ZZ"])], [SparsePauliOp(["XX"])], [SparsePauliOp(["YY"])]],
            [[SparsePauliOp(["ZI"])], [SparsePauliOp(["IZ"])], [SparsePauliOp(["XI"])]],
        ]

        pub = (qc, observables, parameter_values)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            program_set = mock_run.call_args[0][0]
            self.assertIsInstance(program_set, ProgramSet)

            self.assertIsInstance(job, BasePrimitiveJob)

    def test_broadcasting_with_scalar_observable(self):
        """Test broadcasting with scalar observable and array parameters."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        param_values = np.linspace(0, np.pi, 5)
        observable = SparsePauliOp(["Z"])
        pub = (qc, observable, param_values)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_broadcasting_with_array_observables(self):
        """Test broadcasting with array observables and scalar parameters."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        observables = [
            SparsePauliOp(["ZZ"]),
            SparsePauliOp(["XX"]),
            SparsePauliOp(["YY"]),
            SparsePauliOp(["ZI"]),
        ]

        pub = (qc, observables)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_different_precisions_raises_error(self):
        """Test that pubs with different precisions raise an error."""
        qc = QuantumCircuit(1)
        qc.h(0)
        observable = SparsePauliOp(["Z"])

        # Create pubs with different precisions
        obs_array = ObservablesArray([observable])
        pub1 = EstimatorPub(circuit=qc, observables=obs_array, precision=0.01)
        pub2 = EstimatorPub(circuit=qc, observables=obs_array, precision=0.02)

        with self.assertRaises(ValueError) as context:
            self.estimator.run([pub1, pub2])

        self.assertIn("same precision", str(context.exception))

    def test_non_broadcastable_shapes_raises_error(self):
        """Test that non-broadcastable shapes raise an error."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        # Create observables with shape (3,)
        observables = [SparsePauliOp(["Z"]), SparsePauliOp(["X"]), SparsePauliOp(["Y"])]

        # Create parameter values with shape (2, 1) - not broadcastable with (3,)
        param_values = np.array([[0.0], [np.pi / 4]])

        pub = (qc, observables, param_values)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            with self.assertRaises(ValueError) as context:
                self.estimator.run([pub], precision=0.01)

            self.assertIn("not broadcastable", str(context.exception))

    def test_job_status_methods(self):
        """Test job status methods."""
        mock_task = Mock()
        mock_task.id = "test-task-id"
        mock_task.state.return_value = "RUNNING"

        job = BraketEstimatorJob(
            mock_task, _JobMetadata(pubs=[], pub_metadata=[], precision=0.01, shots=10000)
        )

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

    def test_run_local_pauli_sum(self):
        """Tests that correct results are returned when one observable is a Pauli sum"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(Parameter("theta"), 0)
        circuit.rz(Parameter("phi"), 0)
        circuit.cx(0, 1)
        circuit.h(0)

        params = np.vstack(
            [
                np.linspace(-np.pi, np.pi, 25),
                np.linspace(-4 * np.pi, 4 * np.pi, 25),
            ]
        ).T

        observables = [
            [SparsePauliOp(["XX", "IY"], [0.5, 0.5])],
            [SparsePauliOp("XX")],
            [SparsePauliOp("IY")],
        ]
        observables = [
            [observable.apply_layout(circuit.layout) for observable in observable_set]
            for observable_set in observables
        ]
        estimator_pub = circuit, observables, params

        self.assertTrue(
            np.allclose(
                self.estimator.run([estimator_pub]).result()[0].data.evs,
                BackendEstimatorV2(backend=self.backend).run([estimator_pub]).result()[0].data.evs,
                rtol=0.3,
                atol=0.2,
            )
        )

    def test_run_local_all_pauli_sums(self):
        """Tests that correct results are returned when all observables are Pauli sums"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(Parameter("theta"), 0)
        circuit.rz(Parameter("phi"), 0)
        circuit.cx(0, 1)
        circuit.h(0)

        params = np.vstack(
            [
                np.linspace(-np.pi, np.pi, 25),
                np.linspace(-4 * np.pi, 4 * np.pi, 25),
            ]
        ).T

        observables = [
            [SparsePauliOp(["XX", "IY"], [0.5, 0.5])],
            [SparsePauliOp(["YY", "ZI", "XY"], [0.5, 0.5, 0.1])],
        ]
        observables = [
            [observable.apply_layout(circuit.layout) for observable in observable_set]
            for observable_set in observables
        ]
        estimator_pub = circuit, observables, params

        self.assertTrue(
            np.allclose(
                self.estimator.run([estimator_pub]).result()[0].data.evs,
                BackendEstimatorV2(backend=self.backend).run([estimator_pub]).result()[0].data.evs,
                rtol=0.3,
                atol=0.2,
            )
        )

    def tests_run_local_broadcasting(self):
        """Tests that correct results are returned with broadcasted arrays"""
        theta = Parameter("θ")
        chsh_circuit = QuantumCircuit(3)
        chsh_circuit.h(0)
        chsh_circuit.cx(0, 1)
        chsh_circuit.ry(theta, 0)
        chsh_circuit.h(2)
        parameter_values = np.array(  # shape (3, 6)
            [np.linspace(0, 2 * np.pi, 6), np.linspace(0, np.pi, 6), np.linspace(0, np.pi / 2, 6)]
        )
        observables = [
            [
                [SparsePauliOp(["IZZ"])],
                [SparsePauliOp(["IZX"])],
                [SparsePauliOp(["III"])],
            ],
            [
                [SparsePauliOp(["IXZ"])],
                [SparsePauliOp(["IXX"])],
                [SparsePauliOp(["YYY"])],
            ],
        ]
        estimator_pub = chsh_circuit, observables, parameter_values

        self.assertTrue(
            np.allclose(
                self.estimator.run([estimator_pub]).result()[0].data.evs,
                BackendEstimatorV2(backend=self.backend).run([estimator_pub]).result()[0].data.evs,
                rtol=0.3,
                atol=0.2,
            )
        )
