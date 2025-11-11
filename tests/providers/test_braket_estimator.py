"""Tests for BraketEstimator."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimatorV2, BasePrimitiveJob
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp

from braket.program_sets import ProgramSet
from qiskit_braket_provider.providers import BraketLocalBackend
from qiskit_braket_provider.providers.braket_estimator import BraketEstimator
from qiskit_braket_provider.providers.braket_estimator_job import BraketEstimatorJob


class TestBraketEstimator(unittest.TestCase):
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

    def test_make_obs_key(self):
        """Test observable key creation."""
        obs = SparsePauliOp(["ZZ"])
        key = BraketEstimator._make_obs_key(obs)

        # Key should be a string representation
        self.assertIsInstance(key, str)

        # Same observable should produce same key
        obs2 = SparsePauliOp(["ZZ"])
        key2 = BraketEstimator._make_obs_key(obs2)
        self.assertEqual(key, key2)

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

    def test_circuit_conversion_to_braket(self):
        """Test that Qiskit circuits are properly converted to Braket."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(np.pi / 4, 0)

        observable = SparsePauliOp(["ZZ"])
        pub = (qc, observable)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            # This should not raise any errors
            self.estimator.run([pub], precision=0.01)
            mock_run.assert_called_once()

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

        metadata = {
            "pubs": [],
            "pub_metadata": [],
            "precision": 0.01,
            "shots": 10000,
        }

        job = BraketEstimatorJob(mock_task, metadata)

        # Test status methods
        self.assertEqual(job.status(), "RUNNING")
        self.assertTrue(job.running())
        self.assertFalse(job.done())
        self.assertFalse(job.cancelled())
        self.assertFalse(job.in_final_state())

        # Test completed state
        mock_task.state.return_value = "COMPLETED"
        self.assertTrue(job.done())
        self.assertTrue(job.in_final_state())
        self.assertFalse(job.running())

        # Test cancelled state
        mock_task.state.return_value = "CANCELLED"
        self.assertTrue(job.cancelled())
        self.assertTrue(job.in_final_state())

        # Test cancel method
        job.cancel()
        mock_task.cancel.assert_called_once()

        # Test job_id
        self.assertEqual(job.job_id(), "test-task-id")

    def test_job_result_reconstruction(self):
        """Test job result reconstruction."""
        qc = QuantumCircuit(1)
        qc.h(0)
        observable = SparsePauliOp(["Z"])
        obs_array = ObservablesArray([observable])
        pub = EstimatorPub(circuit=qc, observables=obs_array, precision=0.01)

        # Create mock task result
        mock_task = Mock()
        mock_task.id = "test-task-id"

        # Create mock result structure
        mock_result_item = Mock()
        mock_result_item.expectation = 0.5
        
        mock_obs_result = Mock()
        mock_obs_result.inputs = [mock_result_item]
        
        mock_binding_result = Mock()
        mock_binding_result.observables = [mock_obs_result]
        mock_binding_result.__getitem__ = Mock(return_value=mock_result_item)

        mock_task_result = [mock_binding_result]
        mock_task.result.return_value = mock_task_result

        metadata = {
            "pubs": [pub],
            "pub_metadata": [
                {
                    "pub_idx": 0,
                    "num_bindings": 1,
                    "broadcast_shape": (),
                    "binding_to_result_map": {
                        0: [(0, 0, 0)]  # List of (position, obs_idx, param_idx)
                    },
                }
            ],
            "precision": 0.01,
            "shots": 10000,
        }

        job = BraketEstimatorJob(mock_task, metadata)

        # Get result
        result = job.result()

        # Verify result structure
        self.assertEqual(len(result), 1)
        pub_result = result[0]
        self.assertIsNotNone(pub_result.data.evs)

        # Test caching - should return same result
        result2 = job.result()
        self.assertIs(result, result2)

    def test_run_local(self):
        # Step 1: Map classical inputs to a quantum problem
        theta = Parameter("θ")
        
        chsh_circuit = QuantumCircuit(2)
        chsh_circuit.h(0)
        chsh_circuit.cx(0, 1)
        chsh_circuit.ry(theta, 0)
        
        number_of_phases = 21
        phases = np.linspace(0, 2 * np.pi, number_of_phases)
        individual_phases = [[ph] for ph in phases]
        
        ops = [
            SparsePauliOp.from_list([("ZZ", 1)]),
            SparsePauliOp.from_list([("ZX", 1)]),
            SparsePauliOp.from_list([("XZ", 1)]),
            SparsePauliOp.from_list([("XX", 1)]),
        ]
        
        # Step 2: Optimize problem for quantum execution.
        isa_observables = [
            operator.apply_layout(chsh_circuit.layout) for operator in ops
        ]
        
        # Step 3: Execute using Qiskit primitives.
        
        # Reshape observable array for broadcasting
        reshaped_ops = np.fromiter(isa_observables, dtype=object).reshape((4, 1))

        # Compare results for the first (and only) PUB
        self.assertTrue(
            np.allclose(
                # TODO: the calls to `reversed` should not be necessary
                [list(reversed(arr)) for arr in self.estimator.run(
                    [(chsh_circuit, reshaped_ops, individual_phases)]
                ).result()[0].data.evs],
                BackendEstimatorV2(backend=self.backend).run(
                    [(chsh_circuit, reshaped_ops, individual_phases)]
                ).result()[0].data.evs,
                rtol=0.3,
                atol=0.2,
            )
        )
