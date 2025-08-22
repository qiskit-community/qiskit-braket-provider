"""Tests for AWS Braket backends."""

import unittest
from typing import Dict, List
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from botocore import errorfactory
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction as QiskitInstruction
from qiskit.circuit.library import TwoLocal
from qiskit.circuit.random import random_circuit
from qiskit.primitives import BackendEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import Target
from qiskit_algorithms.minimum_eigensolvers import VQE, VQEResult
from qiskit_algorithms.optimizers import SLSQP

from braket.aws import AwsDevice, AwsQuantumTaskBatch
from braket.aws.queue_information import QueueDepthInfo, QueueType
from braket.circuits import Circuit
from braket.device_schema import DeviceActionType
from braket.program_sets import ProgramSet
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit_braket_provider import AWSBraketProvider, exception, version
from qiskit_braket_provider.providers import BraketAwsBackend, BraketLocalBackend
from qiskit_braket_provider.providers.adapter import aws_device_to_target
from qiskit_braket_provider.providers.braket_backend import AWSBraketBackend
from tests.providers.mocks import (
    RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES,
    RIGETTI_MOCK_M_3_QPU_CAPABILITIES,
    MockBraketBackend,
    MockMeasLevelEnum,
)


def combine_dicts(dict1: Dict[str, float], dict2: Dict[str, float]) -> Dict[str, List[float]]:
    """Combines dictionaries with different keys.

    Args:
        dict1: first
        dict2: second

    Returns:
        merged dicts with list of keys
    """
    combined_dict: Dict[str, List[float]] = {}
    for key in dict1.keys():
        if key in combined_dict:
            combined_dict[key].append(dict1[key])
        else:
            combined_dict[key] = [dict1[key]]
    for key in dict2.keys():
        if key in combined_dict:
            combined_dict[key].append(dict2[key])
        else:
            combined_dict[key] = [dict2[key]]
    return combined_dict


class TestBraketBackend(TestCase):
    """Test class for BraketBackend."""

    def test_repr(self):
        """Test the repr method of BraketBackend."""
        backend = BraketLocalBackend(name="default")
        self.assertEqual(repr(backend), "BraketBackend[default]")

    def test_invalid_device(self):
        """Test the device method of BraketBackend."""
        with self.assertRaises(NotImplementedError):
            _ = MockBraketBackend(name="default")._device


class TestBraketAwsBackend(TestCase):
    """Tests BraketBackend."""

    def test_device_backend(self):
        """Tests device backend."""
        device = Mock()
        device.properties = RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES
        backend = BraketAwsBackend(device=device)
        self.assertTrue(backend)
        self.assertIsInstance(backend.target, Target)
        self.assertIsNone(backend.max_circuits)
        user_agent = f"QiskitBraketProvider/{version.__version__}"
        device.aws_session.add_braket_user_agent.assert_called_with(user_agent)
        with self.assertRaises(NotImplementedError):
            backend.dtm()
        with self.assertRaises(NotImplementedError):
            backend.meas_map()
        with self.assertRaises(NotImplementedError):
            backend.qubit_properties(0)
        with self.assertRaises(NotImplementedError):
            backend.drive_channel(0)
        with self.assertRaises(NotImplementedError):
            backend.acquire_channel(0)
        with self.assertRaises(NotImplementedError):
            backend.measure_channel(0)
        with self.assertRaises(NotImplementedError):
            backend.control_channel([0, 1])

    def test_invalid_identifiers(self):
        """Test the invalid identifiers of BraketAwsBackend."""
        with self.assertRaises(ValueError):
            BraketAwsBackend()

        with self.assertRaises(ValueError):
            BraketAwsBackend(arn="some_arn", device="some_device")

    def test_local_backend(self):
        """Tests local backend."""
        backend = BraketLocalBackend(name="default")
        self.assertTrue(backend)
        self.assertIsInstance(backend.target, Target)
        self.assertIsNone(backend.max_circuits)
        with self.assertRaises(NotImplementedError):
            backend.dtm()
        with self.assertRaises(NotImplementedError):
            backend.meas_map()
        with self.assertRaises(NotImplementedError):
            backend.qubit_properties(0)
        with self.assertRaises(NotImplementedError):
            backend.drive_channel(0)
        with self.assertRaises(NotImplementedError):
            backend.acquire_channel(0)
        with self.assertRaises(NotImplementedError):
            backend.measure_channel(0)
        with self.assertRaises(NotImplementedError):
            backend.control_channel([0, 1])

    def test_local_backend_output(self):
        """Test local backend output"""
        first_backend = BraketLocalBackend(name="braket_dm")
        self.assertEqual(first_backend.backend_name, "braket_dm")

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

        results = [backend.run(circuit).result() for circuit in circuits]

        # Result 0
        self.assertEqual(results[0].get_counts(), {"11": 1024})
        # Result 1
        _00 = results[1].get_counts()["00"]
        _11 = results[1].get_counts()["11"]
        self.assertEqual(_00 + _11, 1024)

    def test_local_backend_circuit_shots0(self):
        """Tests local backend with circuit with shots=0."""
        backend = BraketLocalBackend(name="default")

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.x(1)

        result = backend.run(circuit, shots=0).result()

        inv_sqrt_2 = 1 / np.sqrt(2)
        self.assertTrue(
            np.allclose(result.get_statevector(), np.array([0, 0, inv_sqrt_2, inv_sqrt_2]))
        )

    def test_deprecation_warning_on_init(self):
        """Test that a deprecation warning is raised when initializing AWSBraketBackend"""
        mock_aws_device = Mock(spec=AwsDevice)
        mock_aws_device.properties = RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES

        with self.assertWarns(DeprecationWarning):
            AWSBraketBackend(device=mock_aws_device)

    def test_deprecation_warning_on_subclass(self):
        """Test that a deprecation warning is raised when subclassing AWSBraketBackend"""

        with self.assertWarns(DeprecationWarning):

            class SubclassAWSBraketBackend(AWSBraketBackend):  # pylint: disable=unused-variable
                """A subclass of AWSBraketBackend for testing purposes"""

                pass

    def test_run_multiple_circuits(self):
        """Tests run with multiple circuits"""
        device = Mock()
        device.properties = RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES
        backend = BraketAwsBackend(device=device)
        mock_task_1 = Mock(spec=LocalQuantumTask)
        mock_task_1.id = "0"
        mock_task_2 = Mock(spec=LocalQuantumTask)
        mock_task_2.id = "1"
        mock_batch = Mock(spec=AwsQuantumTaskBatch)
        mock_batch.tasks = [mock_task_1, mock_task_2]
        backend._device.run_batch = Mock(return_value=mock_batch)
        circuit = QuantumCircuit(1)
        circuit.h(0)

        backend.run([circuit, circuit], shots=0, meas_level=2)

        braket_circuit = Circuit().h(0)
        device.run_batch.assert_called_with([braket_circuit, braket_circuit], shots=0)
        device.run_batch.assert_called_once()

    def test_run_multiple_circuits_program_set(self):
        """Tests run with multiple circuits"""
        device = Mock()
        device.properties = RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES
        device.properties.action["braket.ir.openqasm.program_set"] = {
            "actionType": "braket.ir.openqasm.program_set",
            "version": ["1"],
            "maximumExecutables": 500,
            "maximumTotalShots": 1000000,
        }
        backend = BraketAwsBackend(device=device)
        backend._device.run = Mock(return_value=Mock(spec=LocalQuantumTask))
        circuit = QuantumCircuit(1)
        circuit.h(0)

        backend.run([circuit, circuit], shots=5, meas_level=2)

        braket_circuit = Circuit().h(0)
        device.run.assert_called_with(
            ProgramSet([braket_circuit, braket_circuit], shots_per_executable=5)
        )
        device.run.assert_called_once()

    def test_run_invalid_run_input(self):
        """Tests run with invalid input to run"""
        device = Mock()
        device.properties = RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES
        backend = BraketAwsBackend(device=device)
        with self.assertRaises(exception.QiskitBraketException):
            backend.run(1, shots=0)

    @patch(
        "braket.devices.LocalSimulator.run",
        side_effect=[
            Mock(return_value=Mock(id="0", spec=LocalQuantumTask)),
            Exception("Mock exception"),
        ],
    )
    def test_local_backend_run_exception(self, braket_devices_run):
        """Tests local backend with exception thrown during second run"""
        backend = BraketLocalBackend(name="default")

        circuit = QuantumCircuit(1)
        circuit.h(0)

        with self.assertRaises(Exception):
            backend.run([circuit, circuit], shots=0)  # First run should pass
        braket_devices_run.assert_called()

    def test_meas_level_enum(self):
        """Check that enum meas level can be successfully accessed without error"""
        backend = BraketLocalBackend(name="default")
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        backend.run(circuit, shots=10, meas_level=MockMeasLevelEnum.LEVEL_TWO)

    def test_meas_level_2(self):
        """Check that there's no error for asking for classified measurement results."""
        backend = BraketLocalBackend(name="default")
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        backend.run(circuit, shots=10, meas_level=2)

    def test_meas_level_1(self):
        """Check that there's an exception for asking for raw measurement results."""
        backend = BraketLocalBackend(name="default")
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with self.assertRaises(exception.QiskitBraketException):
            backend.run(circuit, shots=10, meas_level=1)

    def test_vqe(self):
        """Tests VQE."""
        local_simulator = BraketLocalBackend(name="default")
        h2_op = SparsePauliOp(
            ["II", "IZ", "ZI", "ZZ", "XX"],
            coeffs=[
                -1.052373245772859,
                0.39793742484318045,
                -0.39793742484318045,
                -0.01128010425623538,
                0.18093119978423156,
            ],
        )

        estimator = BackendEstimator(backend=local_simulator, skip_transpilation=False)

        ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
        slsqp = SLSQP(maxiter=1)

        vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=slsqp)

        result = vqe.compute_minimum_eigenvalue(h2_op)

        self.assertIsInstance(result, VQEResult)
        self.assertEqual(len(result.optimal_parameters), 8)
        self.assertEqual(len(list(result.optimal_point)), 8)

    def test_random_circuits(self):
        """Tests with random circuits."""
        backend = BraketLocalBackend(name="braket_sv")

        for i in range(1, 10):
            circuit = random_circuit(i, 5, seed=42)
            qiskit_sv = Statevector(circuit)
            with self.subTest(f"Random circuit with {i} qubits and 0 shots."):
                self.assertTrue(
                    np.allclose(
                        backend.run(circuit, shots=0).result().get_statevector(),
                        qiskit_sv.data,
                    )
                )
            with self.subTest(f"Random circuit with {i} qubits and {(shots := 10000)} shots."):
                braket_result = backend.run(circuit, shots=shots).result().get_counts()
                qiskit_result = qiskit_sv.probabilities_dict()

                combined_results = combine_dicts(
                    {k: float(v) / shots for k, v in braket_result.items()},
                    qiskit_result,
                )

                for key, values in combined_results.items():
                    if len(values) == 1:
                        self.assertTrue(
                            values[0] < 0.05,
                            f"Missing {key} key in one of the results.",
                        )
                    else:
                        percent_diff = abs(((float(values[0]) - values[1]) / values[0]) * 100)
                        abs_diff = abs(values[0] - values[1])
                        self.assertTrue(
                            percent_diff < 10 or abs_diff < 0.05,
                            f"Key {key} with percent difference {percent_diff} "
                            f"and absolute difference {abs_diff}. Original values {values}",
                        )

    @patch("qiskit_braket_provider.providers.braket_backend.AwsQuantumTask")
    @patch("qiskit_braket_provider.providers.braket_backend.BraketQuantumTask")
    def test_retrieve_job_task_ids(self, mock_braket_quantum_task, mock_aws_quantum_task):
        """Test method for retrieving job task IDs."""
        device = Mock()
        device.properties = RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES
        backend = BraketAwsBackend(device=device)
        task_id = "task1;task2;task3"
        expected_task_ids = task_id.split(";")

        backend.retrieve_job(task_id)

        # Assert
        mock_aws_quantum_task.assert_any_call(arn=expected_task_ids[0])
        mock_aws_quantum_task.assert_any_call(arn=expected_task_ids[1])
        mock_aws_quantum_task.assert_any_call(arn=expected_task_ids[2])
        mock_braket_quantum_task.assert_called_once_with(
            task_id=task_id,
            backend=backend,
            tasks=[mock_aws_quantum_task(arn=task_id) for task_id in expected_task_ids],
        )

    @unittest.skip("Call to external resources.")
    def test_retrieve_job(self):
        """Tests retrieve task by id."""
        backend = AWSBraketProvider().get_backend("SV1")
        circuits = [
            transpile(random_circuit(3, 2, seed=seed), backend=backend, seed_transpiler=42)
            for seed in range(3)
        ]
        job = backend.run(circuits, shots=10)
        task_id = job.task_id()
        retrieved_job = backend.retrieve_job(task_id)

        job_result = job.result()
        retrieved_job_result = retrieved_job.result()

        self.assertEqual(job_result.task_id, retrieved_job_result.task_id)
        self.assertEqual(job_result.status, retrieved_job_result.status)
        self.assertEqual(job_result.backend_version, retrieved_job_result.backend_version)
        self.assertEqual(job_result.backend_name, retrieved_job_result.backend_name)

    @unittest.skip("Call to external resources.")
    def test_running_incompatible_verbatim_circuit_on_aspen_raises_error(self):
        """Tests working of verbatim=True and disable_qubit_rewiring=True.

        Note that in case of Rigetti devices, both of those parameters are
        needed if one wishes to run instructions wrapped in verbatim boxes.
        """
        device = AWSBraketProvider().get_backend("Aspen-M-2")
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)

        with self.assertRaises(errorfactory.ClientError):
            device.run(circuit, verbatim=True, disable_qubit_rewiring=True)

    @unittest.skip("Call to external resources.")
    def test_running_circuit_with_disabled_rewiring_requires_matching_topolog(self):
        """Tests working of disable_qubit_rewiring=True."""
        device = AWSBraketProvider().get_backend("Aspen-M-2")
        circuit = QuantumCircuit(4)
        circuit.cz(0, 3)

        with self.assertRaises(errorfactory.ClientError):
            device.run(circuit, disable_qubit_rewiring=True)

    @unittest.skip("Call to external resources.")
    def test_native_circuits_with_measurements_can_be_run_in_verbatim_mode(self):
        """Tests running circuit with measurement in verbatim mode."""
        backend = AWSBraketProvider().get_backend("Aspen-M-2")
        circuit = QuantumCircuit(2)
        circuit.cz(0, 1)
        circuit.measure_all()

        result = backend.run(circuit, shots=10, verbatim=True, disable_qubit_rewiring=True).result()

        self.assertEqual(sum(result.get_counts().values()), 10)

    @patch("qiskit_braket_provider.providers.braket_backend.to_braket")
    def test_native_transpilation(self, mock_to_braket):
        """Tests running circuit with native mode"""
        mock_device = Mock()
        mock_device.properties = RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES
        mock_device.properties.paradigm.connectivity.connectivityGraph = {
            "0": ["1"],
            "1": ["0", "2"],
            "2": ["1"],
        }
        mock_device.properties.paradigm.nativeGateSet = ["rx", "rz", "cnot"]

        mock_batch = Mock()
        mock_batch.tasks = [Mock(id="abcd1234")]
        mock_device.run_batch.return_value = mock_batch

        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 2)

        backend = AWSBraketBackend(device=mock_device)

        backend.run(circuit, native=True)
        assert mock_to_braket.call_args.kwargs["basis_gates"] == {"rx", "rz", "cx"}
        assert mock_to_braket.call_args.kwargs["connectivity"] == [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
        ]

        backend.run(circuit, verbatim=True)
        assert mock_to_braket.call_args.kwargs["basis_gates"] is None
        assert mock_to_braket.call_args.kwargs["verbatim"] is True
        assert mock_to_braket.call_args.kwargs["connectivity"] is None

    @patch("qiskit_braket_provider.providers.braket_provider.AwsDevice")
    def test_queue_depth(self, mocked_device):
        """Tests queue depth."""

        mock_return_value = QueueDepthInfo(
            quantum_tasks={QueueType.NORMAL: "19", QueueType.PRIORITY: "3"},
            jobs="0 (3 prioritized job(s) running)",
        )
        mocked_device.properties = RIGETTI_MOCK_M_3_QPU_CAPABILITIES
        mocked_device.queue_depth.return_value = mock_return_value
        backend = BraketAwsBackend(device=mocked_device)
        result = backend.queue_depth()

        mocked_device.queue_depth.assert_called_once()
        assert isinstance(result, QueueDepthInfo)
        self.assertEqual(result, mock_return_value)


class TestAWSBackendTarget(TestCase):
    """Tests target for AWS Braket backend."""

    def test_target(self):
        """Tests target."""
        mock_device = Mock()
        mock_device.properties = RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES

        target = aws_device_to_target(mock_device)
        self.assertEqual(target.num_qubits, 30)
        self.assertEqual(len(target.operations), 2)
        self.assertEqual(len(target.instructions), 60)
        self.assertIn("Target for Amazon Braket QPU", target.description)

    def test_target_invalid_device(self):
        """Tests target."""
        mock_device = Mock()
        mock_device.properties = None

        with self.assertRaises(exception.QiskitBraketException):
            aws_device_to_target(mock_device)

    def test_fully_connected(self):
        """Tests if instruction_props is correctly populated for fully connected topology."""
        mock_device = Mock()
        mock_device.properties = RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES.copy(deep=True)
        mock_device.properties.paradigm.connectivity.fullyConnected = True
        mock_device.properties.paradigm.qubitCount = 2
        mock_device.properties.action.get(DeviceActionType.OPENQASM).supportedOperations = ["CNOT"]

        instruction_props = aws_device_to_target(mock_device)

        cx_instruction = QiskitInstruction(name="cx", num_qubits=2, num_clbits=0, params=[])
        measure_instruction = QiskitInstruction(
            name="measure", num_qubits=1, num_clbits=1, params=[]
        )

        expected_instruction_props = [
            (cx_instruction, (0, 1)),
            (cx_instruction, (1, 0)),
            (measure_instruction, (0,)),
            (measure_instruction, (1,)),
        ]
        for index, instruction in enumerate(instruction_props.instructions):
            self.assertEqual(
                instruction[0].num_qubits,
                expected_instruction_props[index][0].num_qubits,
            )
            self.assertEqual(
                instruction[0].num_clbits,
                expected_instruction_props[index][0].num_clbits,
            )
            self.assertEqual(instruction[0].params, expected_instruction_props[index][0].params)
            self.assertEqual(instruction[0].name, expected_instruction_props[index][0].name)

            self.assertEqual(instruction[1], expected_instruction_props[index][1])
