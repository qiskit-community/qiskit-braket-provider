"""Tests for Qiskti to Braket adapter."""
from unittest import TestCase

from braket.circuits import Circuit, FreeParameter, observables
from braket.devices import LocalSimulator

import numpy as np

from qiskit import QuantumCircuit, execute, BasicAer
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import I, Z, X

from qiskit.circuit.library.standard_gates import (
    HGate,
    CHGate,
    IGate,
    PhaseGate,
    CPhaseGate,
    RGate,
    RXGate,
    CRXGate,
    RXXGate,
    RYGate,
    CRYGate,
    RYYGate,
    RZGate,
    CRZGate,
    RZZGate,
    RZXGate,
    XXMinusYYGate,
    XXPlusYYGate,
    ECRGate,
    SGate,
    SdgGate,
    CSGate,
    CSdgGate,
    SwapGate,
    CSwapGate,
    iSwapGate,
    SXGate,
    SXdgGate,
    CSXGate,
    DCXGate,
    TGate,
    TdgGate,
    UGate,
    U1Gate,
    CU1Gate,
    U2Gate,
    U3Gate,
    CU3Gate,
    XGate,
    CXGate,
    CCXGate,
    C3SXGate,
    RCCXGate,
    RC3XGate,
    YGate,
    CYGate,
    ZGate,
    CZGate,
    CCZGate,
)

from qiskit_braket_provider.providers.adapter import (
    convert_qiskit_to_braket_circuit,
    qiskit_gate_name_to_braket_gate_mapping,
    qiskit_gate_names_to_braket_gates,
    qiskit_to_braket_gate_names_mapping,
    wrap_circuits_in_verbatim_box,
)

from tests.providers.test_braket_backend import combine_dicts

_EPS = 1e-10  # global variable used to chop very small numbers to zero

std_gate_qubits_list = [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]

standard_gates = [
    IGate(),
    SXGate(),
    XGate(),
    CXGate(),
    RZGate(Parameter("λ")),
    RGate(Parameter("ϴ"), Parameter("φ")),
    C3SXGate(),
    CCXGate(),
    DCXGate(),
    CHGate(),
    CPhaseGate(Parameter("ϴ")),
    CRXGate(Parameter("ϴ")),
    CRYGate(Parameter("ϴ")),
    CRZGate(Parameter("ϴ")),
    CSwapGate(),
    CSXGate(),
    # CUGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ"), Parameter("γ")), qiskit's
    # assign_parameter method not working for this gate as of 6/13/2023
    CU1Gate(Parameter("λ")),
    CU3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
    CYGate(),
    CZGate(),
    CCZGate(),
    HGate(),
    PhaseGate(Parameter("ϴ")),
    RCCXGate(),
    RC3XGate(),
    RXGate(Parameter("ϴ")),
    RXXGate(Parameter("ϴ")),
    RYGate(Parameter("ϴ")),
    RYYGate(Parameter("ϴ")),
    RZZGate(Parameter("ϴ")),
    # RZXGate(Parameter("ϴ")), braket treats q0 as q1 and q0 as q1 after translating with this gate
    XXMinusYYGate(Parameter("ϴ")),
    XXPlusYYGate(Parameter("ϴ")),
    # ECRGate(), braket treats q0 as q1 and q0 as q1 after translating with this gate
    SGate(),
    SdgGate(),
    CSGate(),
    CSdgGate(),
    SwapGate(),
    iSwapGate(),
    SXdgGate(),
    TGate(),
    TdgGate(),
    UGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
    U1Gate(Parameter("λ")),
    U2Gate(Parameter("φ"), Parameter("λ")),
    U3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
    YGate(),
    ZGate(),
]

flipped_gates = [RZXGate(Parameter("ϴ")), ECRGate()]


class TestAdapter(TestCase):
    """Tests adapter."""

    def test_state_preparation_01(self):
        """Tests state_preparation handling of Adapter"""
        input_state_vector = np.array([np.sqrt(3) / 2, np.sqrt(2) * complex(1, 1) / 4])

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.prepare_state(input_state_vector, 0)

        braket_circuit = convert_qiskit_to_braket_circuit(qiskit_circuit)
        braket_circuit.state_vector()  # pylint: disable=no-member
        result = LocalSimulator().run(braket_circuit)
        output_state_vector = np.array(result.result().values[0])

        self.assertTrue(
            (np.linalg.norm(input_state_vector - output_state_vector)) < _EPS
        )

    def test_state_preparation_00(self):
        """Tests state_preparation handling of Adapter"""
        input_state_vector = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.prepare_state(input_state_vector, 0)

        braket_circuit = convert_qiskit_to_braket_circuit(qiskit_circuit)
        braket_circuit.state_vector()  # pylint: disable=no-member
        result = LocalSimulator().run(braket_circuit)
        output_state_vector = np.array(result.result().values[0])

        self.assertTrue(
            (np.linalg.norm(input_state_vector - output_state_vector)) < _EPS
        )

    def test_u_gate(self):
        """Tests adapter conversion of u gate"""
        qiskit_circuit = QuantumCircuit(1)
        backend = BasicAer.get_backend("statevector_simulator")
        device = LocalSimulator()
        for _ in range(8):
            qiskit_circuit.u(np.pi / 2, np.pi / 3, np.pi / 4, 0)

            job = execute(qiskit_circuit, backend)

            braket_circuit = convert_qiskit_to_braket_circuit(qiskit_circuit)
            braket_circuit.state_vector()  # pylint: disable=no-member

            braket_output = device.run(braket_circuit).result().values[0]
            qiskit_output = np.array(job.result().get_statevector(qiskit_circuit))

            self.assertTrue(np.linalg.norm(braket_output - qiskit_output) < _EPS)

    def test_standard_gate_decomp(self):
        """Tests adapter decomposition of all standard gates to forms that can be translated"""
        aer_backend = BasicAer.get_backend("statevector_simulator")
        device = LocalSimulator()

        for standard_gate in standard_gates:
            with self.subTest(f"Circuit with {standard_gate.name} gate."):
                qiskit_circuit = QuantumCircuit(standard_gate.num_qubits)
                qiskit_circuit.append(standard_gate, range(standard_gate.num_qubits))

                parameters = standard_gate.params
                if parameters:
                    parameter_values = [
                        (137 / 61) * np.pi / i for i in range(1, len(parameters) + 1)
                    ]
                    parameter_bindings = dict(zip(parameters, parameter_values))
                    qiskit_circuit = qiskit_circuit.assign_parameters(
                        parameter_bindings
                    )

                braket_circuit = convert_qiskit_to_braket_circuit(qiskit_circuit)

                task = device.run(braket_circuit, shots=1000)
                braket_counts = task.result().measurement_counts
                braket_result = {}
                for measurement in list(braket_counts.keys()):
                    braket_result[measurement] = braket_counts[measurement]

                qiskit_job = execute(qiskit_circuit, aer_backend, shots=1000)
                qiskit_result = qiskit_job.result().get_counts()

                combined_results = combine_dicts(
                    {k: float(v) / 1000.0 for k, v in braket_result.items()},
                    qiskit_result,
                )

                for key, values in combined_results.items():
                    if len(values) == 1:
                        self.assertTrue(
                            values[0] < 0.05,
                            f"Missing {key} key in one of the results.",
                        )
                    else:
                        percent_diff = abs(
                            ((float(values[0]) - values[1]) / values[0]) * 100
                        )
                        abs_diff = abs(values[0] - values[1])
                        self.assertTrue(
                            percent_diff < 10 or abs_diff < 0.05,
                            f"Key {key} with percent difference {percent_diff} "
                            f"and absolute difference {abs_diff}. Original values {values}",
                        )

    def test_flipped_gate_decomp(self):
        """Tests adapter translation of gates which flip q0 and q1 when translated"""
        aer_backend = BasicAer.get_backend("statevector_simulator")
        device = LocalSimulator()

        for gate in flipped_gates:
            with self.subTest(f"Circuit with {gate.name} gate."):
                qiskit_circuit = QuantumCircuit(2)
                qiskit_circuit.append(gate, range(2))
                parameters = gate.params
                if parameters:
                    parameter_values = [
                        (137 / 61) * np.pi / i for i in range(1, len(parameters) + 1)
                    ]
                    parameter_bindings = dict(zip(parameters, parameter_values))
                    qiskit_circuit = qiskit_circuit.assign_parameters(
                        parameter_bindings
                    )
                braket_circuit = convert_qiskit_to_braket_circuit(qiskit_circuit)

                task = device.run(braket_circuit, shots=1000)
                braket_counts = task.result().measurement_counts
                braket_result = {}
                for measurement in list(braket_counts.keys()):
                    braket_result[measurement[::-1]] = braket_counts[measurement]

                qiskit_job = execute(qiskit_circuit, aer_backend, shots=1000)
                qiskit_result = qiskit_job.result().get_counts()

                combined_results = combine_dicts(
                    {k: float(v) / 1000.0 for k, v in braket_result.items()},
                    qiskit_result,
                )

                for key, values in combined_results.items():
                    if len(values) == 1:
                        self.assertTrue(
                            values[0] < 0.05,
                            f"Missing {key} key in one of the results.",
                        )
                    else:
                        percent_diff = abs(
                            ((float(values[0]) - values[1]) / values[0]) * 100
                        )
                        abs_diff = abs(values[0] - values[1])
                        self.assertTrue(
                            percent_diff < 10 or abs_diff < 0.05,
                            f"Key {key} with percent difference {percent_diff} "
                            f"and absolute difference {abs_diff}. Original values {values}",
                        )

    def test_exponential_gate_decomp(self):
        """Tests adapter translation of exponential gates(note that translation
        flips q0 and q1 here as well)"""
        aer_backend = BasicAer.get_backend("statevector_simulator")
        device = LocalSimulator()
        qiskit_circuit = QuantumCircuit(2)

        operator = (Z ^ Z) - 0.1 * (X ^ I)
        evo = PauliEvolutionGate(operator, time=2)

        qiskit_circuit.append(evo, range(2))
        braket_circuit = convert_qiskit_to_braket_circuit(qiskit_circuit)

        task = device.run(braket_circuit, shots=1000)
        braket_counts = task.result().measurement_counts
        braket_result = {}
        for measurement in list(braket_counts.keys()):
            braket_result[measurement[::-1]] = braket_counts[measurement]

        qiskit_job = execute(qiskit_circuit, aer_backend, shots=1000)
        qiskit_result = qiskit_job.result().get_counts()

        combined_results = combine_dicts(
            {k: float(v) / 1000.0 for k, v in braket_result.items()}, qiskit_result
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

    def test_mappers(self):
        """Tests mappers."""
        self.assertEqual(
            list(sorted(qiskit_to_braket_gate_names_mapping.keys())),
            list(sorted(qiskit_gate_names_to_braket_gates.keys())),
        )

        self.assertEqual(
            list(sorted(qiskit_to_braket_gate_names_mapping.values())),
            list(sorted(qiskit_gate_name_to_braket_gate_mapping.keys())),
        )

    def test_convert_parametric_qiskit_to_braket_circuit(self):
        """Tests convert_qiskit_to_braket_circuit works with parametric circuits."""

        theta = Parameter("θ")
        phi = Parameter("φ")
        lam = Parameter("λ")
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.rz(theta, 0)
        qiskit_circuit.u(theta, phi, lam, 0)
        qiskit_circuit.u(theta, phi, np.pi, 0)
        braket_circuit = convert_qiskit_to_braket_circuit(qiskit_circuit)

        braket_circuit_ans = (
            Circuit()  # pylint: disable=no-member
            .rz(0, FreeParameter("θ"))
            .phaseshift(0, FreeParameter("λ"))
            .ry(0, FreeParameter("θ"))
            .phaseshift(0, FreeParameter("φ"))
            .phaseshift(0, np.pi)
            .ry(0, FreeParameter("θ"))
            .phaseshift(0, FreeParameter("φ"))
        )

        self.assertEqual(braket_circuit, braket_circuit_ans)

    def test_sample_result_type(self):
        """Tests sample result type with observables Z"""

        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.h(0)
        qiskit_circuit.cnot(0, 1)
        qiskit_circuit.measure(0, 0)
        braket_circuit = convert_qiskit_to_braket_circuit(qiskit_circuit)

        circuits = (
            Circuit()  # pylint: disable=no-member
            .h(0)
            .cnot(0, 1)
            .sample(observable=observables.Z(), target=0)
        )

        self.assertEqual(braket_circuit, circuits)


class TestVerbatimBoxWrapper(TestCase):
    """Test wrapping in Verbatim box."""

    def test_wrapped_circuits_have_one_instruction_equivalent_to_original_one(self):
        """Test circuits wrapped in verbatim box have correct instructions."""
        circuits = [
            Circuit().rz(1, 0.1).cz(0, 1).rx(0, 0.1),  # pylint: disable=no-member
            Circuit().cz(0, 1).cz(1, 2),  # pylint: disable=no-member
        ]

        wrapped_circuits = wrap_circuits_in_verbatim_box(circuits)

        # Verify circuits comprise of verbatim box
        self.assertTrue(
            all(
                wrapped.instructions[0].operator.name == "StartVerbatimBox"
                for wrapped in wrapped_circuits
            )
        )

        self.assertTrue(
            all(
                wrapped.instructions[-1].operator.name == "EndVerbatimBox"
                for wrapped in wrapped_circuits
            )
        )

        # verify that the contents of the verbatim box are identical
        # to the original circuit
        self.assertTrue(
            all(
                wrapped.instructions[1:-1] == original.instructions
                for wrapped, original in zip(wrapped_circuits, circuits)
            )
        )
