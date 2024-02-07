"""Tests for Qiskit to Braket adapter."""
from unittest import TestCase
from unittest.mock import Mock

from braket.circuits import Circuit, FreeParameter, Gate, Instruction, observables
from braket.circuits.angled_gate import AngledGate, TripleAngledGate
from braket.devices import LocalSimulator

import numpy as np
import pytest

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile,
)
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers.basicaer import BasicAer
from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit.library import standard_gates as qiskit_gates

from qiskit_braket_provider.providers.adapter import (
    to_qiskit,
    to_braket,
    convert_qiskit_to_braket_circuit,
    convert_qiskit_to_braket_circuits,
    GATE_NAME_TO_BRAKET_GATE,
    GATE_NAME_TO_QISKIT_GATE,
    wrap_circuits_in_verbatim_box,
    get_controlled_gateset,
)
from qiskit_braket_provider.providers.braket_backend import BraketLocalBackend

from tests.providers.test_braket_backend import combine_dicts

_EPS = 1e-10  # global variable used to chop very small numbers to zero

standard_gates = [
    qiskit_gates.IGate(),
    qiskit_gates.SXGate(),
    qiskit_gates.XGate(),
    qiskit_gates.CXGate(),
    qiskit_gates.RZGate(Parameter("λ")),
    qiskit_gates.RGate(Parameter("ϴ"), Parameter("φ")),
    qiskit_gates.C3SXGate(),
    qiskit_gates.CCXGate(),
    qiskit_gates.DCXGate(),
    qiskit_gates.CHGate(),
    qiskit_gates.CPhaseGate(Parameter("ϴ")),
    qiskit_gates.CRXGate(Parameter("ϴ")),
    qiskit_gates.CRYGate(Parameter("ϴ")),
    qiskit_gates.CRZGate(Parameter("ϴ")),
    qiskit_gates.CSwapGate(),
    qiskit_gates.CSXGate(),
    qiskit_gates.CUGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ"), Parameter("γ")),
    qiskit_gates.CU1Gate(Parameter("λ")),
    qiskit_gates.CU3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
    qiskit_gates.CYGate(),
    qiskit_gates.CZGate(),
    qiskit_gates.CCZGate(),
    qiskit_gates.HGate(),
    qiskit_gates.PhaseGate(Parameter("ϴ")),
    qiskit_gates.RCCXGate(),
    qiskit_gates.RC3XGate(),
    qiskit_gates.RXGate(Parameter("ϴ")),
    qiskit_gates.RXXGate(Parameter("ϴ")),
    qiskit_gates.RYGate(Parameter("ϴ")),
    qiskit_gates.RYYGate(Parameter("ϴ")),
    qiskit_gates.RZZGate(Parameter("ϴ")),
    qiskit_gates.RZXGate(Parameter("ϴ")),
    qiskit_gates.XXMinusYYGate(Parameter("ϴ"), Parameter("φ")),
    qiskit_gates.XXPlusYYGate(Parameter("ϴ"), Parameter("φ")),
    qiskit_gates.ECRGate(),
    qiskit_gates.SGate(),
    qiskit_gates.SdgGate(),
    qiskit_gates.CSGate(),
    qiskit_gates.CSdgGate(),
    qiskit_gates.SwapGate(),
    qiskit_gates.iSwapGate(),
    qiskit_gates.SXdgGate(),
    qiskit_gates.TGate(),
    qiskit_gates.TdgGate(),
    qiskit_gates.UGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
    qiskit_gates.U1Gate(Parameter("λ")),
    qiskit_gates.U2Gate(Parameter("φ"), Parameter("λ")),
    qiskit_gates.U3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
    qiskit_gates.YGate(),
    qiskit_gates.ZGate(),
]


class TestAdapter(TestCase):
    """Tests adapter."""

    def test_state_preparation_01(self):
        """Tests state_preparation handling of Adapter"""
        input_state_vector = np.array([np.sqrt(3) / 2, np.sqrt(2) * complex(1, 1) / 4])

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.prepare_state(input_state_vector, 0)

        braket_circuit = to_braket(qiskit_circuit)
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

        braket_circuit = to_braket(qiskit_circuit)
        braket_circuit.state_vector()  # pylint: disable=no-member
        result = LocalSimulator().run(braket_circuit)
        output_state_vector = np.array(result.result().values[0])

        self.assertTrue(
            (np.linalg.norm(input_state_vector - output_state_vector)) < _EPS
        )

    def test_convert_parametric_qiskit_to_braket_circuit_warning(self):
        """Tests that a warning is raised when converting a parametric circuit to a Braket circuit."""
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.h(0)

        with self.assertWarns(DeprecationWarning):
            convert_qiskit_to_braket_circuit(qiskit_circuit)

        with self.assertWarns(DeprecationWarning):
            list(convert_qiskit_to_braket_circuits([qiskit_circuit]))

    def test_u_gate(self):
        """Tests adapter conversion of u gate"""
        qiskit_circuit = QuantumCircuit(1)
        backend = BasicAer.get_backend("statevector_simulator")
        device = LocalSimulator()
        qiskit_circuit.u(np.pi / 2, np.pi / 3, np.pi / 4, 0)

        job = backend.run(qiskit_circuit)

        braket_circuit = to_braket(qiskit_circuit)
        braket_circuit.state_vector()  # pylint: disable=no-member

        braket_output = device.run(braket_circuit).result().values[0]
        qiskit_output = np.array(job.result().get_statevector(qiskit_circuit))

        self.assertTrue(np.linalg.norm(braket_output - qiskit_output) < _EPS)

    def test_standard_gate_decomp(self):
        """Tests adapter decomposition of all standard gates to forms that can be translated"""
        aer_backend = BasicAer.get_backend("statevector_simulator")
        backend = BraketLocalBackend()

        for standard_gate in standard_gates:
            qiskit_circuit = QuantumCircuit(standard_gate.num_qubits)
            qiskit_circuit.append(standard_gate, range(standard_gate.num_qubits))

            parameters = standard_gate.params
            if parameters:
                parameter_values = [
                    (137 / 61) * np.pi / i for i in range(1, len(parameters) + 1)
                ]
                parameter_bindings = dict(zip(parameters, parameter_values))
                qiskit_circuit = qiskit_circuit.assign_parameters(parameter_bindings)

            with self.subTest(f"Circuit with {standard_gate.name} gate."):
                braket_job = backend.run(qiskit_circuit, shots=1000)
                braket_result = braket_job.result().get_counts()

                transpiled_circuit = transpile(qiskit_circuit, backend=aer_backend)
                qiskit_job = aer_backend.run(transpiled_circuit, shots=1000)
                qiskit_result = qiskit_job.result().get_counts()

                combined_results = combine_dicts(
                    {k: float(v) / 1000.0 for k, v in braket_result.items()},
                    qiskit_result,
                )

                for key, values in combined_results.items():
                    percent_diff = abs(
                        ((float(values[0]) - values[1]) / values[0]) * 100
                    )
                    abs_diff = abs(values[0] - values[1])
                    self.assertTrue(
                        percent_diff < 10 or abs_diff < 0.05,
                        f"Key {key} with percent difference {percent_diff} "
                        f"and absolute difference {abs_diff}. Original values {values}",
                    )

    def test_global_phase(self):
        """Tests conversion when transpiler generates a global phase"""
        qiskit_circuit = QuantumCircuit(1, global_phase=np.pi / 2)
        qiskit_circuit.h(0)

        braket_circuit = to_braket(qiskit_circuit)

        expected_braket_circuit = Circuit().h(0).gphase(np.pi / 2)

        self.assertEqual(braket_circuit.global_phase, qiskit_circuit.global_phase)
        self.assertEqual(braket_circuit, expected_braket_circuit)

    def test_exponential_gate_decomp(self):
        """Tests adapter translation of exponential gates"""
        aer_backend = BasicAer.get_backend("statevector_simulator")
        backend = BraketLocalBackend()
        qiskit_circuit = QuantumCircuit(2)

        operator = SparsePauliOp(
            ["ZZ", "XI"],
            coeffs=[
                1,
                -0.1,
            ],
        )
        evo = PauliEvolutionGate(operator, time=2)

        qiskit_circuit.append(evo, range(2))

        braket_job = backend.run(qiskit_circuit, shots=1000)
        braket_result = braket_job.result().get_counts()

        transpiled_circuit = transpile(qiskit_circuit, backend=aer_backend)
        qiskit_job = aer_backend.run(transpiled_circuit, shots=1000)
        qiskit_result = qiskit_job.result().get_counts()

        combined_results = combine_dicts(
            {k: float(v) / 1000.0 for k, v in braket_result.items()}, qiskit_result
        )

        for key, values in combined_results.items():
            percent_diff = abs(((float(values[0]) - values[1]) / values[0]) * 100)
            abs_diff = abs(values[0] - values[1])
            self.assertTrue(
                percent_diff < 10 or abs_diff < 0.05,
                f"Key {key} with percent difference {percent_diff} "
                f"and absolute difference {abs_diff}. Original values {values}",
            )

    def test_mappers(self):
        """Tests mappers."""
        qiskit_to_braket_gate_names = {
            "p": "phaseshift",
            "cx": "cnot",
            "tdg": "ti",
            "sdg": "si",
            "sx": "v",
            "sxdg": "vi",
            "rzz": "zz",
            "id": "i",
            "ccx": "ccnot",
            "cp": "cphaseshift",
            "rxx": "xx",
            "ryy": "yy",
        }

        qiskit_to_braket_gate_names |= {
            g: g
            for g in [
                "u",
                "u1",
                "u2",
                "u3",
                "x",
                "y",
                "z",
                "t",
                "s",
                "swap",
                "iswap",
                "rx",
                "ry",
                "rz",
                "h",
                "cy",
                "cz",
                "cswap",
                "ecr",
            ]
        }

        self.assertEqual(
            list(sorted(qiskit_to_braket_gate_names.keys())),
            list(sorted(GATE_NAME_TO_BRAKET_GATE.keys())),
        )

        self.assertEqual(
            list(sorted(qiskit_to_braket_gate_names.values())),
            list(sorted(GATE_NAME_TO_QISKIT_GATE.keys())),
        )

    def test_type_error_on_bad_input(self):
        """Test raising TypeError if adapter does not receive a Qiskit QuantumCircuit."""
        circuit = Mock()

        message = f"Expected a QuantumCircuit, got {type(circuit)} instead."
        with pytest.raises(TypeError, match=message):
            to_braket(circuit)

    def test_convert_parametric_qiskit_to_braket_circuit(self):
        """Tests to_braket works with parametric circuits."""

        theta = Parameter("θ")
        phi = Parameter("φ")
        lam = Parameter("λ")
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.rz(theta, 0)
        qiskit_circuit.u(theta, phi, lam, 0)
        qiskit_circuit.u(theta, phi, np.pi, 0)
        braket_circuit = to_braket(qiskit_circuit)

        expected_braket_circuit = (
            Circuit()  # pylint: disable=no-member
            .rz(0, FreeParameter("θ"))
            .phaseshift(0, FreeParameter("λ"))
            .ry(0, FreeParameter("θ"))
            .phaseshift(0, FreeParameter("φ"))
            .phaseshift(0, np.pi)
            .ry(0, FreeParameter("θ"))
            .phaseshift(0, FreeParameter("φ"))
        )

        self.assertEqual(braket_circuit, expected_braket_circuit)

    def test_barrier(self):
        """Tests conversion with barrier."""
        qiskit_circuit = QuantumCircuit(2)
        qiskit_circuit.x(0)
        qiskit_circuit.barrier()
        qiskit_circuit.x(1)

        with pytest.warns(UserWarning, match="contains barrier instructions"):
            braket_circuit = to_braket(qiskit_circuit)

        expected_braket_circuit = Circuit().x(0).x(1)

        self.assertEqual(braket_circuit, expected_braket_circuit)

    def test_sample_result_type(self):
        """Tests sample result type with observables Z"""

        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.measure(0, 0)
        braket_circuit = to_braket(qiskit_circuit)

        expected_braket_circuit = (
            Circuit()  # pylint: disable=no-member
            .h(0)
            .cnot(0, 1)
            .sample(observable=observables.Z(), target=0)
        )

        self.assertEqual(braket_circuit, expected_braket_circuit)

    def test_sample_result_type_different_indices(self):
        """
        Tests the translation of a measure instruction.

        We test that the issue #132 has been fixed. The qubit index
        can be different from the classical bit index. The classical bit
        is ignored during the translation.
        """

        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.measure(0, 1)
        braket_circuit = to_braket(qiskit_circuit)

        expected_braket_circuit = (
            Circuit()  # pylint: disable=no-member
            .h(0)
            .cnot(0, 1)
            .sample(observable=observables.Z(), target=0)
        )

        self.assertEqual(braket_circuit, expected_braket_circuit)

    def test_multiple_registers(self):
        """
        Tests the use of multiple registers.

        Confirming that #51 has been fixed.
        """
        qreg_a = QuantumRegister(2, "qreg_a")
        qreg_b = QuantumRegister(1, "qreg_b")
        creg = ClassicalRegister(2, "creg")
        qiskit_circuit = QuantumCircuit(qreg_a, qreg_b, creg)
        qiskit_circuit.h(qreg_a[0])
        qiskit_circuit.cx(qreg_a[0], qreg_b[0])
        qiskit_circuit.x(qreg_a[1])
        qiskit_circuit.measure(qreg_a[0], creg[1])
        qiskit_circuit.measure(qreg_b[0], creg[0])
        braket_circuit = to_braket(qiskit_circuit)

        expected_braket_circuit = (
            Circuit()  # pylint: disable=no-member
            .h(0)
            .cnot(0, 2)
            .x(1)
            .sample(observable=observables.Z(), target=0)
            .sample(observable=observables.Z(), target=2)
        )
        self.assertEqual(braket_circuit, expected_braket_circuit)

    def test_get_controlled_gateset(self):
        """Tests that the correct controlled gateset is returned for all maximum qubit counts."""
        max1 = {"ch", "cs", "csdg", "csx", "crx", "cry", "crz", "ccz"}
        max3 = max1.union({"c3sx"})
        unlimited = max3.union({"mcx"})
        assert get_controlled_gateset(0) == set()
        assert get_controlled_gateset(1) == max1
        assert get_controlled_gateset(2) == max1
        assert get_controlled_gateset(3) == max3
        assert get_controlled_gateset(4) == max3
        assert get_controlled_gateset() == unlimited


class TestFromBraket(TestCase):
    """Test Braket circuit conversion."""

    def test_type_error_on_bad_input(self):
        """Test raising TypeError if adapter does not receive a Braket Circuit."""
        circuit = Mock()

        message = f"Expected a Circuit, got {type(circuit)} instead."
        with pytest.raises(TypeError, match=message):
            to_qiskit(circuit)

    def test_all_standard_gates(self):
        """
        Tests braket to qiskit conversion with standard gates.
        """

        gate_set = [attr for attr in dir(Gate) if attr[0].isupper()]

        for gate_name in gate_set:
            if gate_name.lower() not in GATE_NAME_TO_QISKIT_GATE:
                continue

            gate = getattr(Gate, gate_name)
            if issubclass(gate, AngledGate):
                op = gate(0)
            elif issubclass(gate, TripleAngledGate):
                op = gate(0, 0, 0)
            else:
                op = gate()
            target = range(op.qubit_count)
            instr = Instruction(op, target)

            braket_circuit = Circuit().add_instruction(instr)
            qiskit_circuit = to_qiskit(braket_circuit)

            expected_qiskit_circuit = QuantumCircuit(op.qubit_count)
            expected_qiskit_circuit.append(
                GATE_NAME_TO_QISKIT_GATE.get(gate_name.lower()), target
            )
            expected_qiskit_circuit.measure_all()
            expected_qiskit_circuit = expected_qiskit_circuit.assign_parameters(
                {p: 0 for p in expected_qiskit_circuit.parameters}
            )

            self.assertEqual(qiskit_circuit, expected_qiskit_circuit)

    def test_parametric_gates(self):
        """
        Tests braket to qiskit conversion with standard gates.
        """
        braket_circuit = Circuit().rx(0, FreeParameter("alpha"))
        qiskit_circuit = to_qiskit(braket_circuit)

        uuid = qiskit_circuit.parameters[0]._uuid

        expected_qiskit_circuit = QuantumCircuit(1)
        expected_qiskit_circuit.rx(Parameter("alpha", uuid=uuid), 0)

        expected_qiskit_circuit.measure_all()
        self.assertEqual(qiskit_circuit, expected_qiskit_circuit)

    def test_control_modifier(self):
        """
        Tests braket to qiskit conversion with controlled gates.
        """
        braket_circuit = Circuit().x(1, control=[0])
        qiskit_circuit = to_qiskit(braket_circuit)

        expected_qiskit_circuit = QuantumCircuit(2)
        cx = qiskit_gates.XGate().control(1)
        expected_qiskit_circuit.append(cx, [0, 1])

        expected_qiskit_circuit.measure_all()
        self.assertEqual(qiskit_circuit, expected_qiskit_circuit)

    def test_unused_middle_qubit(self):
        """
        Tests braket to qiskit conversion with non-continuous qubit registers.
        """
        braket_circuit = Circuit().x(3, control=[0, 2], control_state="10")
        qiskit_circuit = to_qiskit(braket_circuit)

        expected_qiskit_circuit = QuantumCircuit(3)
        cx = qiskit_gates.XGate().control(2, ctrl_state="01")
        expected_qiskit_circuit.append(cx, [0, 1, 2])
        expected_qiskit_circuit.measure_all()

        self.assertEqual(qiskit_circuit, expected_qiskit_circuit)

    def test_control_modifier_with_control_state(self):
        """
        Tests braket to qiskit conversion with controlled gates and control state.
        """
        braket_circuit = Circuit().x(3, control=[0, 1, 2], control_state="100")
        qiskit_circuit = to_qiskit(braket_circuit)

        expected_qiskit_circuit = QuantumCircuit(4)
        cx = qiskit_gates.XGate().control(3, ctrl_state="001")
        expected_qiskit_circuit.append(cx, [0, 1, 2, 3])
        expected_qiskit_circuit.measure_all()

        self.assertEqual(qiskit_circuit, expected_qiskit_circuit)

    def test_power(self):
        """
        Tests braket to qiskit conversion with gate exponentiation.
        """
        braket_circuit = Circuit().x(0, power=0.5)
        qiskit_circuit = to_qiskit(braket_circuit)

        expected_qiskit_circuit = QuantumCircuit(1)
        sx = qiskit_gates.XGate().power(0.5)
        expected_qiskit_circuit.append(sx, [0])
        expected_qiskit_circuit.measure_all()

        self.assertEqual(qiskit_circuit, expected_qiskit_circuit)


class TestVerbatimBoxWrapper(TestCase):
    """Test wrapping in Verbatim box."""

    def test_wrapped_circuits_have_one_instruction_equivalent_to_original_one(self):
        """Test circuits wrapped in verbatim box have correct instructions."""
        circuits = [
            Circuit().rz(1, 0.1).cz(0, 1).rx(0, 0.1),
            Circuit().cz(0, 1).cz(1, 2),
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
