"""Tests for Qiskit to Braket adapter."""

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pytest
from braket.circuits import Circuit, FreeParameter, Gate, Instruction
from braket.circuits.angled_gate import AngledGate, DoubleAngledGate, TripleAngledGate
from braket.device_schema.ionq import IonqDeviceCapabilities
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import GlobalPhaseGate, PauliEvolutionGate
from qiskit.circuit.library import standard_gates as qiskit_gates
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit_ionq import ionq_gates

from qiskit_braket_provider.providers.adapter import (
    _GATE_NAME_TO_BRAKET_GATE,
    _GATE_NAME_TO_QISKIT_GATE,
    _get_controlled_gateset,
    _validate_angle_restrictions,
    convert_qiskit_to_braket_circuit,
    convert_qiskit_to_braket_circuits,
    native_angle_restrictions,
    to_braket,
    to_qiskit,
)

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


qiskit_ionq_gates = [
    ionq_gates.GPIGate(Parameter("φ")),
    ionq_gates.GPI2Gate(Parameter("φ")),
    ionq_gates.MSGate(Parameter("φ0"), Parameter("φ1"), Parameter("ϴ")),
    ionq_gates.ZZGate(Parameter("ϴ")),
]


def check_to_braket_unitary_correct(qiskit_circuit: QuantumCircuit) -> bool:
    """Checks if endianness-reversed Qiskit circuit matrix matches Braket counterpart"""
    return np.allclose(
        to_braket(qiskit_circuit).to_unitary(),
        Operator(qiskit_circuit).reverse_qargs().to_matrix(),
    )


class TestAdapter(TestCase):
    """Tests adapter."""

    def test_state_preparation_01(self):
        """Tests state_preparation handling of Adapter"""
        input_state_vector = np.array([np.sqrt(3) / 2, np.sqrt(2) * complex(1, 1) / 4])

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.prepare_state(input_state_vector, 0)

        self.assertTrue(check_to_braket_unitary_correct(qiskit_circuit))

    def test_state_preparation_00(self):
        """Tests state_preparation handling of Adapter"""
        input_state_vector = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.prepare_state(input_state_vector, 0)

        self.assertTrue(check_to_braket_unitary_correct(qiskit_circuit))

    def test_convert_parametric_qiskit_to_braket_circuit_warning(self):
        """Tests that a warning is raised when converting a parametric circuit to a Braket circuit."""
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.h(0)

        with self.assertWarns(DeprecationWarning):
            convert_qiskit_to_braket_circuit(qiskit_circuit)
            pass

        with self.assertWarns(DeprecationWarning):
            list(convert_qiskit_to_braket_circuits([qiskit_circuit]))
            pass

    def test_u_gate(self):
        """Tests adapter conversion of u gate"""
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.u(np.pi / 2, np.pi / 3, np.pi / 4, 0)

        self.assertTrue(check_to_braket_unitary_correct(qiskit_circuit))

    def test_standard_gate_decomp(self):
        """Tests adapter decomposition of all standard gates to forms that can be translated"""
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
                self.assertTrue(check_to_braket_unitary_correct(qiskit_circuit))

    def test_ionq_gates(self):
        """Tests adapter decomposition of all standard gates to forms that can be translated"""
        for gate in qiskit_ionq_gates:
            qiskit_circuit = QuantumCircuit(gate.num_qubits)
            qiskit_circuit.append(gate, range(gate.num_qubits))

            parameters = gate.params
            parameter_values = [
                (137 / 61) * np.pi / i for i in range(1, len(parameters) + 1)
            ]
            parameter_bindings = dict(zip(parameters, parameter_values))
            qiskit_circuit = qiskit_circuit.assign_parameters(parameter_bindings)

            with self.subTest(f"Circuit with {gate.name} gate."):
                self.assertTrue(check_to_braket_unitary_correct(qiskit_circuit))

    def test_global_phase(self):
        """Tests conversion when transpiler generates a global phase"""
        qiskit_circuit = QuantumCircuit(1, global_phase=np.pi / 2)
        qiskit_circuit.h(0)
        gate = GlobalPhaseGate(1.23)
        qiskit_circuit.append(gate, [])

        braket_circuit = to_braket(qiskit_circuit)
        expected_braket_circuit = Circuit().h(0).gphase(1.23).gphase(np.pi / 2)
        self.assertEqual(
            braket_circuit.global_phase, qiskit_circuit.global_phase + gate.params[0]
        )
        self.assertEqual(braket_circuit, expected_braket_circuit)

        braket_circuit_no_gphase = to_braket(qiskit_circuit, basis_gates={"h"})
        self.assertEqual(braket_circuit_no_gphase.global_phase, 0)
        self.assertEqual(braket_circuit_no_gphase, Circuit().h(0))

    def test_exponential_gate_decomp(self):
        """Tests adapter translation of exponential gates"""
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

        self.assertTrue(check_to_braket_unitary_correct(qiskit_circuit))

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
            "r": "prx",
            "rxx": "xx",
            "ryy": "yy",
            "zz": "zz",
            "global_phase": "gphase",
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
                "gpi",
                "gpi2",
                "ms",
            ]
        }

        braket_to_qiskit_gate_names = {
            **qiskit_to_braket_gate_names,
            **{"measure": "measure"},
        }

        self.assertEqual(
            set(qiskit_to_braket_gate_names.keys()),
            set(_GATE_NAME_TO_BRAKET_GATE.keys()),
        )

        self.assertEqual(
            set(braket_to_qiskit_gate_names.values()),
            set(_GATE_NAME_TO_QISKIT_GATE.keys()),
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
            .u(0, FreeParameter("θ"), FreeParameter("φ"), FreeParameter("λ"))
            .u(0, FreeParameter("θ"), FreeParameter("φ"), np.pi)
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

    def test_measure(self):
        """Tests the translation of a measure instruction"""

        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.measure(0, 0)
        braket_circuit = to_braket(qiskit_circuit)

        expected_braket_circuit = (
            Circuit().h(0).cnot(0, 1).measure(0)  # pylint: disable=no-member
        )

        self.assertEqual(braket_circuit, expected_braket_circuit)

    def test_measure_order_preserved(self):
        """Tests the translation of measure instructions on multiple qubits"""

        qiskit_circuit = QuantumCircuit(3, 3)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.cx(1, 2)
        qiskit_circuit.measure(0, 1)  # measure qubit 0 into classical bit 1
        qiskit_circuit.measure(1, 2)  # measure qubit 1 into classical bit 2
        qiskit_circuit.measure(2, 0)  # measure qubit 2 into classical bit 0
        braket_circuit = to_braket(qiskit_circuit)

        expected_braket_circuit = (
            Circuit()
            .h(0)
            .cnot(0, 1)
            .cnot(1, 2)
            .measure(2)
            .measure(0)
            .measure(1)  # pylint: disable=no-member
        )

        self.assertEqual(braket_circuit, expected_braket_circuit)

    def test_measure_repeated(self):
        """Tests that repeated measurement on a qubit raises a ValueError."""
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.measure(0, 0)
        qiskit_circuit.measure([0, 1], [0, 1])

        with self.assertRaises(ValueError):
            to_braket(qiskit_circuit)

    def test_gate_after_measure(self):
        """Tests that adding a gate to a measured qubit raises a ValueError."""
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.measure(0, 0)
        qiskit_circuit.h(0)

        with self.assertRaises(ValueError):
            to_braket(qiskit_circuit)

    def test_reset(self):
        """Tests if NotImplementedError is raised for reset operation."""

        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.reset(0)

        with self.assertRaises(NotImplementedError):
            to_braket(qiskit_circuit)

    def test_measure_different_indices(self):
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
            Circuit().h(0).cnot(0, 1).measure(0)  # pylint: disable=no-member
        )

        self.assertEqual(braket_circuit, expected_braket_circuit)

    def test_measure_subset_indices(self):
        """
        Tests the translation of a measure instruction on
        a subset of qubits.
        """

        qiskit_circuit = QuantumCircuit(4, 2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.cx(1, 2)
        qiskit_circuit.cx(2, 3)
        qiskit_circuit.measure(0, 0)
        qiskit_circuit.measure(2, 1)
        braket_circuit = to_braket(qiskit_circuit)

        expected_braket_circuit = (
            Circuit()  # pylint: disable=no-member
            .h(0)
            .cnot(0, 1)
            .cnot(1, 2)
            .cnot(2, 3)
            .measure(0)
            .measure(2)
        )

        self.assertEqual(braket_circuit, expected_braket_circuit)

    def test_measure_all(self):
        """
        Tests the translation of a measure_all instruction
        """

        qiskit_circuit = QuantumCircuit(4, 2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.cx(1, 2)
        qiskit_circuit.cx(2, 3)
        qiskit_circuit.measure_all()
        braket_circuit = to_braket(qiskit_circuit)

        expected_braket_circuit = (
            Circuit()  # pylint: disable=no-member
            .h(0)
            .cnot(0, 1)
            .cnot(1, 2)
            .cnot(2, 3)
            .measure(0)
            .measure(1)
            .measure(2)
            .measure(3)
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
            .measure(2)
            .measure(0)
        )
        self.assertEqual(braket_circuit, expected_braket_circuit)

    def test_verbatim(self):
        """Tests that transpilation is skipped for verbatim circuits."""
        qiskit_circuit = QuantumCircuit(2, 1)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.measure(1, 0)

        assert to_braket(qiskit_circuit, {"x"}, True) == Circuit().add_verbatim_box(
            Circuit().h(0).cnot(0, 1)
        ).measure(1)

    def test_parameter_vector(self):
        """Tests ParameterExpression translation."""
        qiskit_circuit = QuantumCircuit(1)
        v = ParameterVector("v", 2)
        qiskit_circuit.rx(v[0], 0)
        qiskit_circuit.ry(v[1], 0)
        braket_circuit = to_braket(qiskit_circuit)

        expected_braket_circuit = (
            Circuit().rx(0, FreeParameter("v_0")).ry(0, FreeParameter("v_1"))
        )
        assert braket_circuit == expected_braket_circuit

    def test_parameter_expression(self):
        """Tests ParameterExpression translation."""
        qiskit_circuit = QuantumCircuit(1)
        v = ParameterVector("v", 2)
        qiskit_circuit.rx(Parameter("angle_1") + 2 * Parameter("angle_2"), 0)
        qiskit_circuit.ry(v[0] - 2 * v[1], 0)
        braket_circuit = to_braket(qiskit_circuit)

        expected_braket_circuit = (
            Circuit()
            .rx(0, FreeParameter("angle_1") + 2 * FreeParameter("angle_2"))
            .ry(0, FreeParameter("v_0") - 2 * FreeParameter("v_1"))
        )
        assert braket_circuit == expected_braket_circuit

    def test_name_conflict_with_parameter_vector(self):
        """Tests ParameterExpression translation."""
        qiskit_circuit = QuantumCircuit(1)
        v = ParameterVector("v", 1)
        v0 = Parameter("v_0")
        qiskit_circuit.rx(v0, 0)
        qiskit_circuit.ry(v[0] + 1, 0)

        with pytest.raises(ValueError, match="Please rename your parameters."):
            to_braket(qiskit_circuit)

    @patch("qiskit_braket_provider.providers.adapter.transpile")
    def test_invalid_ctrl_state(self, mock_transpile):
        """Tests that control states other than all 1s are rejected."""
        qiskit_circuit = QuantumCircuit(2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1, ctrl_state=0)

        mock_transpile.return_value = qiskit_circuit
        with pytest.raises(ValueError):
            to_braket(qiskit_circuit)

    def test_get_controlled_gateset(self):
        """Tests that the correct controlled gateset is returned for all maximum qubit counts."""
        full_gateset = {"h", "s", "sdg", "sx", "rx", "ry", "rz", "cx", "cz"}
        restricted_gateset = {"rx", "cx", "sx"}
        max1 = {"ch", "cs", "csdg", "csx", "crx", "cry", "crz", "ccz"}
        max3 = max1.union({"c3sx"})
        unlimited = max3.union({"mcx"})
        assert _get_controlled_gateset(full_gateset, 0) == set()
        assert _get_controlled_gateset(full_gateset, 1) == max1
        assert _get_controlled_gateset(full_gateset, 2) == max1
        assert _get_controlled_gateset(full_gateset, 3) == max3
        assert _get_controlled_gateset(full_gateset, 4) == max3
        assert _get_controlled_gateset(full_gateset) == unlimited
        assert _get_controlled_gateset(restricted_gateset, 3) == {"crx", "csx", "c3sx"}
        assert _get_controlled_gateset(restricted_gateset) == {
            "crx",
            "csx",
            "c3sx",
            "mcx",
        }

    def test_connectivity(self):
        """Tests transpiling with connectivity"""
        qiskit_circuit = QuantumCircuit(3)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.rxx(0.1, 0, 2)
        connectivity = [[0, 1], [1, 0], [1, 2], [2, 1]]

        braket_circuit = to_braket(qiskit_circuit, connectivity=connectivity)
        braket_circuit_unconnected = to_braket(qiskit_circuit)
        braket_circuit_verbatim = to_braket(
            qiskit_circuit, verbatim=True, connectivity=connectivity
        )

        def gate_matches_connectivity(gate) -> bool:
            return any(
                (
                    gate.target.union(gate.control).issubset(adjacency)
                    for adjacency in connectivity
                )
            )

        assert all(
            (gate_matches_connectivity(gate) for gate in braket_circuit.instructions)
        )
        assert not all(
            (
                gate_matches_connectivity(gate)
                for gate in braket_circuit_unconnected.instructions
            )
        )
        assert not all(
            (
                gate_matches_connectivity(gate)
                for gate in braket_circuit_verbatim.instructions
            )
        )

    def test_angle_restrictions_rigetti(self):
        """Tests that angle restrictions for native gates are enforced."""
        circuit = QuantumCircuit(1)
        circuit.rx(np.pi / 4, 0)

        restrictions = {"rx": {0: {np.pi, -np.pi, np.pi / 2, -np.pi / 2}}}
        with pytest.raises(ValueError):
            to_braket(circuit, basis_gates={"rx"}, angle_restrictions=restrictions)

    def test_angle_restrictions_rigetti_valid(self):
        """Tests that allowed Rigetti angles pass validation."""
        circuit = QuantumCircuit(1)
        circuit.rx(np.pi / 2, 0)

        restrictions = {"rx": {0: {np.pi, -np.pi, np.pi / 2, -np.pi / 2}}}
        braket_circuit = to_braket(
            circuit, basis_gates={"rx"}, angle_restrictions=restrictions
        )
        assert len(braket_circuit.instructions) == 1

    def test_validate_angle_restrictions_extra_index(self):
        """Restrictions on unused parameter indices should be ignored."""
        _validate_angle_restrictions("rx", [0.0], {"rx": {1: {0.0}}})

    def test_native_angle_restrictions_default(self):
        """Unknown device types should return an empty restriction map."""
        properties = GateModelSimulatorDeviceCapabilities.parse_obj(
            {
                "braketSchemaHeader": {
                    "name": "braket.device_schema.simulators.gate_model_simulator_device_capabilities",
                    "version": "1",
                },
                "service": {"executionWindows": [], "shotsRange": [1, 10]},
                "action": {
                    "braket.ir.jaqcd.program": {
                        "actionType": "braket.ir.jaqcd.program",
                        "version": ["1"],
                        "supportedOperations": ["x"],
                    }
                },
                "paradigm": {"qubitCount": 2},
                "deviceParameters": {},
            }
        )
        assert not native_angle_restrictions(properties)

    def test_angle_restrictions_ionq(self):
        """Tests IonQ MS gate angle range enforcement."""
        circuit = QuantumCircuit(2)
        circuit.append(ionq_gates.MSGate(0, 0, 3), [0, 1])

        restrictions = {"ms": {2: (0.0, 0.25)}}
        with pytest.raises(ValueError):
            to_braket(circuit, basis_gates={"ms"}, angle_restrictions=restrictions)

    def test_angle_restrictions_ionq_valid(self):
        """Tests that allowed IonQ MS angles pass validation."""
        circuit = QuantumCircuit(2)
        circuit.append(ionq_gates.MSGate(0, 0, 0.25), [0, 1])

        restrictions = {"ms": {2: (0.0, 0.25)}}
        braket_circuit = to_braket(
            circuit, basis_gates={"ms"}, angle_restrictions=restrictions
        )
        assert len(braket_circuit.instructions) == 1

    def test_native_angle_restrictions_ionq(self):
        """Tests that IonQ capabilities return the correct angle restriction map."""
        properties = IonqDeviceCapabilities.parse_obj(
            {
                "braketSchemaHeader": {
                    "name": "braket.device_schema.ionq.ionq_device_capabilities",
                    "version": "1",
                },
                "service": {
                    "braketSchemaHeader": {
                        "name": "braket.device_schema.device_service_properties",
                        "version": "1",
                    },
                    "executionWindows": [],
                    "shotsRange": [1, 10],
                    "deviceCost": {"price": 0.25, "unit": "minute"},
                    "deviceDocumentation": {
                        "imageUrl": "",
                        "summary": "",
                        "externalDocumentationUrl": "",
                    },
                    "deviceLocation": "us-east-1",
                    "updatedAt": "2020-06-16T00:00:00",
                },
                "action": {
                    "braket.ir.jaqcd.program": {
                        "actionType": "braket.ir.jaqcd.program",
                        "version": ["1"],
                        "supportedOperations": ["x"],
                    }
                },
                "paradigm": {
                    "braketSchemaHeader": {
                        "name": "braket.device_schema.gate_model_qpu_paradigm_properties",
                        "version": "1",
                    },
                    "qubitCount": 2,
                    "nativeGateSet": [],
                    "connectivity": {
                        "fullyConnected": False,
                        "connectivityGraph": {"0": ["1"], "1": ["0"]},
                    },
                },
                "deviceParameters": {},
            }
        )

        assert native_angle_restrictions(properties) == {"ms": {2: (0.0, 0.25)}}

    def test_angle_restrictions_skip_parameter_expression(self):
        """Parameters that are expressions should bypass angle validation."""
        theta = Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.rx(theta, 0)

        restrictions = {"rx": {0: {0.0}}}
        braket_circuit = to_braket(
            circuit, basis_gates={"rx"}, angle_restrictions=restrictions
        )
        assert len(braket_circuit.instructions) == 1


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
        Tests Braket to Qiskit conversion with standard gates.
        """

        gate_set = [
            attr
            for attr in dir(Gate)
            if attr[0].isupper() and attr.lower() in _GATE_NAME_TO_QISKIT_GATE
        ]

        # pytest.mark.parametrize is incompatible with TestCase
        param_sets = [
            [0.1, 0.2, 0.3],
            [
                FreeParameter("angle_1"),
                FreeParameter("angle_2"),
                FreeParameter("angle_3"),
            ],
        ]

        for gate_name in gate_set:
            for params_braket in param_sets:
                gate = getattr(Gate, gate_name)
                if issubclass(gate, AngledGate):
                    op = gate(params_braket[0])
                elif issubclass(gate, DoubleAngledGate):
                    op = gate(params_braket[0], params_braket[1])
                elif issubclass(gate, TripleAngledGate):
                    op = gate(*params_braket)
                else:
                    op = gate()
                target = range(op.qubit_count)
                instr = Instruction(op, target)

                braket_circuit = Circuit().add_instruction(instr)
                qiskit_circuit = to_qiskit(braket_circuit)
                param_uuids = {
                    param.name: param._uuid for param in qiskit_circuit.parameters
                }
                params_qiskit = [
                    (
                        Parameter(param.name, uuid=param_uuids.get(param.name))
                        if isinstance(param, FreeParameter)
                        else param
                    )
                    for param in params_braket
                ]

                expected_qiskit_circuit = QuantumCircuit(op.qubit_count)
                qiskit_gate = _GATE_NAME_TO_QISKIT_GATE.get(gate_name.lower())
                expected_qiskit_circuit.append(qiskit_gate, target)
                expected_qiskit_circuit.measure_all()
                expected_qiskit_circuit = expected_qiskit_circuit.assign_parameters(
                    dict(
                        zip(
                            # Need to use qiskit_gate.parameters because
                            # qiskit_circuit.parameters is sorted alphabetically
                            [list(expr.parameters)[0] for expr in qiskit_gate.params],
                            params_qiskit[: len(qiskit_gate.params)],
                        )
                    )
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

    def test_unsupported_braket_gate(self):
        """Tests if TypeError is raised for unsupported Braket gate."""

        gate = getattr(Gate, "CNot")
        op = gate()
        instr = Instruction(op, range(2))
        circuit = Circuit().add_instruction(instr)

        with self.assertRaises(TypeError):
            with patch.dict(
                "qiskit_braket_provider.providers.adapter._GATE_NAME_TO_QISKIT_GATE",
                {"cnot": None},
            ):
                to_qiskit(circuit)

    def test_measure_subset(self):
        """Tests the measure instruction conversion from braket to qiskit"""
        braket_circuit = Circuit().h(0).cnot(0, 1).measure(0)
        qiskit_circuit = to_qiskit(braket_circuit)

        expected_qiskit_circuit = QuantumCircuit(2, 1)
        expected_qiskit_circuit.h(0)
        expected_qiskit_circuit.cx(0, 1)
        expected_qiskit_circuit.measure(0, 0)

        self.assertEqual(qiskit_circuit, expected_qiskit_circuit)

    def test_measure_multiple_indices(self):
        """
        Tests the measure instruction conversion with multiple
        indices in the braket measure target.
        """
        braket_circuit = Circuit().h(0).cnot(0, 1).cnot(1, 2).measure([0, 1, 2])
        qiskit_circuit = to_qiskit(braket_circuit)

        expected_qiskit_circuit = QuantumCircuit(3, 3)
        expected_qiskit_circuit.h(0)
        expected_qiskit_circuit.cx(0, 1)
        expected_qiskit_circuit.cx(1, 2)
        expected_qiskit_circuit.measure(0, 0)
        expected_qiskit_circuit.measure(1, 1)
        expected_qiskit_circuit.measure(2, 2)

        self.assertEqual(qiskit_circuit, expected_qiskit_circuit)

    def test_measure_different_indices(self):
        """
        Tests the measure instruction conversion from with
        the ordering of the targets unsorted.
        """
        braket_circuit = Circuit().h(0).cnot(0, 1).measure([1, 0])
        qiskit_circuit = to_qiskit(braket_circuit)

        expected_qiskit_circuit = QuantumCircuit(2, 2)
        expected_qiskit_circuit.h(0)
        expected_qiskit_circuit.cx(0, 1)
        expected_qiskit_circuit.measure(1, 0)
        expected_qiskit_circuit.measure(0, 1)

        self.assertEqual(qiskit_circuit, expected_qiskit_circuit)
