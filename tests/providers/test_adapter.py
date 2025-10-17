"""Tests for Qiskit to Braket adapter."""

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, generate_preset_pass_manager
from qiskit.circuit import Instruction as QiskitInstruction
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import GlobalPhaseGate, PauliEvolutionGate
from qiskit.circuit.library import standard_gates as qiskit_gates
from qiskit.quantum_info import Kraus, Operator, SparsePauliOp
from qiskit.transpiler import Target
from qiskit_ionq import ionq_gates

import braket.circuits.noises as braket_noises
from braket.circuits import Circuit, FreeParameter, Gate, Instruction
from braket.circuits.angled_gate import AngledGate, DoubleAngledGate, TripleAngledGate
from braket.device_schema.ionq import IonqDeviceCapabilities
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.devices import LocalSimulator
from braket.experimental_capabilities import EnableExperimentalCapability
from braket.ir.openqasm import Program
from qiskit_braket_provider.providers.adapter import (
    _BRAKET_GATE_NAME_TO_QISKIT_GATE,
    _BRAKET_SUPPORTED_NOISES,
    _QISKIT_GATE_NAME_TO_BRAKET_GATE,
    _get_controlled_gateset,
    _validate_angle_restrictions,
    convert_qiskit_to_braket_circuit,
    convert_qiskit_to_braket_circuits,
    native_angle_restrictions,
    to_braket,
    to_qiskit,
)
from qiskit_braket_provider.providers.braket_instructions import CCPRx, MeasureFF

_EPS = 1e-10  # global variable used to chop very small numbers to zero

qiskit_ionq_gates = [
    ionq_gates.GPIGate(Parameter("φ")),
    ionq_gates.GPI2Gate(Parameter("φ")),
    ionq_gates.MSGate(Parameter("φ0"), Parameter("φ1"), Parameter("ϴ")),
]

_BRAKET_SUPPORTED_NOISE_INSTANCES = {
    "kraus": braket_noises.Kraus([np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]),
    "bitflip": braket_noises.BitFlip(0.1),
    "depolarizing": braket_noises.Depolarizing(0.2),
    "amplitudedamping": braket_noises.AmplitudeDamping(0.3),
    "generalizedamplitudedamping": braket_noises.GeneralizedAmplitudeDamping(0.5, 0.4),
    "phasedamping": braket_noises.PhaseDamping(0.4),
    "phaseflip": braket_noises.PhaseFlip(0.3),
    "paulichannel": braket_noises.PauliChannel(0.1, 0.0, 0.0),
    "twoqubitdepolarizing": braket_noises.TwoQubitDepolarizing(0.2),
    "twoqubitdephasing": braket_noises.TwoQubitDephasing(0.3),
    # "twoqubitpaulichannel": braket_noises.TwoQubitPauliChannel({"XX": 0.9}),
}


def check_to_braket_unitary_correct(
    qiskit_circuit: QuantumCircuit, optimization_level: int | None = None
) -> bool:
    """Checks if endianness-reversed Qiskit circuit matrix matches Braket counterpart"""
    return np.allclose(
        to_braket(qiskit_circuit, optimization_level=optimization_level).to_unitary(),
        Operator(qiskit_circuit.decompose()).reverse_qargs().to_matrix(),
    )


def check_to_braket_openqasm_unitary_correct(qasm_program: Program | str):
    """Checks that to_braket converts an OpenQASM correctly"""
    return np.allclose(
        to_braket(qasm_program).to_unitary(), Circuit.from_ir(qasm_program).to_unitary()
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

        with self.assertWarns(DeprecationWarning):
            list(convert_qiskit_to_braket_circuits([qiskit_circuit]))

    def test_u_gate(self):
        """Tests adapter conversion of u gate"""
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.u(np.pi / 2, np.pi / 3, np.pi / 4, 0)

        self.assertTrue(check_to_braket_unitary_correct(qiskit_circuit))

    def test_standard_gate_decomp(self):
        """Tests adapter decomposition of all standard gates to forms that can be translated"""
        gates = qiskit_gates.get_standard_gate_name_mapping()
        for name in {"delay", "global_phase", "measure", "reset"}:
            gates.pop(name)
        for standard_gate in gates.values():
            qiskit_circuit = QuantumCircuit(standard_gate.num_qubits)
            qiskit_circuit.append(standard_gate, range(standard_gate.num_qubits))

            parameters = standard_gate.params
            if parameters:
                parameter_values = [(137 / 61) * np.pi / i for i in range(1, len(parameters) + 1)]
                parameter_bindings = dict(zip(parameters, parameter_values))
                qiskit_circuit = qiskit_circuit.assign_parameters(parameter_bindings)
            with self.subTest(f"Circuit with {standard_gate.name} gate."):
                self.assertTrue(check_to_braket_unitary_correct(qiskit_circuit, 0))

    def test_ionq_gates(self):
        """Tests adapter decomposition of all standard gates to forms that can be translated"""
        target = Target()
        for gate in qiskit_ionq_gates:
            target.add_instruction(gate)

        for gate in qiskit_ionq_gates:
            qiskit_circuit = QuantumCircuit(gate.num_qubits)
            qiskit_circuit.append(gate, range(gate.num_qubits))

            parameters = gate.params
            parameter_values = [(137 / 61) * np.pi / i for i in range(1, len(parameters) + 1)]
            parameter_bindings = dict(zip(parameters, parameter_values))
            qiskit_circuit = qiskit_circuit.assign_parameters(parameter_bindings)

            with self.subTest(f"Circuit with {gate.name} gate."):
                self.assertTrue(
                    np.allclose(
                        to_braket(qiskit_circuit, target=target).to_unitary(),
                        Operator(qiskit_circuit.decompose()).reverse_qargs().to_matrix(),
                    )
                )

    def test_global_phase(self):
        """Tests conversion when transpiler generates a global phase"""
        qiskit_circuit = QuantumCircuit(1, global_phase=np.pi / 2)
        qiskit_circuit.h(0)
        gate = GlobalPhaseGate(1.23)
        qiskit_circuit.append(gate, [])

        braket_circuit = to_braket(qiskit_circuit, optimization_level=2)
        expected_braket_circuit = Circuit().h(0).gphase(1.23 + np.pi / 2)
        self.assertEqual(braket_circuit.global_phase, qiskit_circuit.global_phase + gate.params[0])
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
            "kraus": "kraus",
            "CCPRx": "cc_prx",
            "MeasureFF": "measure_ff",
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
                "unitary",
                "kraus",
            ]
        }

        braket_to_qiskit_gate_names = {
            **qiskit_to_braket_gate_names,
            **{"measure": "measure"},
        }

        self.assertEqual(
            set(qiskit_to_braket_gate_names.keys()),
            set(_QISKIT_GATE_NAME_TO_BRAKET_GATE.keys()),
        )

        self.assertEqual(
            set(braket_to_qiskit_gate_names.values()),
            set(_BRAKET_GATE_NAME_TO_QISKIT_GATE.keys()),
        )

    def test_type_error_on_bad_input(self):
        """Test raising TypeError if adapter does not receive a Qiskit QuantumCircuit."""
        circuit = Mock()

        message = f"Expected a QuantumCircuit, got {type(circuit)} instead."
        with pytest.raises(TypeError, match=message):
            to_braket(circuit)

    def test_target_with_loose_constraints(self):
        """
        Tests that to_braket raises a ValueError if both target and loose constraints are supplied.
        """
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)

        target = Target()
        target.add_instruction(qiskit_gates.HGate())

        with pytest.raises(ValueError):
            to_braket(circuit, target=target, basis_gates={"h"})
        with pytest.raises(ValueError):
            to_braket(circuit, target=target, connectivity=[[0, 1], [1, 2]])

    def test_pass_manager_with_other_arguments(self):
        """
        Tests that to_braket raises a ValueError if pass_manager is supplied
        with target or loose constraints.
        """
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)

        target = Target()
        target.add_instruction(qiskit_gates.HGate())
        pass_manager = generate_preset_pass_manager(2, target=target)

        with pytest.raises(ValueError):
            to_braket(circuit, pass_manager=pass_manager, target=target)
        with pytest.raises(ValueError):
            to_braket(circuit, pass_manager=pass_manager, basis_gates={"h"})
        with pytest.raises(ValueError):
            to_braket(circuit, pass_manager=pass_manager, connectivity=[[0, 1], [1, 2]])

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
            Circuit().h(0).cnot(0, 1).cnot(1, 2).measure(2).measure(0).measure(1)  # pylint: disable=no-member
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
            to_braket(qiskit_circuit, basis_gates={"reset"})

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
            .x(1)
            .cnot(0, 2)
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

        expected_braket_circuit = Circuit().rx(0, FreeParameter("v_0")).ry(0, FreeParameter("v_1"))
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

        braket_circuit = to_braket(
            qiskit_circuit, basis_gates={"h", "cx", "rxx"}, connectivity=connectivity
        )
        braket_circuit_unconnected = to_braket(qiskit_circuit)
        braket_circuit_verbatim = to_braket(
            qiskit_circuit, verbatim=True, connectivity=connectivity
        )

        def gate_matches_connectivity(gate) -> bool:
            return any(
                (gate.target.union(gate.control).issubset(adjacency) for adjacency in connectivity)
            )

        assert all((gate_matches_connectivity(gate) for gate in braket_circuit.instructions))
        assert not all(
            (gate_matches_connectivity(gate) for gate in braket_circuit_unconnected.instructions)
        )
        assert not all(
            (gate_matches_connectivity(gate) for gate in braket_circuit_verbatim.instructions)
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
        braket_circuit = to_braket(circuit, basis_gates={"rx"}, angle_restrictions=restrictions)
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
        braket_circuit = to_braket(circuit, basis_gates={"ms"}, angle_restrictions=restrictions)
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
        braket_circuit = to_braket(circuit, basis_gates={"rx"}, angle_restrictions=restrictions)
        assert len(braket_circuit.instructions) == 1

    def test_kraus_conversion_with_to_braket(self):
        """test qiskit Kraus operator converts to Braket"""
        op = Kraus(
            [
                np.array([[0.5, 0.5], [0.5, 0.5]]),
                np.array([[0.5, -0.5], [-0.5, 0.5]]),
            ]
        )
        qc = QuantumCircuit(1)
        qc.append(Kraus(op), [0])
        bqc = to_braket(qc)
        assert len(bqc.instructions) == 1

        mat = bqc.instructions[0].operator.to_matrix()
        assert np.allclose(mat, op.data)

    def test_braket_noise_to_qiskit_conversion(self):
        """check Kraus matrix conversion of Braket noises to Qiskit"""
        self.assertEqual(
            set(_BRAKET_SUPPORTED_NOISE_INSTANCES.keys()),
            set(_BRAKET_SUPPORTED_NOISES),
        )

        for noise_channel in _BRAKET_SUPPORTED_NOISE_INSTANCES.values():
            if noise_channel.qubit_count == 1:
                instruct = Instruction(noise_channel, target=[0])
            else:
                instruct = Instruction(noise_channel, target=[0, 1])
            qc = Circuit()
            qc.x(0)
            qc.cnot(0, 1)
            qc.add_instruction(instruct)

            qqc = to_qiskit(qc)
            assert len(qqc.data) == 6

            if noise_channel.qubit_count == 1:
                assert np.all(
                    np.isclose(
                        noise_channel.to_matrix(),
                        np.array(qqc.data[2].operation.params),
                    )
                )
            elif noise_channel.qubit_count == 2:
                braket_kraus = [
                    np.reshape(
                        np.transpose(
                            np.reshape(k, [2] * 4),
                            [1, 0, 3, 2],
                        ),
                        (4, 4),
                    )
                    for k in noise_channel.to_matrix()
                ]
                assert np.all(
                    np.isclose(np.array(braket_kraus), np.array(qqc.data[2].operation.params))
                )

    def test_all_braket_noises_converted_simulated(self):
        """check all supported noises and test outcome distributions via braket_dm"""
        qc = Circuit()
        qc.h(0)
        qc.h(1)
        qc.cnot(0, 1)
        for noise_channel in _BRAKET_SUPPORTED_NOISE_INSTANCES.values():
            if noise_channel.qubit_count == 1:
                instruct = Instruction(noise_channel, target=[0])
            else:
                instruct = Instruction(noise_channel, target=[0, 1])
            qc.add_instruction(instruct)
        qc.density_matrix([0, 1])
        qqc = to_qiskit(qc)
        assert len(qqc.data) == 16

        bqc = to_braket(qqc)

        ## removing measurement moments to append density_matrix result
        bqc.moments._moments.popitem(last=True)
        bqc.moments._moments.popitem(last=True)
        bqc._measure_targets = []
        bqc.density_matrix([0, 1])

        res_orig = LocalSimulator("braket_dm").run(qc, shots=0).result().values[0]
        res_conv = LocalSimulator("braket_dm").run(bqc, shots=0).result().values[0]

        assert len(bqc.instructions) == len(qc.instructions)
        assert np.all(np.isclose(res_orig, res_conv))

    def test_kraus_braket_bit_ordering(self):
        """check qiskit <-> braket conversions respect proper bit-ordering"""
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        qc.cx(1, 0)
        qc.append(Kraus([mat]), [0, 1])
        qc.h(1)
        qc.x(1)
        bqc = to_braket(qc)
        res = LocalSimulator("braket_dm").run(bqc, shots=1000).result().measurement_counts
        assert res["11"] == 1000

        mat0 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        mat1 = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        mat2 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        mat3 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

        # in standard notation, this acts on all states and returns |q0q1> = |01>
        # however, qiskit interprets this in reverse order, returning |10>, i.e.
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.append(Kraus([mat0, mat1, mat2, mat3]), [0, 1])

        bqc = to_braket(qc)
        res = LocalSimulator("braket_dm").run(bqc, shots=1000).result().measurement_counts
        assert res["10"] == 1000

        # if we however go from braket -> qiskit -> braket, we expect it to match however
        qc = Circuit()
        qc.h(0).h(0).kraus([0, 1], [mat0, mat1, mat2, mat3])
        qqc = to_braket(to_qiskit(qc))
        res = LocalSimulator("braket_dm").run(qqc, shots=1000).result().measurement_counts
        assert res["01"] == 1000

    def test_roundtrip_openqasm_subroutine(self):
        qasm_string = """
        const int[8] n = 4;
        input bit[n] x;
        
        qubit q;
        
        def parity(bit[n] cin) -> bit {
            bit c = false;
            for int[8] i in [0: n - 1] {
                c ^= cin[i];
            }
            return c;
        }
        
        if (parity(x)) {
            x q;
        } else {
            i q;
        }
        """
        qasm_program = Program(source=qasm_string, inputs={"x": "1011"})
        self.assertTrue(check_to_braket_openqasm_unitary_correct(qasm_program))

    def test_roundtrip_openqasm_custom_gate(self):
        qasm_string = """
        qubit[3] q;
        
        gate majority a, b, c {
            // set c to the majority of {a, b, c}
            ctrl @ x c, b;
            ctrl @ x c, a;
            ctrl(2) @ x a, b, c;
        }
        
        pow(0.5) @ x q[0:1];     // sqrt x
        inv @ v q[1];          // inv of (sqrt x)
        // this should flip q[2] to 1
        majority q[0], q[1], q[2];
        """
        self.assertTrue(check_to_braket_openqasm_unitary_correct(qasm_string))

    def test_conditional_gate_with_condition_attribute(self):
        """Tests that operations with condition attribute raise NotImplementedError."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)

        x_instr = QiskitInstruction("x", 1, 0, [])
        x_instr.condition = (0, 1)
        qc.append(x_instr, [1])

        with pytest.raises(
            NotImplementedError,
            match="Conditional operations are not supported.*Only MeasureFF and CCPRx",
        ):
            to_braket(qc)

    def test_conditional_gate_with_if_test(self):
        """Tests that if_test conditional operations raise NotImplementedError."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.h(1)
        qc.measure(qr, cr)

        with qc.if_test((cr, 3)):
            qc.x(0)

        with pytest.raises(
            NotImplementedError,
            match="Conditional operations are not supported.*Only MeasureFF and CCPRx",
        ):
            to_braket(qc, verbatim=True)

    def test_conditional_gate_with_while_loop(self):
        """Tests that while_loop conditional operations raise NotImplementedError."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.measure(0, 0)

        with qc.while_loop((cr, 1)):
            qc.x(0)
            qc.measure(0, 0)

        with pytest.raises(
            NotImplementedError,
            match="Conditional operations are not supported.*Only MeasureFF and CCPRx",
        ):
            to_braket(qc, verbatim=True)

    def test_ccprx_and_measureff_gates_allowed(self):
        """Tests that CCPRx and MeasureFF gates are allowed and don't raise conditional error."""
        with EnableExperimentalCapability():
            qc = QuantumCircuit(1, 1)
            qc.r(np.pi, 0, 0)
            qc.append(MeasureFF(feedback_key=0), qargs=[0])
            qc.append(CCPRx(np.pi, 0, feedback_key=0), qargs=[0])

            braket_circuit = to_braket(qc, verbatim=True)
            self.assertGreater(len(braket_circuit.instructions), 0)


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

        gate_set = {
            attr
            for attr in dir(Gate)
            if attr[0].isupper() and attr.lower() in _BRAKET_GATE_NAME_TO_QISKIT_GATE
        }
        gate_set -= {"Unitary"}

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
                param_uuids = {param.name: param.uuid for param in qiskit_circuit.parameters}
                params_qiskit = [
                    (
                        Parameter(param.name, uuid=param_uuids.get(param.name))
                        if isinstance(param, FreeParameter)
                        else param
                    )
                    for param in params_braket
                ]

                expected_qiskit_circuit = QuantumCircuit(op.qubit_count)
                qiskit_gate = _BRAKET_GATE_NAME_TO_QISKIT_GATE.get(gate_name.lower())
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
        Tests braket to qiskit conversion with free parameters.
        """
        braket_circuit = Circuit().rx(0, FreeParameter("alpha"))
        qiskit_circuit = to_qiskit(braket_circuit)

        uuid = qiskit_circuit.parameters[0].uuid

        expected_qiskit_circuit = QuantumCircuit(1)
        expected_qiskit_circuit.rx(Parameter("alpha", uuid=uuid), 0)

        expected_qiskit_circuit.measure_all()
        self.assertEqual(qiskit_circuit, expected_qiskit_circuit)

    def test_parametric_pow_gate(self):
        """
        Test braket to qiskit with powers of parameters
        """
        braket_circuit = Circuit().rx(0, FreeParameter("alpha") ** 2)
        qiskit_circuit = to_qiskit(braket_circuit)

        uuid = qiskit_circuit.parameters[0].uuid

        expected_qiskit_circuit = QuantumCircuit(1)
        expected_qiskit_circuit.rx(Parameter("alpha", uuid=uuid) ** 2, 0)

        expected_qiskit_circuit.measure_all()
        self.assertEqual(qiskit_circuit, expected_qiskit_circuit)

    def test_unsupported_parameter_division(self):
        braket_circuit = Circuit().rx(0, 1j * FreeParameter("alpha"))
        with pytest.raises(
            TypeError,
            match="unrecognized parameter type in conversion: <class 'sympy.core.numbers.ImaginaryUnit'>",
        ):
            to_qiskit(braket_circuit)

    def test_unitary(self):
        """
        Tests braket to qiskit conversion with UnitaryGate.
        """
        braket_circuit = Circuit().h(0).unitary([0, 1], Gate.CNot().to_matrix())
        qiskit_circuit = to_qiskit(braket_circuit)

        expected_qiskit_circuit = QuantumCircuit(2)
        expected_qiskit_circuit.h(0)
        expected_qiskit_circuit.unitary(qiskit_gates.CXGate().to_matrix(), [0, 1])
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
                "qiskit_braket_provider.providers.adapter._BRAKET_GATE_NAME_TO_QISKIT_GATE",
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


class TestThereAndBackAgain(TestCase):
    """testing whether or not to_braket and to_qiskit work together"""

    def test_all_standard_gates(self):
        """
        Tests whether or not we can loop
        """

        gate_set = {
            attr
            for attr in dir(Gate)
            if attr[0].isupper() and attr.lower() in _BRAKET_GATE_NAME_TO_QISKIT_GATE
        }

        gate_set -= {"Unitary"}

        # pytest.mark.parametrize is incompatible with TestCase
        param_sets = [
            [0.1, 0.2, 0.3],
            [
                FreeParameter("alpha"),
                FreeParameter("beta"),
                FreeParameter("gamma"),
            ],
            [
                FreeParameter("alpha") + FreeParameter("delta"),
                FreeParameter("beta") + FreeParameter("epsilon"),
                FreeParameter("gamma") ** 2,
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
                qiskit_circuit = to_qiskit(braket_circuit, add_measurements=False)

                # deep copy is necessary to avoid parameter table inconsistency in the MS gate
                qiskit_back_circuit = to_qiskit(
                    to_braket(qiskit_circuit.copy()), add_measurements=False
                )

                num_para = len(qiskit_circuit.parameters)
                values = [0.5, 0.4, 0.8, 0.1, 0.2, 0.3]

                qiskit_circuit = qiskit_circuit.assign_parameters(values[:num_para], inplace=False)
                qiskit_back_circuit = qiskit_back_circuit.assign_parameters(values[:num_para])
                assert np.allclose(
                    Operator(qiskit_circuit).data, Operator(qiskit_back_circuit).data
                )

    def test_simple_travels(self):
        qc = QuantumCircuit(1, 1)
        qc.rz(0.1, 0)
        circ = Circuit().rz(0, 0.1)

        stayed_home = to_qiskit(to_braket(qc), add_measurements=False)  # passes
        lonely_mountain_and_back = to_qiskit(
            to_braket(to_qiskit(circ, add_measurements=False)), add_measurements=False
        )  # fails
        assert np.allclose(Operator(stayed_home).data, Operator(lonely_mountain_and_back).data)
