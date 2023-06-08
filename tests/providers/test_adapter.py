"""Tests for Qiskti to Braket adapter."""
from unittest import TestCase

from braket.circuits import Circuit, FreeParameter, observables
from braket.devices import LocalSimulator
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter

from qiskit_braket_provider.providers.adapter import (
    convert_qiskit_to_braket_circuit,
    decompose_fully,
    qiskit_gate_name_to_braket_gate_mapping,
    qiskit_gate_names_to_braket_gates,
    qiskit_to_braket_gate_names_mapping,
    wrap_circuits_in_verbatim_box,
)

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import I, Z, X

from qiskit.circuit.library.standard_gates import (
    HGate, CHGate, IGate, PhaseGate, CPhaseGate, MCPhaseGate, RGate, RXGate,
    CRXGate, RXXGate, RYGate, CRYGate, RYYGate, RZGate, CRZGate, RZZGate,
    RZXGate, XXMinusYYGate, XXPlusYYGate, ECRGate, SGate, SdgGate, CSGate,
    CSdgGate, SwapGate, CSwapGate, iSwapGate, SXGate, SXdgGate, CSXGate,
    DCXGate, TGate, TdgGate, UGate, CUGate, U1Gate, CU1Gate, MCU1Gate, U2Gate,
    U3Gate, CU3Gate, XGate, CXGate, CCXGate, C3XGate, C3SXGate, C4XGate,
    RCCXGate, RC3XGate, MCXGate, MCXGrayCode, MCXRecursive, MCXVChain, YGate,
    CYGate, ZGate, CZGate, CCZGate
)

_EPS = 1e-10  # global variable used to chop very small numbers to zero

list_list = [[],[0], [0, 1], [0, 1, 2], [0,1,2,3], [0,1,2,3,4]]

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
        CUGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ"), Parameter("γ")),
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
        RZXGate(Parameter("ϴ")),
        XXMinusYYGate(Parameter("ϴ")),
        XXPlusYYGate(Parameter("ϴ")),
        ECRGate(),
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
        ZGate()
    ]


class TestAdapter(TestCase):
    """Tests adapter."""

    def test_state_preparation_01(self):
        "Tests state_preparation handling of Adapter"
        input_state_vector = np.array([np.sqrt(3)/2, np.sqrt(2)*complex(1,1)/4])

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.prepare_state(input_state_vector, 0)

        braket_circuit = convert_qiskit_to_braket_circuit(qiskit_circuit)
        braket_circuit.state_vector()

        result = LocalSimulator().run(braket_circuit)
        output_state_vector = np.array(result.result().values[0])

        self.assertTrue(
            (np.sqrt(np.sum(np.square(np.abs(input_state_vector - output_state_vector)))) < _EPS)
        )

    def test_state_preparation_00(self):
        "Tests state_preparation handling of Adapter"
        input_state_vector = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
        
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.prepare_state(input_state_vector, 0)

        #Note that the function convert_qiskit_to_braket_circuit
        #operates under the assumption that the qubit it is operating on
        #is originally in the zero state
        braket_circuit = convert_qiskit_to_braket_circuit(qiskit_circuit)
        braket_circuit.state_vector()

        result = LocalSimulator().run(braket_circuit)
        output_state_vector = np.array(result.result().values[0])

        self.assertTrue(
            (np.sqrt(np.sum(np.square(np.abs(input_state_vector - output_state_vector)))) < _EPS)
        )

    def test_u_gate(self):
        qiskit_circuit = QuantumCircuit(1)
        device = LocalSimulator()
        for _ in range(8):
            qiskit_circuit.u(np.pi/2, np.pi/3, np.pi/4, 0)

            simulator = Aer.get_backend('statevector_simulator')
            job = execute(qiskit_circuit, simulator)

            braket_circuit = convert_qiskit_to_braket_circuit(qiskit_circuit)
            braket_circuit.state_vector()

            braket_output = device.run(braket_circuit).result().values[0]
            qiskit_output = np.array(job.result().get_statevector(qiskit_circuit))

            self.assertTrue(
                np.sqrt(np.sum(np.square(np.abs(braket_output - qiskit_output)))) < _EPS
            )

    def test_standard_gate_decomp(self):
        translatable = True
        for standard_gate in standard_gates:
            circuit = QuantumCircuit(5)
            circuit.append(standard_gate,list_list[standard_gate.num_qubits])
            decomp_circuit = decompose_fully(circuit)
            for simple_gate in decomp_circuit.data:
                translatable = translatable and simple_gate[0].name in qiskit_gate_names_to_braket_gates.keys()
        self.assertTrue(translatable)

    def test_exponential_gate_decomp_00(self):
        translatable = True

        operator = (Z ^ Z) - 0.1 * (X ^ I)
        evo = PauliEvolutionGate(operator, time=0.2)

        # plug it into a circuit
        circuit = QuantumCircuit(2)
        circuit.append(evo, range(2))
        decomp_circuit = decompose_fully(circuit)
        for simple_gate in decomp_circuit.data:
            translatable = translatable and simple_gate[0].name in qiskit_gate_names_to_braket_gates.keys()
        self.assertTrue(translatable)

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
            .rz(0, FreeParameter("λ"))
            .ry(0, FreeParameter("θ"))
            .rz(0, FreeParameter("φ"))
            .phaseshift(0, (FreeParameter("φ")+FreeParameter("λ")) * (0.5))
            .x(0)
            .phaseshift(0, (FreeParameter("φ")+FreeParameter("λ")) * (0.5))
            .x(0)
            .rz(0, np.pi)
            .ry(0, FreeParameter("θ"))
            .rz(0, FreeParameter("φ"))
            .phaseshift(0, (FreeParameter("φ")+np.pi) * (0.5))
            .x(0)
            .phaseshift(0, (FreeParameter("φ")+np.pi) * (0.5))
            .x(0)
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
