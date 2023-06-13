"""Tests for Qiskti to Braket adapter."""
from unittest import TestCase

from braket.circuits import Circuit, FreeParameter, observables
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qiskit_braket_provider.providers.adapter import (
    convert_qiskit_to_braket_circuit,
    qiskit_gate_name_to_braket_gate_mapping,
    qiskit_gate_names_to_braket_gates,
    qiskit_to_braket_gate_names_mapping,
    wrap_circuits_in_verbatim_box,
)


class TestAdapter(TestCase):
    """Tests adapter."""

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
            .rx(0, np.pi / 2)
            .rz(0, FreeParameter("θ"))
            .rx(0, -np.pi / 2)
            .rz(0, FreeParameter("φ"))
            .rz(0, np.pi)
            .rx(0, np.pi / 2)
            .rz(0, FreeParameter("θ"))
            .rx(0, -np.pi / 2)
            .rz(0, FreeParameter("φ"))
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
