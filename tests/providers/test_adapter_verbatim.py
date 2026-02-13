"""Tests for verbatim pragma support in Qiskit to Braket adapter."""

from unittest import TestCase

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp

from braket.circuits import Circuit
from braket.ir.openqasm import Program
from qiskit_braket_provider import to_qiskit


class TestVerbatimPragmaSupport(TestCase):
    """Tests for verbatim pragma support."""

    def test_single_verbatim_box_with_gates(self):
        """Tests conversion of a single verbatim box containing gates."""
        openqasm_str = """
OPENQASM 3.0;
qubit[2] q;
#pragma braket verbatim
box {
    h q[0];
    cnot q[0], q[1];
}
"""
        qiskit_circuit = to_qiskit(openqasm_str)
        
        # Check that circuit has the expected structure
        self.assertEqual(qiskit_circuit.num_qubits, 2)
        
        # Find BoxOp instructions
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 1, "Should have exactly one BoxOp")
        
        # Check BoxOp label
        box_op = box_ops[0]
        self.assertEqual(box_op.operation.label, "verbatim")
        
        # Check BoxOp contains the correct gates
        verbatim_circuit = box_op.operation.body
        self.assertEqual(len(verbatim_circuit.data), 2, "Verbatim box should contain 2 gates")
        self.assertEqual(verbatim_circuit.data[0].operation.name, "h")
        self.assertEqual(verbatim_circuit.data[1].operation.name, "cx")

    def test_multiple_verbatim_boxes(self):
        """Tests conversion of multiple verbatim boxes."""
        openqasm_str = """
OPENQASM 3.0;
qubit[2] q;
#pragma braket verbatim
box {
    h q[0];
}
x q[1];
#pragma braket verbatim
box {
    cnot q[0], q[1];
}
"""
        qiskit_circuit = to_qiskit(openqasm_str)
        
        # Find BoxOp instructions
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 2, "Should have exactly two BoxOps")
        
        # Check both BoxOps have the same label
        self.assertEqual(box_ops[0].operation.label, "verbatim")
        self.assertEqual(box_ops[1].operation.label, "verbatim")
        
        # Check first verbatim box contains H gate
        self.assertEqual(len(box_ops[0].operation.body.data), 1)
        self.assertEqual(box_ops[0].operation.body.data[0].operation.name, "h")
        
        # Check second verbatim box contains CNOT gate
        self.assertEqual(len(box_ops[1].operation.body.data), 1)
        self.assertEqual(box_ops[1].operation.body.data[0].operation.name, "cx")
        
        # Check that X gate is outside verbatim boxes (in main circuit)
        non_box_gates = [instr for instr in qiskit_circuit.data 
                        if not isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(non_box_gates), 1)
        self.assertEqual(non_box_gates[0].operation.name, "x")

    def test_custom_verbatim_box_name(self):
        """Tests conversion with custom verbatim box name."""
        openqasm_str = """
OPENQASM 3.0;
qubit[2] q;
#pragma braket verbatim
box {
    h q[0];
}
"""
        qiskit_circuit = to_qiskit(openqasm_str, verbatim_box_name="custom_verbatim")
        
        # Find BoxOp instructions
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 1)
        
        # Check custom label
        self.assertEqual(box_ops[0].operation.label, "custom_verbatim")

    def test_gates_outside_verbatim_box(self):
        """Tests that gates outside verbatim boxes go to main circuit."""
        openqasm_str = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
#pragma braket verbatim
box {
    cnot q[0], q[1];
}
x q[1];
"""
        qiskit_circuit = to_qiskit(openqasm_str)
        
        # Check main circuit has gates outside verbatim box
        non_box_gates = [instr for instr in qiskit_circuit.data 
                        if not isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(non_box_gates), 2, "Should have 2 gates outside verbatim box")
        self.assertEqual(non_box_gates[0].operation.name, "h")
        self.assertEqual(non_box_gates[1].operation.name, "x")
        
        # Check verbatim box has CNOT
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 1)
        self.assertEqual(len(box_ops[0].operation.body.data), 1)
        self.assertEqual(box_ops[0].operation.body.data[0].operation.name, "cx")

    def test_empty_verbatim_box(self):
        """Tests conversion of an empty verbatim box."""
        openqasm_str = """
OPENQASM 3.0;
qubit[2] q;
#pragma braket verbatim
box {
}
"""
        qiskit_circuit = to_qiskit(openqasm_str)
        
        # Find BoxOp instructions
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 1)
        
        # Check verbatim box is empty
        self.assertEqual(len(box_ops[0].operation.body.data), 0)

    def test_nested_verbatim_box_error(self):
        """Tests that nested verbatim boxes raise an error."""
        # The OpenQASM parser itself rejects nested pragmas with "pragmas must be global"
        # So we test this at the context level directly
        from braket.default_simulator.openqasm.interpreter import VerbatimBoxDelimiter
        from qiskit_braket_provider.providers.adapter import _QiskitProgramContext
        
        context = _QiskitProgramContext()
        context.add_qubits("q", 2)
        
        # Start first verbatim box
        context.add_verbatim_marker(VerbatimBoxDelimiter.START_VERBATIM)
        
        # Try to start nested verbatim box - should raise error
        with pytest.raises(ValueError, match="Nested verbatim boxes are not supported"):
            context.add_verbatim_marker(VerbatimBoxDelimiter.START_VERBATIM)

    def test_unclosed_verbatim_box_error(self):
        """Tests that unclosed verbatim box raises an error."""
        # This test depends on how the interpreter handles unclosed boxes
        # The interpreter might catch this before our code does
        openqasm_str = """
OPENQASM 3.0;
qubit[2] q;
#pragma braket verbatim
box {
    h q[0];
"""
        # The interpreter should catch syntax errors
        with pytest.raises(Exception):  # Could be various exception types
            to_qiskit(openqasm_str)

    def test_verbatim_end_without_start_error(self):
        """Tests that ending a verbatim box without starting one raises an error."""
        # This is tricky to test directly since the OpenQASM parser handles box syntax
        # We would need to manually call add_verbatim_marker(END_VERBATIM) without START_VERBATIM
        from braket.default_simulator.openqasm.interpreter import VerbatimBoxDelimiter
        from qiskit_braket_provider.providers.adapter import _QiskitProgramContext
        
        context = _QiskitProgramContext()
        context.add_qubits("q", 2)
        
        with pytest.raises(ValueError, match="Verbatim box end marker without matching start"):
            context.add_verbatim_marker(VerbatimBoxDelimiter.END_VERBATIM)

    def test_verbatim_box_with_measurements(self):
        """Tests verbatim box with measurements inside."""
        openqasm_str = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
#pragma braket verbatim
box {
    h q[0];
    cnot q[0], q[1];
}
c[0] = measure q[0];
c[1] = measure q[1];
"""
        qiskit_circuit = to_qiskit(openqasm_str)
        
        # Check BoxOp exists
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 1)
        
        # Check measurements are outside the box
        measurements = [instr for instr in qiskit_circuit.data 
                       if instr.operation.name == "measure"]
        self.assertEqual(len(measurements), 2)

    def test_verbatim_box_qubit_mapping(self):
        """Tests that qubit indices are correctly mapped in verbatim boxes."""
        openqasm_str = """
OPENQASM 3.0;
qubit[3] q;
#pragma braket verbatim
box {
    h q[0];
    cnot q[1], q[2];
}
"""
        qiskit_circuit = to_qiskit(openqasm_str)
        
        # Find BoxOp
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 1)
        
        # Check gates in verbatim box have correct qubit indices
        verbatim_circuit = box_ops[0].operation.body
        h_gate = verbatim_circuit.data[0]
        cnot_gate = verbatim_circuit.data[1]
        
        # Use _index attribute for Qiskit qubits
        self.assertEqual(h_gate.qubits[0]._index, 0)
        self.assertEqual(cnot_gate.qubits[0]._index, 1)
        self.assertEqual(cnot_gate.qubits[1]._index, 2)

    def test_circuit_without_verbatim_pragma(self):
        """Tests that circuits without verbatim pragmas work as before."""
        openqasm_str = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
"""
        qiskit_circuit = to_qiskit(openqasm_str)
        
        # Should have no BoxOps
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 0)
        
        # Should have gates directly in main circuit
        self.assertEqual(len(qiskit_circuit.data), 2)
        self.assertEqual(qiskit_circuit.data[0].operation.name, "h")
        self.assertEqual(qiskit_circuit.data[1].operation.name, "cx")

    def test_braket_circuit_without_verbatim(self):
        """Tests that Braket circuits without verbatim pragmas work as before."""
        braket_circuit = Circuit().h(0).cnot(0, 1)
        qiskit_circuit = to_qiskit(braket_circuit, add_measurements=False)
        
        # Should have no BoxOps
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 0)
        
        # Should have gates directly in main circuit
        self.assertEqual(len(qiskit_circuit.data), 2)
        self.assertEqual(qiskit_circuit.data[0].operation.name, "h")
        self.assertEqual(qiskit_circuit.data[1].operation.name, "cx")

    def test_braket_program_without_verbatim(self):
        """Tests that Braket Programs without verbatim pragmas work as before."""
        program = Program(
            source="""
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
"""
        )
        qiskit_circuit = to_qiskit(program)
        
        # Should have no BoxOps
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 0)
