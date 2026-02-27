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
#pragma braket verbatim
box {
    h $0;
    cnot $0, $1;
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
#pragma braket verbatim
box {
    h $0;
}
x $1;
#pragma braket verbatim
box {
    cnot $0, $1;
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
#pragma braket verbatim
box {
    h $0;
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
h $0;
#pragma braket verbatim
box {
    cnot $0, $1;
}
x $1;
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
#pragma braket verbatim
box {
    h $0;
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
bit[2] c;
#pragma braket verbatim
box {
    h $0;
    cnot $0, $1;
}
c[0] = measure $0;
c[1] = measure $1;
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
#pragma braket verbatim
box {
    h $0;
    cnot $1, $2;
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
        
        # Check qubit indices by finding their position in the circuit's qubit list
        h_qubit_idx = verbatim_circuit.find_bit(h_gate.qubits[0]).index
        cnot_qubit0_idx = verbatim_circuit.find_bit(cnot_gate.qubits[0]).index
        cnot_qubit1_idx = verbatim_circuit.find_bit(cnot_gate.qubits[1]).index
        
        self.assertEqual(h_qubit_idx, 0)
        self.assertEqual(cnot_qubit0_idx, 1)
        self.assertEqual(cnot_qubit1_idx, 2)

    def test_circuit_without_verbatim_pragma(self):
        """Tests that circuits without verbatim pragmas work as before."""
        openqasm_str = """
OPENQASM 3.0;
h $0;
cnot $0, $1;
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
h $0;
cnot $0, $1;
"""
        )
        qiskit_circuit = to_qiskit(program)
        
        # Should have no BoxOps
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 0)

    def test_non_contiguous_physical_qubits(self):
        """Tests that non-contiguous physical qubits are correctly mapped."""
        openqasm_str = """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $2;
    cnot $2, $5;
}
"""
        qiskit_circuit = to_qiskit(openqasm_str)
        
        # Circuit should have 6 qubits (0-5) to accommodate qubit 5
        self.assertEqual(qiskit_circuit.num_qubits, 6)
        
        # Find BoxOp
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 1)
        
        # Check gates in verbatim box use correct qubit indices
        verbatim_circuit = box_ops[0].operation.body
        self.assertEqual(len(verbatim_circuit.data), 2)
        
        h_gate = verbatim_circuit.data[0]
        cnot_gate = verbatim_circuit.data[1]
        
        # Verify H gate is on qubit 2
        h_qubit_idx = verbatim_circuit.find_bit(h_gate.qubits[0]).index
        self.assertEqual(h_qubit_idx, 2)
        
        # Verify CNOT gate is on qubits 2 and 5
        cnot_qubit0_idx = verbatim_circuit.find_bit(cnot_gate.qubits[0]).index
        cnot_qubit1_idx = verbatim_circuit.find_bit(cnot_gate.qubits[1]).index
        self.assertEqual(cnot_qubit0_idx, 2)
        self.assertEqual(cnot_qubit1_idx, 5)

    def test_unclosed_verbatim_box_circuit_property_error(self):
        """Tests that accessing circuit property with unclosed verbatim box raises ValueError."""
        from braket.default_simulator.openqasm.interpreter import VerbatimBoxDelimiter
        from qiskit_braket_provider.providers.adapter import _QiskitProgramContext
        
        context = _QiskitProgramContext()
        context.add_qubits("q", 2)
        
        # Start verbatim box
        context.add_verbatim_marker(VerbatimBoxDelimiter.START_VERBATIM)
        
        # Try to access circuit property while in verbatim box - should raise error
        with pytest.raises(ValueError, match="Unclosed verbatim box at end of program"):
            _ = context.circuit

    def test_bit_declaration_with_identifier_size(self):
        """Tests that bit declarations with identifier sizes are handled correctly.
        
        This tests that a classical bit declaration isn't declared as a classical register
        when the size is an Identifier or expression that can't be determined yet.
        """
        # Create OpenQASM 3.0 program with bit declaration using identifier size
        # This happens in function parameters like bit[n] where n is a variable
        openqasm_str = """
OPENQASM 3.0;
input bit[n] alpha;
h $0;
"""
        
        # This should not raise an error and should handle the identifier size gracefully
        qiskit_circuit = to_qiskit(openqasm_str)
        
        # Verify circuit is created successfully
        self.assertIsNotNone(qiskit_circuit)
        self.assertEqual(qiskit_circuit.num_qubits, 1)

    def test_verbatim_box_adds_qubits_to_main_circuit(self):
        """Tests that verbatim boxes add qubits to main circuit when needed.
        
        This tests that qubits are added to the main circuit
        when the verbatim circuit has more qubits than the main circuit.
        """
        # Create OpenQASM 3.0 program where verbatim box is the first thing
        # This means the main circuit starts with 0 qubits, and the verbatim box
        # will need to add qubits to accommodate its gates
        openqasm_str = """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $0;
    cnot $0, $1;
    cnot $1, $2;
}
x $0;
"""
        
        # Convert to Qiskit
        qiskit_circuit = to_qiskit(openqasm_str)
        
        # Verify circuit has enough qubits for the verbatim box
        self.assertIsNotNone(qiskit_circuit)
        # The verbatim box uses 3 qubits ($0, $1, $2), so main circuit should have at least 3
        self.assertGreaterEqual(qiskit_circuit.num_qubits, 3)
        
        # Verify BoxOp is created
        box_ops = [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]
        self.assertEqual(len(box_ops), 1, "Expected one verbatim BoxOp")
        
        # Verify the BoxOp has 3 qubits
        self.assertEqual(box_ops[0].operation.body.num_qubits, 3)
