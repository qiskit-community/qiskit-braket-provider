"""Test script to see how to_qiskit() handles OpenQASM 3 if-else statements."""

from qiskit_braket_provider.providers.adapter import to_qiskit

# Simple OpenQASM 3 program with if-else statement
openqasm_program = """
OPENQASM 3.0;

// Declare qubits and classical bits
qubit[2] q;
bit c0;
bit c1;

// Apply Hadamard to first qubit
h q[0];

// Measure first qubit
c0 = measure q[0];

// If-else based on measurement result
if (c0 == 1) {
    x q[1];  // Apply X gate if measurement is 1
} else {
    h q[1];  // Apply H gate if measurement is 0
}

// Measure second qubit
c1 = measure q[1];
"""

print("OpenQASM 3 Program:")
print("=" * 60)
print(openqasm_program)
print("=" * 60)
print()

try:
    # Convert to Qiskit
    qiskit_circuit = to_qiskit(openqasm_program)
    
    print("Converted Qiskit Circuit:")
    print("=" * 60)
    print(qiskit_circuit)
    print()
    
    print("Circuit Diagram:")
    print("=" * 60)
    print(qiskit_circuit.draw(output='text'))
    print()
    
    print("Circuit Instructions:")
    print("=" * 60)
    for i, instruction in enumerate(qiskit_circuit.data):
        print(f"{i}: {instruction.operation.name} on qubits {instruction.qubits}, clbits {instruction.clbits}")
    
except Exception as e:  # noqa: BLE001
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
