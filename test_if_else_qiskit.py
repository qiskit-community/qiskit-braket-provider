"""Test script to see how Qiskit natively handles if-else statements."""

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

# Create a Qiskit circuit with if-else using Qiskit's native control flow
qr = QuantumRegister(2, 'q')
cr = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qr, cr)

# Apply Hadamard to first qubit
circuit.h(0)

# Measure first qubit
circuit.measure(0, 0)

# Create if-else using Qiskit's control flow
# If c[0] == 1, apply X gate, else apply H gate
with circuit.if_test((cr[0], 1)) as else_:
    circuit.x(1)
with else_:
    circuit.h(1)

# Measure second qubit
circuit.measure(1, 1)

print("Qiskit Circuit with if-else:")
print("=" * 60)
print(circuit)
print()

print("Circuit Diagram:")
print("=" * 60)
print(circuit.draw(output='text'))
print()

print("Circuit Instructions:")
print("=" * 60)
for i, instruction in enumerate(circuit.data):
    print(f"{i}: {instruction.operation.name} on qubits {instruction.qubits}, clbits {instruction.clbits}")
print()

# Now let's try to convert this to OpenQASM 3
print("OpenQASM 3 representation:")
print("=" * 60)
try:
    qasm3_str = circuit.qasm3()
    print(qasm3_str)
except Exception as e:  # noqa: BLE001
    print(f"Error: {type(e).__name__}: {e}")
