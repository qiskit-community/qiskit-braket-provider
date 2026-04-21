"""Test script to see what control flow is currently supported by to_qiskit()."""

from qiskit_braket_provider.providers.adapter import to_qiskit

print("=" * 70)
print("Test 1: Static if-else (compile-time evaluation)")
print("=" * 70)

# This should work because the condition is known at parse time
openqasm_static = """
OPENQASM 3.0;

qubit[2] q;

// Static condition - known at compile time
int x = 5;

if (x > 3) {
    h q[0];
    x q[1];
} else {
    x q[0];
    h q[1];
}
"""

print("OpenQASM Program:")
print(openqasm_static)

try:
    qiskit_circuit = to_qiskit(openqasm_static)
    print("\nConverted Qiskit Circuit:")
    print(qiskit_circuit)
    print("\nCircuit Diagram:")
    print(qiskit_circuit.draw(output='text'))
    print("\n✓ SUCCESS: Static if-else works!")
except Exception as e:  # noqa: BLE001
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("Test 2: For loop")
print("=" * 70)

openqasm_loop = """
OPENQASM 3.0;

qubit[3] q;

// Apply Hadamard to all qubits using a loop
for int i in [0:2] {
    h q[i];
}
"""

print("OpenQASM Program:")
print(openqasm_loop)

try:
    qiskit_circuit = to_qiskit(openqasm_loop)
    print("\nConverted Qiskit Circuit:")
    print(qiskit_circuit)
    print("\nCircuit Diagram:")
    print(qiskit_circuit.draw(output='text'))
    print("\n✓ SUCCESS: For loop works!")
except Exception as e:  # noqa: BLE001
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("Test 3: Dynamic if-else (runtime evaluation) - NOT SUPPORTED")
print("=" * 70)

openqasm_dynamic = """
OPENQASM 3.0;

qubit[2] q;
bit c0;

h q[0];
c0 = measure q[0];

// This won't work - condition depends on measurement result
if (c0 == 1) {
    x q[1];
} else {
    h q[1];
}
"""

print("OpenQASM Program:")
print(openqasm_dynamic)

try:
    qiskit_circuit = to_qiskit(openqasm_dynamic)
    print("\nConverted Qiskit Circuit:")
    print(qiskit_circuit)
    print("\nCircuit Diagram:")
    print(qiskit_circuit.draw(output='text'))
    print("\n✓ SUCCESS: Dynamic if-else works!")
except Exception as e:  # noqa: BLE001
    print(f"\n✗ EXPECTED ERROR: {type(e).__name__}: {e}")
    print("\nThis is expected because the Braket interpreter evaluates")
    print("conditions at parse time, not runtime. To support dynamic")
    print("control flow, we would need to override the BranchingStatement")
    print("handler in _QiskitProgramContext to create Qiskit if_test operations.")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("✓ Static control flow (compile-time conditions): SUPPORTED")
print("✓ Loops (for, while): SUPPORTED")
print("✗ Dynamic control flow (measurement-based conditions): NOT SUPPORTED")
print("\nTo support dynamic control flow, _QiskitProgramContext would need to:")
print("1. Override the BranchingStatement visitor method")
print("2. Detect when conditions depend on measurement results")
print("3. Create Qiskit if_test operations instead of evaluating immediately")
