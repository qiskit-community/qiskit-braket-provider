"""Test script to evaluate circuit performance with #pragma braket verbatim."""

from qiskit_braket_provider import to_braket, to_qiskit

# OpenQASM 3.0 circuit with verbatim pragma
oq3 = '''OPENQASM 3.0;
h $0;
#pragma braket verbatim
box {
    cnot $0, $1;
}
'''

def test_verbatim_pragma():
   
    braket_circuit = to_qiskit(oq3)
    print(braket_circuit.qregs)
    print(braket_circuit.cregs)
    braket_circuit = to_braket(braket_circuit)
    

if __name__ == "__main__":
    test_verbatim_pragma()
