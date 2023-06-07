"""Qiskit's circuits transpilation to Braket"""
from qiskit import transpile as qiskit_transpile

def transpile(circuit):
    BRAKET_GATES = ['u1', 'u2', 'u3', 'p', 'cx', 'x', 'y', 'z', 't', 'tdg', 's', 'sdg', 'sx', 'sxdg', 'swap', 'rx', 'ry', 'rz', 'rzz', 'id', 'h', 'cy', 'cz', 'ccx', 'cswap', 'cp', 'rxx', 'ryy', 'ecr']
    return qiskit_transpile(circuit, basis_gates=BRAKET_GATES)