"""Example of usage of Qiskit-Braket provider."""

from qiskit import QuantumCircuit

from braket.jobs import save_job_result
from qiskit_braket_provider import AWSBraketProvider

provider = AWSBraketProvider()
backend = provider.get_backend("SV1")
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

results = backend.run(circuit, shots=1)

print(results.result().get_counts())
save_job_result(results.result().get_counts())
