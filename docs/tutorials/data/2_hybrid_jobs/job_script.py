"""Example of Hybrid Job payload with VQE."""
from braket.jobs import save_job_result
from qiskit.opflow import (
    I,
    X,
    Z,
)
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance

from qiskit_braket_provider import AWSBraketProvider


def main():
    backend = AWSBraketProvider().get_backend("SV1")

    h2_op = (
        (-1.052373245772859 * I ^ I)
        + (0.39793742484318045 * I ^ Z)
        + (-0.39793742484318045 * Z ^ I)
        + (-0.01128010425623538 * Z ^ Z)
        + (0.18093119978423156 * X ^ X)
    )

    quantum_instance = QuantumInstance(
        backend, seed_transpiler=42, seed_simulator=42, shots=10
    )
    ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
    slsqp = SLSQP(maxiter=1)

    vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=quantum_instance)

    vqe_result = vqe.compute_minimum_eigenvalue(h2_op)

    save_job_result(
        {
            "VQE": {
                "eigenstate": vqe_result.eigenstate,
                "eigenvalue": vqe_result.eigenvalue.real,
                "optimal_parameters": list(vqe_result.optimal_parameters.values()),
                "optimal_point": vqe_result.optimal_point.tolist(),
                "optimal_value": vqe_result.optimal_value.real,
            }
        }
    )
