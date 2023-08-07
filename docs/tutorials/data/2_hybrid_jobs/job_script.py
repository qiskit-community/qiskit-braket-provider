"""Example of Hybrid Job payload with VQE."""
from braket.jobs import save_job_result
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import BackendEstimator

from qiskit_braket_provider import AWSBraketProvider


def main():
    backend = AWSBraketProvider().get_backend("SV1")

    h2_op = SparsePauliOp(
        ["II", "IZ", "ZI", "ZZ", "XX"],
        coeffs=[
            -1.052373245772859,
            0.39793742484318045,
            -0.39793742484318045,
            -0.01128010425623538,
            0.18093119978423156,
        ],
    )

    estimator = BackendEstimator(
        backend=backend,
        options={"seed_simulator": 42, "seed_transpiler": 42, "shots": 10},
        skip_transpilation=False,
    )
    ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
    slsqp = SLSQP(maxiter=1)

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=slsqp)

    vqe_result = vqe.compute_minimum_eigenvalue(h2_op)

    save_job_result(
        {
            "VQE": {
                "eigenvalue": vqe_result.eigenvalue.real,
                "optimal_parameters": list(vqe_result.optimal_parameters.values()),
                "optimal_point": vqe_result.optimal_point.tolist(),
                "optimal_value": vqe_result.optimal_value.real,
            }
        }
    )
