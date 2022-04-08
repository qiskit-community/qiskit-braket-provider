# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Iterable, List, Union

import numpy

from braket.circuits import gates
from braket.circuits import Instruction, Circuit, result_types
from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)

# TODO: add Angled Gates
# First element is executed first!
_qiskit_2_braket_conversion = {
    "u1": lambda lam: [gates.Rz(lam)],
    "u2": lambda phi, lam: [gates.Rz(lam), gates.Ry(numpy.pi / 2), gates.Rz(phi)],
    "u3": lambda theta, phi, lam: [
        gates.Rz(lam),
        gates.Rx(numpy.pi / 2),
        gates.Rz(theta),
        gates.Rx(-numpy.pi / 2),
        gates.Rz(phi),
    ],
    "cx": lambda: [gates.CNot()],
    "x": lambda: [gates.X()],
    "y": lambda: [gates.Y()],
    "z": lambda: [gates.Z()],
    "t": lambda: [gates.T()],
    "tdg": lambda: [gates.Ti()],
    "s": lambda: [gates.S()],
    "sdg": lambda: [gates.Si()],
    "sx": lambda: [gates.V()],
    "sxdg": lambda: [gates.Vi()],
    "swap": lambda: [gates.Swap()],
    # "rx": lambda: [gates.Rx()],
    #     # "rxx": RXXGate,
    # "ry": lambda: [gates.Ry()],
    # "rz": lambda: [gates.Rz()],
    #     # "rzz": RZZGate,
    "id": lambda: [gates.I()],
    "h": lambda: [gates.H()],
    "cy": lambda: [gates.CY()],
    "cz": lambda: [gates.CZ()],
    #     # "ch": CHGate,
    #     # "crx": CRXGate,
    #     # "cry": CRYGate,
    #     # "crz": CRZGate,
    #     # "cu1": CU1Gate,
    #     # "cu3": CU3Gate,
    "ccx": lambda: [gates.CCNot()],
    "cswap": lambda: [gates.CSwap()],
}


def convert_experiment(circuit: Union[QuantumCircuit, List[QuantumCircuit]]) -> Circuit:
    """Return a Braket quantum circuit from a Qiskit quantum circuit.
     Args:
            circuit (QuantumCircuit): Qiskit Quantum Cricuit

    Returns:
        Circuit: Braket circuit
    """
    quantum_circuit = Circuit()
    for qiskit_gates in circuit.data:
        name = qiskit_gates[0].name
        if name == "measure":
            quantum_circuit.add_result_type(
                result_types.Probability([qiskit_gates[1][0].index])
            )
        elif name == "barrier":
            # This does not exist
            pass
        else:
            params = []
            if hasattr(qiskit_gates[0], "params"):
                params = qiskit_gates[0].params
            for gate in _qiskit_2_braket_conversion[name](*params):
                instruction = Instruction(
                    operator=gate, target=[i.index for i in qiskit_gates[1]]
                )
                quantum_circuit += instruction
    return quantum_circuit


def convert_circuit(circuit: List[QuantumCircuit]) -> Iterable[Circuit]:
    """Converts all Qiskit circuits to Braket circuits.
     Args:
            circuit (List(QuantumCircuit)): Qiskit Quantum Cricuit

    Returns:
        Circuit (Iterable[Circuit]): Braket circuit
    """
    for experiment in circuit:
        yield convert_experiment(experiment)
