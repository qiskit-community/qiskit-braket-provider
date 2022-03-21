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

import braket.circuits.gates as gates
import numpy
from braket.circuits import Instruction, Circuit, result_types, Gate
from qiskit import QuantumCircuit
from qiskit.qobj import QasmQobj, QasmQobjExperiment, QasmQobjInstruction

logger = logging.getLogger(__name__)

# _qiskit_2_braket_conversion = {
#     # "u1": U1Gate,
#     # "u2": U2Gate,
#     # "u3": U3Gate,
#     "x": gates.X,
#     "y": gates.Y,
#     "z": gates.Z,
#     "t": gates.T,
#     "tdg": gates.Ti,
#     "s": gates.S,
#     "sdg": gates.Si,
#     "sx": gates.V,
#     "sxdg": gates.Vi,
#     "swap": gates.Swap,
#     "rx": gates.Rx,
#     # "rxx": RXXGate,
#     "ry": gates.Ry,
#     "rz": gates.Rz,
#     # "rzz": RZZGate,
#     "id": gates.I,
#     "h": gates.H,
#     "cx": gates.CNot,
#     "cy": gates.CY,
#     "cz": gates.CZ,
#     # "ch": CHGate,
#     # "crx": CRXGate,
#     # "cry": CRYGate,
#     # "crz": CRZGate,
#     # "cu1": CU1Gate,
#     # "cu3": CU3Gate,
#     "ccx": gates.CCNot,
#     "cswap": gates.CSwap
# }

# TODO: look into a possibility to use device's native gates set (no the IBMQ natives!)
# First element is executed first!
_qiskit_2_braket_conversion = {
    "u1": lambda lam: [gates.Rz(lam)],
    "u2": lambda phi, lam: [gates.Rz(lam), gates.Ry(numpy.pi / 2), gates.Rz(phi)],
    "u3": lambda theta, phi, lam: [gates.Rz(lam),
                                   gates.Rx(numpy.pi / 2),
                                   gates.Rz(theta),
                                   gates.Rx(-numpy.pi / 2),
                                   gates.Rz(phi)],
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
    "cswap": lambda: [gates.CSwap()]

}


def convert_experiment(circuit) -> Circuit:
    qc = Circuit()
    print(circuit)
    for qiskitGates in circuit.data:
        print("--N-")
        print(qiskitGates[0].name)
        name = qiskitGates[0].name
        if name == 'measure':
            print("--m--")
            print([qiskitGates[1][0].index])
            qc.add_result_type(result_types.Probability([qiskitGates[1][0].index]))
        elif name == 'barrier':
            # This does not exist
            pass
        else:
            params = []
            if hasattr(qiskitGates[0], 'params'):
                print("--P--")
                print(qiskitGates[0].params)
                params = qiskitGates[0].params
            gates: List[Gate] = _qiskit_2_braket_conversion[name](*params)
            for gate in gates:
                print("operator")
                print(gate)
                print("qbit")
                print([i.index for i in qiskitGates[1]])
                instruction = Instruction(operator=gate, target=[i.index for i in qiskitGates[1]])
                qc += instruction
        print()
        print()
        print()
    print(qc)
    return qc


def convert_circuit(circuit: Union[QuantumCircuit, List[QuantumCircuit]]) -> Iterable[Circuit]:
    for experiment in circuit:
        yield convert_experiment(experiment)
