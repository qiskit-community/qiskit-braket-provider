from math import pi
from numbers import Number
from typing import Iterable

import qiskit.circuit.library as qiskit_gates
import qiskit.quantum_info as qiskit_qi
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Gate, Parameter
from sympy import Expr

from braket.default_simulator.openqasm.program_context import AbstractProgramContext
from qiskit.circuit import Instruction as QiskitInstruction
from qiskit_ionq import ionq_gates

from qiskit_braket_provider.providers import braket_instructions

_BRAKET_GATE_NAME_TO_QISKIT_GATE: dict[str, QiskitInstruction | None] = {
    "u": qiskit_gates.UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
    "u1": qiskit_gates.U1Gate(Parameter("theta")),
    "u2": qiskit_gates.U2Gate(Parameter("theta"), Parameter("lam")),
    "u3": qiskit_gates.U3Gate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
    "h": qiskit_gates.HGate(),
    "ccnot": qiskit_gates.CCXGate(),
    "cnot": qiskit_gates.CXGate(),
    "cphaseshift": qiskit_gates.CPhaseGate(Parameter("theta")),
    "cswap": qiskit_gates.CSwapGate(),
    "cy": qiskit_gates.CYGate(),
    "cz": qiskit_gates.CZGate(),
    "i": qiskit_gates.IGate(),
    "phaseshift": qiskit_gates.PhaseGate(Parameter("theta")),
    "rx": qiskit_gates.RXGate(Parameter("theta")),
    "ry": qiskit_gates.RYGate(Parameter("theta")),
    "rz": qiskit_gates.RZGate(Parameter("phi")),
    "s": qiskit_gates.SGate(),
    "si": qiskit_gates.SdgGate(),
    "swap": qiskit_gates.SwapGate(),
    "t": qiskit_gates.TGate(),
    "ti": qiskit_gates.TdgGate(),
    "v": qiskit_gates.SXGate(),
    "vi": qiskit_gates.SXdgGate(),
    "x": qiskit_gates.XGate(),
    "xx": qiskit_gates.RXXGate(Parameter("theta")),
    "y": qiskit_gates.YGate(),
    "yy": qiskit_gates.RYYGate(Parameter("theta")),
    "z": qiskit_gates.ZGate(),
    "zz": qiskit_gates.RZZGate(Parameter("theta")),
    "ecr": qiskit_gates.ECRGate(),
    "prx": qiskit_gates.RGate(Parameter("theta"), Parameter("phi")),
    "iswap": qiskit_gates.iSwapGate(),
    "gpi": ionq_gates.GPIGate(Parameter("phi") / (2 * pi)),
    "gpi2": ionq_gates.GPI2Gate(Parameter("phi") / (2 * pi)),
    "ms": ionq_gates.MSGate(
        Parameter("phi0") / (2 * pi),
        Parameter("phi1") / (2 * pi),
        Parameter("theta") / (2 * pi),
        ),
    "gphase": qiskit_gates.GlobalPhaseGate(Parameter("theta")),
    "measure": qiskit_gates.Measure(),
    "unitary": qiskit_gates.UnitaryGate,
    "kraus": qiskit_qi.Kraus,
    "cc_prx": braket_instructions.CCPRx(
        Parameter("angle_1"), Parameter("angle_2"), Parameter("feedback_key")
    ),
    "measure_ff": braket_instructions.MeasureFF(Parameter("feedback_key")),
}


class QiskitProgramContext(AbstractProgramContext):
    def __init__(self):
        super().__init__()
        self._circuit = QuantumCircuit()

    @property
    def circuit(self):
        return self._circuit

    def add_qubits(self, name: str, num_qubits: int | None = 1) -> None:
        super().add_qubits(name, num_qubits)
        self._circuit.add_register(num_qubits)

    def is_builtin_gate(self, name: str) -> bool:
        return name in _BRAKET_GATE_NAME_TO_QISKIT_GATE

    def add_phase_instruction(self, target, phase_value):
        self._circuit.global_phase *= phase_value

    def add_gate_instruction(
        self, gate_name: str, target: tuple[int, ...], params, ctrl_modifiers: list[int], power: int
    ):
        gate: Gate = _BRAKET_GATE_NAME_TO_QISKIT_GATE[gate_name]
        if params:
            gate.params = [float(param) if isinstance(param, Number) else param for param in params]
        if power:
            gate = gate.power(float(power))
        if ctrl_modifiers:
            gate = gate.control(
                len(ctrl_modifiers), ctrl_state=str("".join([str(i) for i in ctrl_modifiers]))
            )
        self._circuit.append(CircuitInstruction(gate, target))

    def handle_parameter_value(self, value: Number | Expr) -> Number | Parameter:
        if isinstance(value, Expr):
            evaluated_value = value.evalf()
            if isinstance(evaluated_value, Number):
                return evaluated_value
            return Parameter(evaluated_value)
        return value

    def add_measure(self, target: tuple[int], classical_targets: Iterable[int] = None):
        for iter, qubit in enumerate(target):
            index = classical_targets[iter] if classical_targets else iter
            self._circuit.measure(qubit, index)
