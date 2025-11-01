"""Util function for provider."""

import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from math import inf, pi
from numbers import Number

import numpy as np
import qiskit.circuit.library as qiskit_gates
import qiskit.quantum_info as qiskit_qi
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import (
    CircuitInstruction,
    ControlledGate,
    Gate,
    Measure,
    Parameter,
    ParameterExpression,
    ParameterVectorElement,
)
from qiskit.circuit import Instruction as QiskitInstruction
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import (
    InstructionProperties,
    PassManager,
    QubitProperties,
    Target,
    TransformationPass,
)
from qiskit_ionq import add_equivalences, ionq_gates
from sympy import Add, Expr, Mul, Pow, Symbol

import braket.circuits.gates as braket_gates
import braket.circuits.noises as braket_noises
from braket import experimental_capabilities as braket_expcaps
from braket.aws import AwsDevice, AwsDeviceType
from braket.circuits import Circuit, Instruction, measure
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.openqasm.program_context import AbstractProgramContext
from braket.device_schema import (
    DeviceActionType,
    DeviceCapabilities,
    OpenQASMDeviceActionProperties,
)
from braket.device_schema.ionq import IonqDeviceCapabilities
from braket.device_schema.rigetti import (
    RigettiDeviceCapabilities,
    RigettiDeviceCapabilitiesV2,
)
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.device_schema.standardized_gate_model_qpu_device_properties_v1 import (
    StandardizedGateModelQpuDeviceProperties as StandardizedPropertiesV1,
)
from braket.device_schema.standardized_gate_model_qpu_device_properties_v2 import (
    StandardizedGateModelQpuDeviceProperties as StandardizedPropertiesV2,
)
from braket.devices import Device, LocalSimulator
from braket.ir.openqasm import Program
from braket.ir.openqasm.modifiers import Control
from braket.parametric import FreeParameter, FreeParameterExpression, Parameterizable
from qiskit_braket_provider.exception import QiskitBraketException
from qiskit_braket_provider.providers import braket_instructions

add_equivalences()

_BRAKET_TO_QISKIT_NAMES = {
    "u": "u",
    "phaseshift": "p",
    "cnot": "cx",
    "x": "x",
    "y": "y",
    "z": "z",
    "t": "t",
    "ti": "tdg",
    "s": "s",
    "si": "sdg",
    "v": "sx",
    "vi": "sxdg",
    "swap": "swap",
    "iswap": "iswap",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "xx": "rxx",
    "yy": "ryy",
    "zz": "rzz",
    "i": "id",
    "h": "h",
    "cy": "cy",
    "cz": "cz",
    "ccnot": "ccx",
    "cswap": "cswap",
    "cphaseshift": "cp",
    "ecr": "ecr",
    "prx": "r",
    "gpi": "gpi",
    "gpi2": "gpi2",
    "ms": "ms",
    "gphase": "global_phase",
    "unitary": "unitary",
    "kraus": "kraus",
}

_CONTROLLED_GATES_BY_QUBIT_COUNT = {
    1: {
        "ch": "h",
        "cs": "s",
        "csdg": "sdg",
        "csx": "sx",
        "crx": "rx",
        "cry": "ry",
        "crz": "rz",
        "ccz": "cz",
    },
    3: {"c3sx": "sx"},
    inf: {"mcx": "cx"},
}

_ADDITIONAL_U_GATES = {"u1", "u2", "u3"}

_EPS = 1e-10  # global variable used to chop very small numbers to zero

_QISKIT_GATE_NAME_TO_BRAKET_GATE: dict[str, Callable] = {
    "u1": lambda lam: [braket_gates.U(0, 0, lam)],
    "u2": lambda phi, lam: [braket_gates.U(pi / 2, phi, lam)],
    "u3": lambda theta, phi, lam: [braket_gates.U(theta, phi, lam)],
    "u": lambda theta, phi, lam: [braket_gates.U(theta, phi, lam)],
    "p": lambda angle: [braket_gates.PhaseShift(angle)],
    "cp": lambda angle: [braket_gates.CPhaseShift(angle)],
    "cx": lambda: [braket_gates.CNot()],
    "x": lambda: [braket_gates.X()],
    "y": lambda: [braket_gates.Y()],
    "z": lambda: [braket_gates.Z()],
    "t": lambda: [braket_gates.T()],
    "tdg": lambda: [braket_gates.Ti()],
    "s": lambda: [braket_gates.S()],
    "sdg": lambda: [braket_gates.Si()],
    "sx": lambda: [braket_gates.V()],
    "sxdg": lambda: [braket_gates.Vi()],
    "swap": lambda: [braket_gates.Swap()],
    "rx": lambda angle: [braket_gates.Rx(angle)],
    "ry": lambda angle: [braket_gates.Ry(angle)],
    "rz": lambda angle: [braket_gates.Rz(angle)],
    "rzz": lambda angle: [braket_gates.ZZ(angle)],
    "id": lambda: [braket_gates.I()],
    "h": lambda: [braket_gates.H()],
    "cy": lambda: [braket_gates.CY()],
    "cz": lambda: [braket_gates.CZ()],
    "ccx": lambda: [braket_gates.CCNot()],
    "cswap": lambda: [braket_gates.CSwap()],
    "rxx": lambda angle: [braket_gates.XX(angle)],
    "ryy": lambda angle: [braket_gates.YY(angle)],
    "ecr": lambda: [braket_gates.ECR()],
    "iswap": lambda: [braket_gates.ISwap()],
    "r": lambda angle_1, angle_2: [braket_gates.PRx(angle_1, angle_2)],
    # IonQ gates
    "gpi": lambda turns: [braket_gates.GPi(2 * pi * turns)],
    "gpi2": lambda turns: [braket_gates.GPi2(2 * pi * turns)],
    "ms": lambda turns_1, turns_2, turns_3: [
        braket_gates.MS(2 * pi * turns_1, 2 * pi * turns_2, 2 * pi * turns_3)
    ],
    "zz": lambda angle: [braket_gates.ZZ(2 * pi * angle)],
    # Global phase
    "global_phase": lambda phase: [braket_gates.GPhase(phase)],
    "unitary": lambda operators: [braket_gates.Unitary(operators[0])],
    "kraus": lambda operators: [braket_noises.Kraus(operators)],
    "CCPRx": lambda angle_1, angle_2, feedback_key: [
        braket_expcaps.iqm.classical_control.CCPRx(angle_1, angle_2, feedback_key)
    ],
    "MeasureFF": lambda feedback_key: [
        braket_expcaps.iqm.classical_control.MeasureFF(feedback_key)
    ],
}

_QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES: dict[str, Callable] = {
    controlled_gate: _QISKIT_GATE_NAME_TO_BRAKET_GATE[base_gate]
    for gate_map in _CONTROLLED_GATES_BY_QUBIT_COUNT.values()
    for controlled_gate, base_gate in gate_map.items()
}

_STANDARD_GATE_NAME_MAPPING = get_standard_gate_name_mapping()

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

_BRAKET_SUPPORTED_NOISES = [
    "kraus",
    "bitflip",
    "depolarizing",
    "amplitudedamping",
    "generalizedamplitudedamping",
    "phasedamping",
    "phaseflip",
    "paulichannel",
    "twoqubitdepolarizing",
    "twoqubitdephasing",
    # "twoqubitpaulichannel" no to_open qasm support yet
]

_TRANSPILER_GATE_SUBSTITUTIONS: dict[tuple[str, tuple[float | str, ...]], Gate] = {
    ("rx", (pi,)): qiskit_gates.XGate(),
    ("rx", (-pi,)): qiskit_gates.XGate(),
    ("rx", (pi / 2,)): qiskit_gates.SXGate(),
    ("rx", (-pi / 2,)): qiskit_gates.SXdgGate(),
}

_Translatable = QuantumCircuit | Circuit | Program | str


class _SubstitutedTarget(Target):
    def __new__(
        cls,
        description: str,
        num_qubits: int,
        qubit_properties: list[QubitProperties],
        gate_substitutions: dict[str, Gate],
    ):
        out = super(_SubstitutedTarget, cls).__new__(
            cls,
            description=description,
            num_qubits=num_qubits,
            qubit_properties=qubit_properties,
        )
        out._gate_substitutions = gate_substitutions
        out._pass_manager = PassManager([_SubstituteGates(gate_substitutions)])
        return out

    def __init__(
        self,
        description: str,
        num_qubits: int,
        qubit_properties: list[QubitProperties],
        gate_substitutions: dict[str, Gate],
    ):
        super().__init__(
            description=description,
            num_qubits=num_qubits,
            qubit_properties=qubit_properties,
        )
        self._gate_substitutions = gate_substitutions
        self._pass_manager = PassManager([_SubstituteGates(gate_substitutions)])

    def __getnewargs__(self):
        return self.description, self.num_qubits, self.qubit_properties, self._gate_substitutions


class _SubstituteGates(TransformationPass):
    def __init__(self, gate_substitutions: dict[str, Gate]):
        super().__init__()
        self._gate_substitutions = gate_substitutions

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.op_nodes():
            op_name = node.op.name
            if op_name in self._gate_substitutions:
                dag.substitute_node(node, self._gate_substitutions[op_name])
        return dag


class _QiskitProgramContext(AbstractProgramContext):
    def __init__(self):
        super().__init__()
        self._circuit = QuantumCircuit()

    @property
    def circuit(self):
        return self._circuit

    def add_qubits(self, name: str, num_qubits: int | None = 1) -> None:
        super().add_qubits(name, num_qubits)
        self._circuit.add_register(num_qubits, num_qubits)

    def is_builtin_gate(self, name: str) -> bool:
        return name in _BRAKET_GATE_NAME_TO_QISKIT_GATE

    def add_phase_instruction(self, target, phase_value):
        self._circuit.global_phase += phase_value

    def add_gate_instruction(
        self, gate_name: str, target: tuple[int, ...], params, ctrl_modifiers: list[int], power: int
    ):
        gate: Gate = _BRAKET_GATE_NAME_TO_QISKIT_GATE[gate_name].copy()
        params = (
            [float(param) if isinstance(param, Number) else param for param in params]
            if params is not None
            else []
        )
        if params:
            gate.params = params
        gate = gate.power(float(power))
        if ctrl_modifiers:
            gate = gate.control(
                len(ctrl_modifiers), ctrl_state=str("".join([str(i) for i in ctrl_modifiers]))
            )
        self._circuit.append(CircuitInstruction(gate, target))

    def handle_parameter_value(self, value: Number | Expr) -> Number | Parameter:
        return _sympy_to_qiskit(value) if isinstance(value, Expr) else value

    def add_measure(self, target: tuple[int], classical_targets: Iterable[int] = None):
        for iter, qubit in enumerate(target):
            index = classical_targets[iter] if classical_targets else iter
            self._circuit.measure(qubit, index)


def native_gate_connectivity(properties: DeviceCapabilities) -> list[list[int]] | None:
    """Returns the connectivity natively supported by a Braket device from its properties

    Args:
        properties (DeviceCapabilities): The device properties of the Braket device.

    Returns:
        list[list[int]] | None: A list of connected qubit pairs or `None` if the device is fully
            connected.
    """
    device_connectivity = properties.paradigm.connectivity
    connectivity = (
        [
            [int(x), int(y)]
            for x, neighborhood in device_connectivity.connectivityGraph.items()
            for y in neighborhood
        ]
        if not device_connectivity.fullyConnected
        else None
    )
    return connectivity


def native_gate_set(properties: DeviceCapabilities) -> set[str]:
    """Returns the gate set natively supported by a Braket device from its properties

    Args:
        properties (DeviceCapabilities): The device properties of the Braket device.

    Returns:
        set[str]: The names of qiskit gates natively supported by the Braket device.
    """
    native_list = properties.paradigm.nativeGateSet
    return {
        _BRAKET_TO_QISKIT_NAMES[op.lower()]
        for op in native_list
        if op.lower() in _BRAKET_TO_QISKIT_NAMES
    }


def native_angle_restrictions(
    properties: DeviceCapabilities,
) -> dict[str, dict[int, set[float] | tuple[float, float]]]:
    """Returns angle restrictions for gates natively supported by a Braket device.

    The returned mapping specifies, for each gate name, constraints on the
    gate parameters indexed by their position. The constraint can either be a
    set of allowed angles or a tuple representing an inclusive ``(min, max)``
    range. Angle units are in radians.

    Args:
        properties (DeviceCapabilities): The device properties of the Braket device.

    Returns:
        dict[str, dict[int, set[float] | tuple[float, float]]]: Mapping
        of gate names to parameter index restrictions.
    """

    if isinstance(properties, (RigettiDeviceCapabilities, RigettiDeviceCapabilitiesV2)):
        return {"rx": {0: {pi, -pi, pi / 2, -pi / 2}}}
    if isinstance(properties, IonqDeviceCapabilities):
        return {"ms": {2: (0.0, 0.25)}}
    return {}


def gateset_from_properties(properties: OpenQASMDeviceActionProperties) -> set[str]:
    """Returns the gateset supported by a Braket device with the given properties

    Args:
        properties (OpenQASMDeviceActionProperties): The action properties of the Braket device.

    Returns:
        set[str]: The names of the gates supported by the device
    """
    gateset = {
        _BRAKET_TO_QISKIT_NAMES[op.lower()]
        for op in properties.supportedOperations
        if op.lower() in _BRAKET_TO_QISKIT_NAMES
    }
    if "u" in gateset:
        gateset.update(_ADDITIONAL_U_GATES)
    max_control = 0
    for modifier in properties.supportedModifiers:
        if isinstance(modifier, Control):
            max_control = modifier.max_qubits
            break
    return gateset.union(_get_controlled_gateset(gateset, max_control))


def _get_controlled_gateset(base_gateset: set[str], max_qubits: int | None = None) -> set[str]:
    """Returns the Qiskit gates expressible as controlled versions of existing Braket gates

    This set can be filtered by the maximum number of control qubits.

    Args:
        base_gateset (set[str]): The base (without control modifiers) gates supported
        max_qubits (int | None): The maximum number of control qubits that can be used to express
            the Qiskit gate as a controlled Braket gate. If `None`, then there is no limit to the
            number of control qubits. Default: `None`.

    Returns:
        set[str]: The names of the controlled gates.
    """
    max_control = max_qubits if max_qubits is not None else inf
    return {
        controlled_gate
        for control_count, gate_map in _CONTROLLED_GATES_BY_QUBIT_COUNT.items()
        for controlled_gate, base_gate in gate_map.items()
        if control_count <= max_control and base_gate in base_gateset
    }


def local_simulator_to_target(simulator: LocalSimulator) -> Target:
    """Converts properties of a Braket LocalSimulator into a Qiskit Target object.

    Args:
        simulator (LocalSimulator): Amazon Braket LocalSimulator

    Returns:
        Target: Target for Qiskit backend
    """
    return _simulator_target(
        simulator, f"Target for Amazon Braket local simulator: {simulator.name}"
    )


def aws_device_to_target(device: AwsDevice) -> Target:
    """Converts properties of Braket AwsDevice into a Qiskit Target object.

    Args:
        device (AwsDevice): Amazon Braket AwsDevice

    Returns:
        Target: Target for Qiskit backend
    """
    match device.type:
        case AwsDeviceType.QPU:
            return _qpu_target(device, f"Target for Amazon Braket QPU: {device.name}")
        case AwsDeviceType.SIMULATOR:
            return _simulator_target(device, f"Target for Amazon Braket simulator: {device.name}")
    raise QiskitBraketException(
        "Cannot convert to target. "
        f"{device.properties.__class__} device capabilities are not supported."
    )


def _simulator_target(device: Device, description: str):
    properties: GateModelSimulatorDeviceCapabilities = device.properties
    target = Target(description=description, num_qubits=properties.paradigm.qubitCount)
    action = (
        properties.action.get(DeviceActionType.OPENQASM)
        if properties.action.get(DeviceActionType.OPENQASM)
        else properties.action.get(DeviceActionType.JAQCD)
    )
    for operation in action.supportedOperations:
        instruction = _BRAKET_GATE_NAME_TO_QISKIT_GATE.get(operation.lower())
        if instruction:
            target.add_instruction(instruction, name=_BRAKET_TO_QISKIT_NAMES[operation.lower()])
    if isinstance(action, OpenQASMDeviceActionProperties):
        max_control = 0
        for modifier in action.supportedModifiers:
            if isinstance(modifier, Control):
                max_control = modifier.max_qubits
                break
        for gate in _get_controlled_gateset(target.keys(), max_control):
            if gate in _STANDARD_GATE_NAME_MAPPING:
                target.add_instruction(_STANDARD_GATE_NAME_MAPPING[gate])
    target.add_instruction(Measure())
    return target


def _qpu_target(device: AwsDevice, description: str):
    properties: DeviceCapabilities = device.properties
    topology = device.topology_graph
    standardized = properties.standardized
    indices = {q: i for i, q in enumerate(sorted(topology.nodes))}

    qubit_properties = []
    instruction_props_measurement = {}
    instruction_props_1q = {}
    instruction_props_2q = defaultdict(dict)
    default_instruction_props = InstructionProperties(error=0)
    # TODO: Support V3 standardized properties
    if isinstance(standardized, (StandardizedPropertiesV1, StandardizedPropertiesV2)):
        props_1q = standardized.oneQubitProperties
        props_2q = standardized.twoQubitProperties
        for q in sorted(int(q) for q in props_1q):
            props = props_1q[str(q)]
            key = (indices[q],)
            for fidelity in props.oneQubitFidelity:
                match fidelity.fidelityType.name.lower():
                    case "readout":
                        instruction_props_measurement[key] = InstructionProperties(
                            # Use highest known error rate
                            error=max(
                                1 - fidelity.fidelity,
                                instruction_props_measurement.get(
                                    key, default_instruction_props
                                ).error,
                            )
                        )
                    case name if "readout_error" not in name:
                        instruction_props_1q[key] = InstructionProperties(
                            error=max(
                                1 - fidelity.fidelity,
                                instruction_props_1q.get(key, default_instruction_props).error,
                            )
                        )
            qubit_properties.append(QubitProperties(t1=props.T1.value, t2=props.T2.value))
        for k, props in props_2q.items():
            for fidelity in props.twoQubitGateFidelity:
                gate_name = _BRAKET_TO_QISKIT_NAMES[fidelity.gateName.lower()]
                edge = tuple(indices[int(q)] for q in k.split("-"))
                instruction_props_2q[gate_name][edge] = InstructionProperties(
                    error=max(
                        1 - fidelity.fidelity,
                        instruction_props_2q[gate_name].get(edge, default_instruction_props).error,
                    )
                )

    default_props_1q = {(i,): None for i in indices.values()}
    default_props_2q = {(indices[q0], indices[q1]): None for q0, q1 in topology.edges}
    if not instruction_props_measurement:
        instruction_props_measurement.update(default_props_1q)
    if not instruction_props_1q:
        instruction_props_1q.update(default_props_1q)

    parameter_restrictions = _get_parameter_restrictions(device, indices)
    target = _SubstitutedTarget(
        description=description,
        num_qubits=len(qubit_properties or indices),
        qubit_properties=qubit_properties or None,
        gate_substitutions=_get_gate_substitutions(parameter_restrictions),
    )
    for operation in properties.paradigm.nativeGateSet:
        braket_name = operation.lower()
        if instruction := _BRAKET_GATE_NAME_TO_QISKIT_GATE.get(braket_name):
            gate_name = instruction.name
            match num_qubits := instruction.num_qubits:
                case 1:
                    _add_instruction_to_qpu_target(
                        target,
                        instruction,
                        parameter_restrictions[braket_name],
                        instruction_props_1q,
                    )
                case 2:
                    _add_instruction_to_qpu_target(
                        target,
                        instruction,
                        parameter_restrictions[braket_name],
                        instruction_props_2q.get(gate_name, default_props_2q),
                    )
                case _:
                    warnings.warn(
                        f"Instruction {gate_name} has {num_qubits} qubits and cannot be added to target"
                    )

    # Add measurement if not already added
    if "measure" not in target:
        target.add_instruction(Measure(), instruction_props_measurement)
    return target


def _get_parameter_restrictions(
    device: AwsDevice, qubit_indices: dict[int, int]
) -> dict[str, dict[tuple[float | str, ...], set[tuple[int, ...]]]]:
    cal = device.gate_calibrations
    parameter_restrictions = defaultdict(lambda: defaultdict(set))
    for gate, target in cal.pulse_sequences if cal else {}:
        qubits = tuple(qubit_indices[q] for q in target)
        if isinstance(gate, Parameterizable):
            param_key = tuple(
                param.name if isinstance(param, FreeParameter) else param
                for param in gate.parameters
            )
            parameter_restrictions[gate.name.lower()][tuple(param_key)].add(qubits)
    return parameter_restrictions


def _get_gate_substitutions(
    parameter_restrictions: dict[str, dict[tuple[float | str, ...], set[tuple[int, ...]]]],
) -> dict[str, Gate]:
    substitutions = {}
    for gate_name, restrictions in parameter_restrictions.items():
        for restriction in restrictions:
            if (gate_name, restriction) in _TRANSPILER_GATE_SUBSTITUTIONS:
                if (
                    gate := _TRANSPILER_GATE_SUBSTITUTIONS[gate_name, restriction].name
                ) not in substitutions:
                    substitution = _BRAKET_GATE_NAME_TO_QISKIT_GATE[gate_name].copy()
                    substitution.params = [
                        Parameter(param) if isinstance(param, str) else param
                        for param in restriction
                    ]
                    substitutions[gate] = substitution
    return substitutions


def _add_instruction_to_qpu_target(
    target: Target,
    instruction: QiskitInstruction,
    gate_parameters: dict[tuple[float | str, ...], set[tuple[int, ...]]],
    gate_props: dict[tuple[int, ...], InstructionProperties],
) -> None:
    if gate_parameters:
        for params, qubits in gate_parameters.items():
            props = {q: props for q, props in gate_props.items() if q in qubits}
            instruction_copy = instruction.copy()
            instruction_copy.params = [
                Parameter(param) if isinstance(param, str) else param for param in params
            ]
            instruction_to_add = _TRANSPILER_GATE_SUBSTITUTIONS.get(
                (instruction.name, params), instruction_copy
            )
            instruction_name = instruction_to_add.name
            if instruction_name in target:
                for k in (instruction_target := target[instruction_name]).keys() - props.keys():
                    instruction_target.pop(k)
            else:
                target.add_instruction(instruction_to_add, props)
    else:
        target.add_instruction(instruction, gate_props)


def to_braket(
    circuit: _Translatable | Iterable[_Translatable],
    basis_gates: Iterable[str] | None = None,
    verbatim: bool = False,
    connectivity: list[list[int]] | None = None,
    angle_restrictions: dict[str, dict[int, set[float] | tuple[float, float]]] | None = None,
    *,
    target: Target | None = None,
    qubit_labels: Sequence[int] | None = None,
    optimization_level: int = 0,
    callback: Callable | None = None,
    num_processes: int | None = None,
    pass_manager: PassManager | None = None,
) -> Circuit | list[Circuit]:
    """Return a Braket quantum circuit from a Qiskit quantum circuit.

    Args:
        circuit (QuantumCircuit | Circuit | Program | str | Iterable): Qiskit or Braket circuit(s)
            or OpenQASM 3 program(s) to transpile and translate to Braket.
        basis_gates (Iterable[str] | None): The gateset to transpile to. Can only be provided
            if target is `None`. If `None` and target is `None`, the transpiler will use all gates
            defined in the Braket SDK. Default: `None`.
        verbatim (bool): Whether to translate the circuit without any modification, in other
            words without transpiling it. Default: False.
        connectivity (list[list[int]] | None): If provided, will transpile to a circuit
            with this connectivity. Default: `None`.
        angle_restrictions (dict[str, dict[int, set[float] | tuple[float, float]]] | None):
            Mapping of gate names to parameter angle constraints used to
            validate numeric parameters. Default: `None`.
        target (Target | None): A backend transpiler target. Can only be provided
            if basis_gates is `None`. Default: `None`.
        qubit_labels (Sequence[int] | None): A list of (not necessarily contiguous) indices of
            qubits in the underlying Amazon Braket device. If not supplied, then the indices are
            assumed to be contiguous.
        optimization_level (int): The optimization level to pass to `qiskit.transpile`. From Qiskit:
            0: no optimization (default) - basic translation, no optimization, trivial layout
            1: light optimization - routing + potential SaberSwap, some gate cancellation and 1Q gate folding
            2: medium optimization - better routing (noise aware) and commutative cancellation
            3: high optimization - gate resynthesis and unitary-breaking passes
        callback (Callable | None): A callback function that will be called after each transpiler
            pass execution. Default: `None`.
        num_processes (int | None): The maximum number of parallel transpilation processes for
            multiple circuits. Default: `None`.
        pass_manager (PassManager): `PassManager` to transpile the circuit; will raise an error if
            used in conjunction with a target, basis gates, or connectivity. Default: `None`.

    Returns:
        Circuit | list[Circuit]: Braket circuit or circuits
    """
    single_instance = isinstance(circuit, _Translatable) or not isinstance(circuit, Iterable)
    if single_instance:
        circuit = [circuit]
    circuit = [to_qiskit(c) if isinstance(c, (Circuit, Program, str)) else c for c in circuit]
    other_types = {type(c).__name__ for c in circuit if not isinstance(c, QuantumCircuit)}
    if other_types:
        raise TypeError(f"Expected only QuantumCircuits, got {other_types} instead.")
    loose_constraints = basis_gates or connectivity
    if pass_manager and (target or loose_constraints):
        raise ValueError(
            "Cannot specify target, basis gates, or connectivity alongside pass manager"
        )
    if loose_constraints and target:
        raise ValueError("Cannot specify basis gates or connectivity alongside target.")

    if pass_manager:
        circuit = pass_manager.run(circuit, callback=callback, num_processes=num_processes)
    elif not verbatim:
        # If basis_gates is not None, then target remains empty
        target = target if basis_gates or target else _default_target(circuit)
        if (
            target
            or connectivity
            or (
                basis_gates
                and not {gate.name for circ in circuit for gate, _, _ in circ.data}.issubset(
                    basis_gates
                )
            )
        ):
            circuit = transpile(
                circuit,
                basis_gates=basis_gates,
                coupling_map=connectivity,
                optimization_level=optimization_level,
                target=target,
                callback=callback,
                num_processes=num_processes,
            )
    if target and isinstance(target, _SubstitutedTarget) and target._gate_substitutions:
        circuit = target._pass_manager.run(circuit)
    translated = [
        _translate_to_braket(
            circ, target, qubit_labels, verbatim, basis_gates, angle_restrictions, pass_manager
        )
        for circ in circuit
    ]
    return translated[0] if single_instance else translated


def _translate_to_braket(
    circuit: _Translatable,
    target: Target | None,
    qubit_labels: Sequence[int] | None,
    verbatim: bool,
    basis_gates: Iterable[str] | None,
    angle_restrictions: dict[str, dict[int, set[float] | tuple[float, float]]] | None,
    pass_manager: PassManager | None,
) -> Circuit:
    # Verify that ParameterVector would not collide with scalar variables after renaming.
    _validate_name_conflicts(circuit.parameters)
    # Handle qiskit to braket conversion
    measured_qubits: dict[int, int] = {}
    braket_circuit = Circuit()
    qubit_labels = qubit_labels or _default_qubit_labels(circuit)
    for circuit_instruction in circuit.data:
        operation = circuit_instruction.operation
        qubits = circuit_instruction.qubits

        if getattr(operation, "condition", None):
            raise NotImplementedError(
                f"Conditional operations are not supported. Found conditional gate '{operation.name}'. "
                f"Only MeasureFF and CCPRx gates are supported in Braket."
            )

        match gate_name := operation.name:
            case "measure":
                qubit = qubits[0]  # qubit count = 1 for measure
                qubit_index = qubit_labels[circuit.find_bit(qubit).index]
                if qubit_index in measured_qubits.values():
                    raise ValueError(f"Cannot measure previously measured qubit {qubit_index}")
                clbit = circuit.find_bit(circuit_instruction.clbits[0]).index
                measured_qubits[clbit] = qubit_index
            case "barrier":
                warnings.warn("The Qiskit circuit contains barrier instructions that are ignored.")
            case "reset":
                raise NotImplementedError(
                    "reset operation not supported by qiskit to braket adapter"
                )
            case "unitary" | "kraus":
                params = _create_free_parameters(operation)
                qubit_indices = [qubit_labels[circuit.find_bit(qubit).index] for qubit in qubits][
                    ::-1
                ]  # reversal for little to big endian notation

                for gate in _QISKIT_GATE_NAME_TO_BRAKET_GATE[gate_name](params):
                    braket_circuit += Instruction(
                        operator=gate,
                        target=qubit_indices,
                    )
            case _:
                if (
                    isinstance(operation, ControlledGate)
                    and operation.ctrl_state != 2**operation.num_ctrl_qubits - 1
                ):
                    raise ValueError("Negative control is not supported")
                # Getting the index from the bit mapping
                qubit_indices = [qubit_labels[circuit.find_bit(qubit).index] for qubit in qubits]
                if intersection := set(measured_qubits.values()).intersection(qubit_indices):
                    raise ValueError(
                        f"Cannot apply operation {gate_name} to measured qubits {intersection}"
                    )
                params = _create_free_parameters(operation)
                # TODO: Use angle_bounds in Target.add_instruction instead of validating here
                _validate_angle_restrictions(gate_name, params, angle_restrictions)
                if gate_name in _QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES:
                    for gate in _QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES[gate_name](*params):
                        gate_qubit_count = gate.qubit_count
                        braket_circuit += Instruction(
                            operator=gate,
                            target=qubit_indices[-gate_qubit_count:],
                            control=qubit_indices[:-gate_qubit_count],
                        )
                else:
                    for gate in _QISKIT_GATE_NAME_TO_BRAKET_GATE[gate_name](*params):
                        braket_circuit += Instruction(
                            operator=gate,
                            target=qubit_indices,
                        )
    global_phase = circuit.global_phase
    if abs(global_phase) > _EPS:
        if (target and "global_phase" in target) or (basis_gates and "global_phase" in basis_gates):
            braket_circuit.gphase(global_phase)
        else:
            warnings.warn(
                f"Device does not support global phase; "
                f"global phase of {global_phase} will not be included in Braket circuit"
            )

    # QPU targets will have qubits/pairs specified for each instruction;
    # Targets whose values consist solely of {None: None} are either simulator or default targets
    if verbatim or (target and any(v != {None: None} for v in target.values())) or pass_manager:
        braket_circuit = Circuit(braket_circuit.result_types).add_verbatim_box(
            Circuit(braket_circuit.instructions)
        )

    for clbit in sorted(measured_qubits):
        braket_circuit.measure(measured_qubits[clbit])

    return braket_circuit


def _default_target(circuits: Iterable[QuantumCircuit]) -> Target:
    num_qubits = max(circuit.num_qubits for circuit in circuits)
    target = Target(num_qubits=num_qubits)
    for braket_name, instruction in _BRAKET_GATE_NAME_TO_QISKIT_GATE.items():
        if name := _BRAKET_TO_QISKIT_NAMES.get(braket_name.lower()):
            target.add_instruction(instruction, name=name)
    target.add_instruction(Measure())
    return target


def _default_qubit_labels(circuit: QuantumCircuit) -> tuple[int, ...]:
    bits = sorted(circuit.find_bit(q).index for q in circuit.qubits)
    return tuple(range(max(bits) + 1)) if bits else tuple()


def _create_free_parameters(operation):
    for i, param in enumerate(params := operation.params):
        match param:
            case ParameterVectorElement():
                renamed_param_name = _rename_param_vector_element(param)
                params[i] = FreeParameter(renamed_param_name)
            case Parameter(name=name):
                params[i] = FreeParameter(name)
            case ParameterExpression():
                renamed_param_name = _rename_param_vector_element(param)
                params[i] = FreeParameterExpression(renamed_param_name)
    return params


def _validate_angle_restrictions(
    gate_name: str,
    params: Iterable,
    angle_restrictions: dict[str, dict[int, set[float] | tuple[float, float]]] | None,
) -> None:
    """Validate gate parameter angles against a restriction map.

    Parameters that are ``FreeParameter`` or ``ParameterExpression`` instances
    are ignored. Numeric angles are validated against the entry in
    ``angle_restrictions`` for the ``gate_name``. Each restriction can be a set
    of discrete allowed values or a ``(min, max)`` tuple describing an inclusive
    range.
    """
    if not angle_restrictions or gate_name not in angle_restrictions:
        return
    restrictions = angle_restrictions[gate_name]
    params = list(params)
    for index, restriction in restrictions.items():
        if index >= len(params):
            continue
        param = params[index]
        if isinstance(
            param,
            (
                FreeParameter,
                FreeParameterExpression,
                ParameterExpression,
            ),
        ):
            continue
        angle = float(param)
        if isinstance(restriction, set):
            if not any(abs(angle - allowed) <= _EPS for allowed in restriction):
                raise ValueError(
                    f"Angle {angle} for {gate_name} parameter {index} is not supported"
                )
        else:
            min_angle, max_angle = restriction
            if angle < min_angle - _EPS or angle > max_angle + _EPS:
                raise ValueError(
                    f"Angle {angle} for {gate_name} parameter {index} "
                    f"not in range [{min_angle}, {max_angle}]"
                )


def _rename_param_vector_element(parameter):
    return str(parameter).replace("[", "_").replace("]", "")


def _validate_name_conflicts(parameters):
    renamed_parameters = {_rename_param_vector_element(param) for param in parameters}
    if len(renamed_parameters) != len(parameters):
        raise ValueError(
            "ParameterVector elements are renamed from v[i] to v_i, which resulted "
            "in a conflict with another parameter. Please rename your parameters."
        )


def to_qiskit(circuit: Circuit | Program | str, add_measurements: bool = True) -> QuantumCircuit:
    """Return a Qiskit quantum circuit from a Braket quantum circuit.

    Args:
        circuit (Circuit | Program | str): Braket quantum circuit or OpenQASM 3 program.
        add_measurements (bool): Whether to append measurements in the conversion

    Returns:
        QuantumCircuit: Qiskit quantum circuit
    """
    if isinstance(circuit, Program):
        return (
            Interpreter(_QiskitProgramContext()).run(circuit.source, inputs=circuit.inputs).circuit
        )
    if isinstance(circuit, str):
        return Interpreter(_QiskitProgramContext()).run(circuit).circuit
    if not isinstance(circuit, Circuit):
        raise TypeError(f"Expected a Circuit, got {type(circuit)} instead.")

    num_measurements = sum(
        isinstance(instr.operator, measure.Measure) for instr in circuit.instructions
    )
    qiskit_circuit = QuantumCircuit(circuit.qubit_count, num_measurements)
    qubit_map = {int(qubit): index for index, qubit in enumerate(sorted(circuit.qubits))}
    cbit = 0
    for instruction in circuit.instructions:
        operator = instruction.operator
        gate_name = operator.name.lower()
        if gate_name in _BRAKET_SUPPORTED_NOISES:
            gate = _create_qiskit_kraus(operator.to_matrix())
        elif gate_name == "unitary":
            gate = _create_qiskit_unitary(operator.to_matrix())
        else:
            gate = _create_qiskit_gate(
                gate_name, operator.parameters if isinstance(operator, Parameterizable) else []
            )
        if (power := instruction.power) != 1:
            gate = gate**power
        if control_qubits := instruction.control:
            ctrl_state = instruction.control_state.as_string[::-1]
            gate = gate.control(len(control_qubits), ctrl_state=ctrl_state)

        target = [qiskit_circuit.qubits[qubit_map[i]] for i in control_qubits]
        target += [qiskit_circuit.qubits[qubit_map[i]] for i in instruction.target]

        if gate_name == "measure":
            qiskit_circuit.append(gate, target, [cbit])
            cbit += 1
        else:
            qiskit_circuit.append(gate, target)
    if num_measurements == 0 and add_measurements:
        qiskit_circuit.measure_all()
    return qiskit_circuit


def _create_qiskit_unitary(matrix: np.ndarray):
    return qiskit_gates.UnitaryGate(_reverse_endianness(matrix))


def _create_qiskit_kraus(gate_params: list[np.ndarray]) -> Instruction:
    """create qiskit.quantum_info.Kraus from Braket Kraus operators and reorder axes"""
    for i, param in enumerate(gate_params):
        assert param.shape[0] == param.shape[1], "Kraus operators must be square matrices."
        gate_params[i] = _reverse_endianness(param)
    return qiskit_qi.Kraus(gate_params)


def _sympy_to_qiskit(expr: Mul | Add | Symbol | Pow) -> ParameterExpression | Parameter:
    """convert a sympy expression to qiskit Parameters recursively"""
    match expr:
        case Mul():
            return _sympy_to_qiskit(expr.args[0]) * _sympy_to_qiskit(expr.args[1])
        case Add():
            return _sympy_to_qiskit(expr.args[0]) + _sympy_to_qiskit(expr.args[1])
        case Symbol():
            return Parameter(expr.name)
        case Pow():
            return _sympy_to_qiskit(expr.args[0]) ** int(expr.args[1])
        case obj if getattr(obj, "is_real", False):
            return float(obj)
    raise TypeError(f"unrecognized parameter type in conversion: {type(expr)}")


def _reverse_endianness(matrix: np.ndarray):
    n_q = int(np.log2(matrix.shape[0]))
    # Convert multi-qubit Kraus from little to big endian notation
    return (
        np.transpose(
            matrix.reshape([2] * n_q * 2),
            list(range(0, n_q))[::-1] + list(range(n_q, 2 * n_q))[::-1],
        ).reshape((2**n_q, 2**n_q))
        if n_q > 1
        else matrix
    )


def _create_qiskit_gate(
    gate_name: str, gate_params: list[float | FreeParameterExpression]
) -> Instruction:
    gate_instance = _BRAKET_GATE_NAME_TO_QISKIT_GATE.get(gate_name)
    if not gate_instance:
        raise TypeError(f'Braket gate "{gate_name}" not supported in Qiskit')
    gate_cls = gate_instance.__class__
    new_gate_params = []
    for param_expression, value in zip(gate_instance.params, gate_params):
        # extract the coefficient in the templated gate
        param = list(param_expression.parameters)[0].sympify()
        coeff = float(param_expression.sympify().subs(param, 1))
        new_gate_params.append(
            _sympy_to_qiskit(coeff * value.expression)
            if isinstance(value, FreeParameterExpression)
            else coeff * value
        )
    return gate_cls(*new_gate_params)


def convert_qiskit_to_braket_circuit(circuit: QuantumCircuit) -> Circuit:
    """Return a Braket quantum circuit from a Qiskit quantum circuit.

    Args:
        circuit (QuantumCircuit): Qiskit Quantum Circuit

    Returns:
        Circuit: Braket circuit
    """
    warnings.warn(
        "convert_qiskit_to_braket_circuit() is deprecated and "
        "will be removed in a future release. "
        "Use to_braket() instead.",
        DeprecationWarning,
    )
    return to_braket(circuit)


def convert_qiskit_to_braket_circuits(
    circuits: list[QuantumCircuit],
) -> Iterable[Circuit]:
    """Converts all Qiskit circuits to Braket circuits.

    Args:
        circuits (List(QuantumCircuit)): Qiskit quantum circuit

    Returns:
        Iterable[Circuit]: Braket circuit
    """
    warnings.warn(
        "convert_qiskit_to_braket_circuits() is deprecated and "
        "will be removed in a future release. "
        "Use to_braket() instead.",
        DeprecationWarning,
    )
    for circuit in circuits:
        yield to_braket(circuit)
