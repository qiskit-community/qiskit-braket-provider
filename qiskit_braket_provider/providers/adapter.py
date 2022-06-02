"""Util function for provider."""
from typing import Iterable, List, Dict, Callable
from typing import Tuple, Union, Optional

from braket.aws import AwsDevice
from braket.circuits import Instruction, Circuit, result_types
from braket.circuits import gates
from braket.device_schema import (
    JaqcdDeviceActionProperties,
    GateModelQpuParadigmProperties,
    DeviceActionType,
)
from braket.device_schema.ionq import IonqDeviceCapabilities
from braket.device_schema.oqc import OqcDeviceCapabilities
from braket.device_schema.rigetti import (
    RigettiDeviceCapabilities,
)
from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorParadigmProperties,
)
from braket.devices import LocalSimulator
from numpy import pi
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction as QiskitInstruction, Parameter, Measure
from qiskit.circuit.library import (
    HGate,
    CXGate,
    CSwapGate,
    CYGate,
    CZGate,
    IGate,
    RXGate,
    RYGate,
    RZGate,
    SGate,
    SdgGate,
    SwapGate,
    TGate,
    TdgGate,
    XGate,
    YGate,
    ZGate,
    RZZGate,
    RXXGate,
    RYYGate,
    SXGate,
    PhaseGate,
    SXdgGate,
    CPhaseGate,
    CCXGate,
    U1Gate,
    U2Gate,
    U3Gate,
)
from qiskit.transpiler import Target, InstructionProperties

from qiskit_braket_provider.exception import QiskitBraketException

qiskit_to_braket_gate_names_mapping = {
    "u1": "u1",
    "u2": "u2",
    "u3": "u3",
    "p": "phaseshift",
    "cx": "cnot",
    "x": "x",
    "y": "y",
    "z": "z",
    "t": "t",
    "tdg": "ti",
    "s": "s",
    "sdg": "si",
    "sx": "v",
    "sxdg": "vi",
    "swap": "swap",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "rzz": "zz",
    "id": "i",
    "h": "h",
    "cy": "cy",
    "cz": "cz",
    "ccx": "ccnot",
    "cswap": "cswap",
    "cp": "cphaseshift",
    "rxx": "xx",
    "ryy": "yy",
}


qiskit_gate_names_to_braket_gates: Dict[str, Callable] = {
    "u1": lambda lam: [gates.Rz(lam)],
    "u2": lambda phi, lam: [gates.Rz(lam), gates.Ry(pi / 2), gates.Rz(phi)],
    "u3": lambda theta, phi, lam: [
        gates.Rz(lam),
        gates.Rx(pi / 2),
        gates.Rz(theta),
        gates.Rx(-pi / 2),
        gates.Rz(phi),
    ],
    "p": lambda angle: [gates.PhaseShift(angle)],
    "cp": lambda angle: [gates.CPhaseShift(angle)],
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
    "rx": lambda angle: [gates.Rx(angle)],
    "ry": lambda angle: [gates.Ry(angle)],
    "rz": lambda angle: [gates.Rz(angle)],
    "rzz": lambda angle: [gates.ZZ(angle)],
    "id": lambda: [gates.I()],
    "h": lambda: [gates.H()],
    "cy": lambda: [gates.CY()],
    "cz": lambda: [gates.CZ()],
    "ccx": lambda: [gates.CCNot()],
    "cswap": lambda: [gates.CSwap()],
    "rxx": lambda angle: [gates.XX(angle)],
    "ryy": lambda angle: [gates.YY(angle)],
}


qiskit_gate_name_to_braket_gate_mapping: Dict[str, Optional[QiskitInstruction]] = {
    "u1": U1Gate(Parameter("theta")),
    "u2": U2Gate(Parameter("theta"), Parameter("lam")),
    "u3": U3Gate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
    "h": HGate(),
    "ccnot": CCXGate(),
    "cnot": CXGate(),
    "cphaseshift": CPhaseGate(Parameter("theta")),
    "cswap": CSwapGate(),
    "cy": CYGate(),
    "cz": CZGate(),
    "i": IGate(),
    "phaseshift": PhaseGate(Parameter("theta")),
    "rx": RXGate(Parameter("theta")),
    "ry": RYGate(Parameter("theta")),
    "rz": RZGate(Parameter("phi")),
    "s": SGate(),
    "si": SdgGate(),
    "swap": SwapGate(),
    "t": TGate(),
    "ti": TdgGate(),
    "v": SXGate(),
    "vi": SXdgGate(),
    "x": XGate(),
    "xx": RXXGate(Parameter("theta")),
    "y": YGate(),
    "yy": RYYGate(Parameter("theta")),
    "z": ZGate(),
    "zz": RZZGate(Parameter("theta")),
}


def _op_to_instruction(operation: str) -> Optional[QiskitInstruction]:
    """Converts Braket operation to Qiskit Instruction.

    Args:
        operation: operation

    Returns:
        Circuit Instruction
    """
    operation = operation.lower()
    return qiskit_gate_name_to_braket_gate_mapping.get(operation, None)


def local_simulator_to_target(simulator: LocalSimulator) -> Target:
    """Converts properties of LocalSimulator into Qiskit Target object.

    Args:
        simulator: AWS LocalSimulator

    Returns:
        target for Qiskit backend
    """
    target = Target()
    target.add_instruction(Measure())

    instructions = [
        inst
        for inst in qiskit_gate_name_to_braket_gate_mapping.values()
        if inst is not None
    ]
    properties = simulator.properties
    paradigm: GateModelSimulatorParadigmProperties = properties.paradigm
    for instruction in instructions:
        instruction_props: Optional[
            Dict[Union[Tuple[int], Tuple[int, int]], Optional[InstructionProperties]]
        ] = {}

        for src in range(paradigm.qubitCount):
            for dst in range(paradigm.qubitCount):
                if src != dst:
                    instruction_props[(src, dst)] = None
                    instruction_props[(dst, src)] = None
        target.add_instruction(instruction, instruction_props)

    return target


def aws_device_to_target(device: AwsDevice) -> Target:
    """Converts properties of Braket device into Qiskit Target object.

    Args:
        device: AWS Braket device

    Returns:
        target for Qiskit backend
    """
    # building target
    target = Target(description=f"Target for AWS Device: {device.name}")
    target.add_instruction(Measure())

    properties = device.properties
    # gate model devices
    if isinstance(
        properties,
        (IonqDeviceCapabilities, RigettiDeviceCapabilities, OqcDeviceCapabilities),
    ):
        action_properties: JaqcdDeviceActionProperties = properties.action.get(
            DeviceActionType.JAQCD
        )
        paradigm: GateModelQpuParadigmProperties = properties.paradigm
        connectivity = paradigm.connectivity
        instructions: List[QiskitInstruction] = []
        for operation in action_properties.supportedOperations:
            instruction = _op_to_instruction(operation)
            if instruction is not None:
                # TODO: remove when target will be supporting > 2 qubit gates  # pylint:disable=fixme
                if instruction.num_qubits <= 2:
                    instructions.append(instruction)

        for instruction in instructions:
            instruction_props: Optional[
                Dict[
                    Union[Tuple[int], Tuple[int, int]], Optional[InstructionProperties]
                ]
            ] = {}
            # adding 1 qubit instructions
            if instruction.num_qubits == 1:
                for i in range(paradigm.qubitCount):
                    instruction_props[(i,)] = None
            # adding 2 qubit instructions
            elif instruction.num_qubits == 2:
                # building coupling map for fully connected device
                if connectivity.fullyConnected:
                    for src in range(paradigm.qubitCount):
                        for dst in range(paradigm.qubitCount):
                            if src != dst:
                                instruction_props[(src, dst)] = None
                                instruction_props[(dst, src)] = None
                # building coupling map for device with connectivity graph
                else:
                    for src, connections in connectivity.connectivityGraph.items():
                        for dst in connections:
                            instruction_props[(int(src), int(dst))] = None
            # for more than 2 qubits
            else:
                instruction_props = None

            target.add_instruction(instruction, instruction_props)

    # gate model simulators
    elif isinstance(properties, GateModelSimulatorDeviceCapabilities):
        simulator_action_properties: JaqcdDeviceActionProperties = (
            properties.action.get(DeviceActionType.JAQCD)
        )
        simulator_paradigm: GateModelSimulatorParadigmProperties = properties.paradigm
        instructions = []
        for operation in simulator_action_properties.supportedOperations:
            instruction = _op_to_instruction(operation)
            if instruction is not None:
                # TODO: remove when target will be supporting > 2 qubit gates  # pylint:disable=fixme
                if instruction.num_qubits <= 2:
                    instructions.append(instruction)
        for instruction in instructions:
            simulator_instruction_props: Optional[
                Dict[
                    Union[Tuple[int], Tuple[int, int]], Optional[InstructionProperties]
                ]
            ] = {}
            # adding 1 qubit instructions
            if instruction.num_qubits == 1:
                for i in range(simulator_paradigm.qubitCount):
                    simulator_instruction_props[(i,)] = None
            # adding 2 qubit instructions
            elif instruction.num_qubits == 2:
                # building coupling map for fully connected device
                for src in range(simulator_paradigm.qubitCount):
                    for dst in range(simulator_paradigm.qubitCount):
                        if src != dst:
                            simulator_instruction_props[(src, dst)] = None
                            simulator_instruction_props[(dst, src)] = None
            target.add_instruction(instruction, simulator_instruction_props)

    else:
        raise QiskitBraketException(
            f"Cannot convert to target. "
            f"{properties.__class__} device capabilities are not supported yet."
        )

    return target


def convert_qiskit_to_braket_circuit(circuit: QuantumCircuit) -> Circuit:
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

            for gate in qiskit_gate_names_to_braket_gates[name](*params):
                instruction = Instruction(
                    operator=gate, target=[i.index for i in qiskit_gates[1]]
                )
                quantum_circuit += instruction
    return quantum_circuit


def convert_qiskit_to_braket_circuits(
    circuits: List[QuantumCircuit],
) -> Iterable[Circuit]:
    """Converts all Qiskit circuits to Braket circuits.
     Args:
            circuits (List(QuantumCircuit)): Qiskit Quantum Cricuit

    Returns:
        Circuit (Iterable[Circuit]): Braket circuit
    """
    for circuit in circuits:
        yield convert_qiskit_to_braket_circuit(circuit)
