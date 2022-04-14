"""Util function for provider."""
from typing import Dict, Tuple, Union, Optional

from braket.aws import AwsDevice
from braket.device_schema import (
    JaqcdDeviceActionProperties,
    GateModelQpuParadigmProperties,
    DeviceActionType,
)
from braket.device_schema.dwave import DwaveDeviceCapabilities
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
from qiskit.circuit import Instruction, Parameter
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
)
from qiskit.transpiler import Target, InstructionProperties

from qiskit_braket_plugin.exception import QiskitBraketException


# TODO: switch to converter  # pylint: disable=fixme
op_to_gate_mapping: Dict[str, Optional[Instruction]] = {
    "h": HGate(),
    "ccnot": None,
    "cnot": CXGate(),
    "cphaseshift": CPhaseGate(Parameter("theta")),
    "cphaseshift00": None,
    "cphaseshift01": None,
    "cphaseshift10": None,
    "cswap": CSwapGate(),
    "cy": CYGate(),
    "cz": CZGate(),
    "i": IGate(),
    "iswap": None,
    "pswap": None,
    "phaseshift": PhaseGate(Parameter("theta")),
    "rx": RXGate(Parameter("theta")),
    "ry": RYGate(Parameter("theta")),
    "rz": RZGate(Parameter("phi")),
    "s": SGate(),
    "si": SdgGate(),
    "swap": SwapGate(),
    "t": TGate(),
    "ti": TdgGate(),
    "unitary": None,
    "v": SXGate(),
    "vi": SXdgGate(),
    "x": XGate(),
    "xx": RXXGate(Parameter("theta")),
    "xy": None,
    "y": YGate(),
    "yy": RYYGate(Parameter("theta")),
    "z": ZGate(),
    "zz": RZZGate(Parameter("theta")),
    "start_verbatim_box": None,
    "end_verbatim_box": None,
}


def _op_to_instruction(operation: str) -> Optional[Instruction]:
    """Converts Braket operation to Qiskit Instruction.

    Args:
        operation: operation

    Returns:
        Circuit Instruction
    """
    operation = operation.lower()
    return op_to_gate_mapping.get(operation, None)


def local_simulator_to_target(simulator: LocalSimulator) -> Target:
    """Converts properties of LocalSimulator into Qiskit Target object.

    Args:
        simulator: AWS LocalSimulator

    Returns:
        target for Qiskit backend
    """
    target = Target()
    instructions = [inst for inst in op_to_gate_mapping.values() if inst is not None]
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
        instructions = []
        for operation in action_properties.supportedOperations:
            instruction = _op_to_instruction(operation)
            if instruction is not None:
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

    # annealing devices
    elif isinstance(properties, DwaveDeviceCapabilities):
        raise NotImplementedError("Dwave devices are not supported yet.")
    else:
        raise QiskitBraketException(
            f"Cannot convert to target. "
            f"{properties.__class__} device capabilities are not supported yet."
        )

    return target
