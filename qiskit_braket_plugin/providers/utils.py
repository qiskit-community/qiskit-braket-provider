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
from braket.device_schema.rigetti import RigettiDeviceCapabilities
from qiskit.circuit import Instruction
from qiskit.circuit.library import HGate
from qiskit.transpiler import Target, InstructionProperties

from qiskit_braket_plugin.exception import QiskitBraketException


def _op_to_instruction(operation: str) -> Instruction:
    """Converts Braket operation to Qiskit Instruction.

    Args:
        operation: operation

    Returns:
        Circuit Instruction
    """
    # TODO: switch to converter  # pylint: disable=fixme
    op_to_gate_mapping = {"h": HGate()}
    operation = operation.lower()
    if operation not in op_to_gate_mapping:
        raise QiskitBraketException(f"Operation {operation} is not supported yet.")
    return op_to_gate_mapping[operation]


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
        instructions = [
            _op_to_instruction(op) for op in action_properties.supportedOperations
        ]

        for instruction in instructions:
            instruction_props: Dict[
                Union[Tuple[int], Tuple[int, int]], Optional[InstructionProperties]
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
                            instruction_props[(src, dst)] = None
            else:
                raise QiskitBraketException(
                    f"Instruction for {instruction.num_qubits}"
                    f" qubits are not supported."
                )
            target.add_instruction(instruction, instruction_props)
    # annealing devices
    elif isinstance(properties, DwaveDeviceCapabilities):
        raise NotImplementedError("Dwave devices are not supported yet.")
    else:
        raise QiskitBraketException(
            f"Cannot convert to target. "
            f"{properties.__class__} device capabilities are not supported yet."
        )

    return target
