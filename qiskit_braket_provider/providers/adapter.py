"""Util function for provider."""
from collections.abc import Callable, Iterable
from typing import Optional, Union
import warnings

from braket.aws import AwsDevice
from braket.circuits import (
    Circuit,
    FreeParameter,
    Instruction,
    gates,
    observables,
)
from braket.device_schema import (
    DeviceActionType,
    GateModelQpuParadigmProperties,
    JaqcdDeviceActionProperties,
    OpenQASMDeviceActionProperties,
)
from braket.device_schema.ionq import IonqDeviceCapabilities
from braket.device_schema.oqc import OqcDeviceCapabilities
from braket.device_schema.rigetti import RigettiDeviceCapabilities
from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorParadigmProperties,
)
from braket.devices import LocalSimulator

from numpy import pi

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction as QiskitInstruction
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import (
    CCXGate,
    CPhaseGate,
    CSwapGate,
    CXGate,
    CYGate,
    CZGate,
    ECRGate,
    HGate,
    IGate,
    PhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    SdgGate,
    SGate,
    SwapGate,
    SXdgGate,
    SXGate,
    TdgGate,
    TGate,
    UGate,
    U1Gate,
    U2Gate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
)

from qiskit.transpiler import InstructionProperties, Target
from qiskit_braket_provider.exception import QiskitBraketException

braket_to_qiskit_names = {
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
}

controlled_gate = {1: {"ch", "cs", "csdg", "csx", "crx", "cry", "crz", "ccz"}, 2: {"c3sx"}}

qiskit_to_braket_gate_names_mapping = {
    "u": "u",
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
    "ecr": "ecr",
}

_EPS = 1e-10  # global variable used to chop very small numbers to zero

qiskit_gate_names_to_braket_gates: dict[str, Callable] = {
    "u1": lambda lam: [gates.PhaseShift(lam)],
    "u2": lambda phi, lam: [
        gates.PhaseShift(lam),
        gates.Ry(pi / 2),
        gates.PhaseShift(phi),
    ],
    "u3": lambda theta, phi, lam: [
        gates.PhaseShift(lam),
        gates.Ry(theta),
        gates.PhaseShift(phi),
    ],
    "u": lambda theta, phi, lam: [
        gates.PhaseShift(lam),
        gates.Ry(theta),
        gates.PhaseShift(phi),
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
    "ecr": lambda: [gates.ECR()],
}

_qiskit_controlled_gate_names_to_braket_gates: dict[str, Callable] = {
    "ch": gates.H,
    "cs": gates.S,
    "csdg": gates.Si,
    "csx": gates.V,
    "ccz": gates.CZ,
    "c3sx": gates.V,
    "mcx": gates.CNot,
    "crx": gates.Rx,
    "cry": gates.Ry,
    "crz": gates.Rz,
}

translatable_qiskit_gates = (
    set(qiskit_gate_names_to_braket_gates.keys())
    .union(set(_qiskit_controlled_gate_names_to_braket_gates))
    .union({"measure", "barrier", "reset"})
)

qiskit_gate_name_to_braket_gate_mapping: dict[str, Optional[QiskitInstruction]] = {
    "u": UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
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
    "ecr": ECRGate(),
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

    instructions = [
        inst
        for inst in qiskit_gate_name_to_braket_gate_mapping.values()
        if inst is not None
    ]
    properties = simulator.properties
    paradigm: GateModelSimulatorParadigmProperties = properties.paradigm

    # add measurement instruction
    target.add_instruction(Measure(), {(i,): None for i in range(paradigm.qubitCount)})

    for instruction in instructions:
        instruction_props: Optional[
            dict[Union[tuple[int], tuple[int, int]], Optional[InstructionProperties]]
        ] = {}

        if instruction.num_qubits == 1:
            for i in range(paradigm.qubitCount):
                instruction_props[(i,)] = None
            target.add_instruction(instruction, instruction_props)
        elif instruction.num_qubits == 2:
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
        action_properties: OpenQASMDeviceActionProperties = (
            properties.action.get(DeviceActionType.OPENQASM)
            if properties.action.get(DeviceActionType.OPENQASM)
            else properties.action.get(DeviceActionType.JAQCD)
        )
        paradigm: GateModelQpuParadigmProperties = properties.paradigm
        connectivity = paradigm.connectivity
        instructions: list[QiskitInstruction] = []

        for operation in action_properties.supportedOperations:
            instruction = _op_to_instruction(operation)
            if instruction is not None:
                # TODO: remove when target will be supporting > 2 qubit gates  # pylint:disable=fixme
                if instruction.num_qubits <= 2:
                    instructions.append(instruction)

        # add measurement instructions
        target.add_instruction(
            Measure(), {(i,): None for i in range(paradigm.qubitCount)}
        )

        for instruction in instructions:
            instruction_props: Optional[
                dict[
                    Union[tuple[int], tuple[int, int]], Optional[InstructionProperties]
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
                    if isinstance(properties, RigettiDeviceCapabilities):

                        def convert_continuous_qubit_indices(
                            connectivity_graph: dict,
                        ) -> dict:
                            """Aspen qubit indices are discontinuous (label between x0 and x7, x being
                            the number of the octagon) while the Qiskit transpiler creates and/or
                            handles coupling maps with continuous indices. This function converts the
                            discontinous connectivity graph from Aspen to a continuous one.

                            Args:
                                connectivity_graph (dict): connectivity graph from Aspen. For example
                                4 qubit system, the connectivity graph will be:
                                    {"0": ["1", "2", "7"], "1": ["0","2","7"], "2": ["0","1","7"],
                                    "7": ["0","1","2"]}

                            Returns:
                                dict: Connectivity graph with continuous indices. For example for an
                                input connectivity graph with discontinuous indices (qubit 0, 1, 2 and
                                then qubit 7) as shown here:
                                    {"0": ["1", "2", "7"], "1": ["0","2","7"], "2": ["0","1","7"],
                                    "7": ["0","1","2"]}
                                the qubit index 7 will be mapped to qubit index 3 for the qiskit
                                transpilation step. Thereby the resultant continous qubit indices
                                output will be:
                                    {"0": ["1", "2", "3"], "1": ["0","2","3"], "2": ["0","1","3"],
                                    "3": ["0","1","2"]}
                            """
                            # Creates list of existing qubit indices which are discontinuous.
                            indices = [int(key) for key in connectivity_graph.keys()]
                            indices.sort()
                            # Creates a list of continuous indices for number of qubits.
                            map_list = list(range(len(indices)))
                            # Creates a dictionary to remap the discountinous indices to continuous.
                            mapper = dict(zip(indices, map_list))
                            # Performs the remapping from the discontinous to the continuous indices.
                            continous_connectivity_graph = {
                                mapper[int(k)]: [mapper[int(v)] for v in val]
                                for k, val in connectivity_graph.items()
                            }
                            return continous_connectivity_graph

                        connectivity.connectivityGraph = (
                            convert_continuous_qubit_indices(
                                connectivity.connectivityGraph
                            )
                        )

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

        # add measurement instructions
        target.add_instruction(
            Measure(), {(i,): None for i in range(simulator_paradigm.qubitCount)}
        )

        for instruction in instructions:
            simulator_instruction_props: Optional[
                dict[
                    Union[tuple[int], tuple[int, int]],
                    Optional[InstructionProperties],
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


def convert_qiskit_to_braket_circuit(circuit: QuantumCircuit, gateset=None) -> Circuit:
    """Return a Braket quantum circuit from a Qiskit quantum circuit.
     Args:
            circuit (QuantumCircuit): Qiskit Quantum Cricuit

    Returns:
        Circuit: Braket circuit
    """
    gateset = gateset or translatable_qiskit_gates
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(
            f"Expected a qiskit.QuantumCircuit, got {type(circuit)} instead"
        )

    braket_circuit = Circuit()
    if not (
        {gate.name for gate, _, _ in circuit.data}.issubset(gateset)
    ):
        circuit = transpile(circuit, basis_gates=gateset)
    if circuit.global_phase > _EPS:
        warnings.warn("Circuit transpilation resulted in global phase shift")
    # handle qiskit to braket conversion
    for circuit_instruction in circuit.data:
        operation = circuit_instruction.operation
        gate_name = operation.name

        qubits = circuit_instruction.qubits

        if gate_name == "measure":
            qubit = qubits[0]  # qubit count = 1 for measure
            qubit_index = circuit.find_bit(qubit).index
            braket_circuit.sample(
                observable=observables.Z(),
                target=[
                    qubit_index,
                ],
            )
        elif gate_name == "barrier":
            # This does not exist
            pass
        elif gate_name == "reset":
            raise NotImplementedError(
                "reset operation not supported by qiskit to braket adapter"
            )
        elif gate_name in _qiskit_controlled_gate_names_to_braket_gates:
            params = _create_free_parameters(operation)

            gate = _qiskit_controlled_gate_names_to_braket_gates[gate_name](*params)
            qubit_indices = [circuit.find_bit(qubit).index for qubit in qubits]
            gate_qubit_count = gate.qubit_count
            target_indices = qubit_indices[-gate_qubit_count:]
            instruction = Instruction(
                # Getting the index from the bit mapping
                operator=gate,
                target=target_indices,
                control=qubit_indices[:-gate_qubit_count],
            )
            braket_circuit += instruction

        else:
            params = _create_free_parameters(operation)
            for gate in qiskit_gate_names_to_braket_gates[gate_name](*params):
                instruction = Instruction(
                    # Getting the index from the bit mapping
                    operator=gate,
                    target=[circuit.find_bit(qubit).index for qubit in qubits],
                )
                braket_circuit += instruction
    return braket_circuit


def _create_free_parameters(operation):
    params = operation.params if hasattr(operation, "params") else []
    for i, param in enumerate(params):
        if isinstance(param, Parameter):
            params[i] = FreeParameter(param.name)
    return params


def convert_qiskit_to_braket_circuits(
    circuits: list[QuantumCircuit], gateset=None
) -> Iterable[Circuit]:
    """Converts all Qiskit circuits to Braket circuits.
     Args:
            circuits (List(QuantumCircuit)): Qiskit Quantum Cricuit

    Returns:
        Circuit (Iterable[Circuit]): Braket circuit
    """
    for circuit in circuits:
        yield convert_qiskit_to_braket_circuit(circuit, gateset)


def wrap_circuits_in_verbatim_box(circuits: list[Circuit]) -> Iterable[Circuit]:
    """Convert each Braket circuit an equivalent one wrapped in verbatim box.

    Args:
           circuits (List(Circuit): circuits to be wrapped in verbatim box.
    Returns:
           Circuits wrapped in verbatim box, comprising the same instructions
           as the original one and with result types preserved.
    """
    return [
        Circuit(circuit.result_types).add_verbatim_box(Circuit(circuit.instructions))
        for circuit in circuits
    ]
