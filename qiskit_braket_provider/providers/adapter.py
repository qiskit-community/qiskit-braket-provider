"""Util function for provider."""
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import warnings

from braket.aws import AwsDevice
from braket.circuits import (
    Circuit,
    FreeParameter,
    Instruction,
    observables,
)
import braket.circuits.gates as braket_gates

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
import qiskit.circuit.library as qiskit_gates

from qiskit.transpiler import InstructionProperties, Target
from qiskit_braket_provider.exception import QiskitBraketException

_EPS = 1e-10  # global variable used to chop very small numbers to zero

GATE_NAME_TO_BRAKET_GATE: Dict[str, Callable] = {
    "u1": lambda lam: [braket_gates.PhaseShift(lam)],
    "u2": lambda phi, lam: [
        braket_gates.PhaseShift(lam),
        braket_gates.Ry(pi / 2),
        braket_gates.PhaseShift(phi),
    ],
    "u3": lambda theta, phi, lam: [
        braket_gates.PhaseShift(lam),
        braket_gates.Ry(theta),
        braket_gates.PhaseShift(phi),
    ],
    "u": lambda theta, phi, lam: [
        braket_gates.PhaseShift(lam),
        braket_gates.Ry(theta),
        braket_gates.PhaseShift(phi),
    ],
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
}


_TRANSLATABLE_QISKIT_GATE_NAMES = set(GATE_NAME_TO_BRAKET_GATE.keys()).union(
    {"measure", "barrier", "reset"}
)

GATE_NAME_TO_QISKIT_GATE: Dict[str, Optional[QiskitInstruction]] = {
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
    "iswap": qiskit_gates.iSwapGate(),
}


def local_simulator_to_target(simulator: LocalSimulator) -> Target:
    """Converts properties of LocalSimulator into Qiskit Target object.

    Args:
        simulator: AWS LocalSimulator

    Returns:
        target for Qiskit backend
    """
    target = Target()

    instructions = [
        inst for inst in GATE_NAME_TO_QISKIT_GATE.values() if inst is not None
    ]
    properties = simulator.properties
    paradigm: GateModelSimulatorParadigmProperties = properties.paradigm

    # add measurement instruction
    target.add_instruction(Measure(), {(i,): None for i in range(paradigm.qubitCount)})

    for instruction in instructions:
        instruction_props: Optional[
            Dict[Union[Tuple[int], Tuple[int, int]], Optional[InstructionProperties]]
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
        instructions: List[QiskitInstruction] = []

        for operation in action_properties.supportedOperations:
            instruction = GATE_NAME_TO_QISKIT_GATE.get(operation.lower(), None)
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
            instruction = GATE_NAME_TO_QISKIT_GATE.get(operation.lower(), None)
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
                Dict[
                    Union[Tuple[int], Tuple[int, int]],
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


def to_braket(circuit: QuantumCircuit) -> Circuit:
    """Return a Braket quantum circuit from a Qiskit quantum circuit.
     Args:
            circuit (QuantumCircuit): Qiskit Quantum Circuit

    Returns:
        Circuit: Braket circuit
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(f"Expected a QuantumCircuit, got {type(circuit)} instead.")

    quantum_circuit = Circuit()
    if not (
        {gate.name for gate, _, _ in circuit.data}.issubset(
            _TRANSLATABLE_QISKIT_GATE_NAMES
        )
    ):
        circuit = transpile(
            circuit, basis_gates=_TRANSLATABLE_QISKIT_GATE_NAMES, optimization_level=0
        )

    # handle qiskit to braket conversion
    for circuit_instruction in circuit.data:
        operation = circuit_instruction.operation
        gate_name = operation.name

        qubits = circuit_instruction.qubits

        if gate_name == "measure":
            qubit = qubits[0]  # qubit count = 1 for measure
            qubit_index = circuit.find_bit(qubit).index
            quantum_circuit.sample(
                observable=observables.Z(),
                target=[
                    qubit_index,
                ],
            )
        elif gate_name == "barrier":
            warnings.warn(
                "The Qiskit circuit contains barrier instructions that are ignored."
            )
        elif gate_name == "reset":
            raise NotImplementedError(
                "reset operation not supported by qiskit to braket adapter"
            )
        else:
            params = operation.params if hasattr(operation, "params") else []

            for i, param in enumerate(params):
                if isinstance(param, Parameter):
                    params[i] = FreeParameter(param.name)

            for gate in GATE_NAME_TO_BRAKET_GATE[gate_name](*params):
                instruction = Instruction(
                    operator=gate,
                    target=[circuit.find_bit(qubit).index for qubit in qubits],
                )
                quantum_circuit += instruction

    if circuit.global_phase > _EPS:
        quantum_circuit.gphase(circuit.global_phase)

    return quantum_circuit


def convert_qiskit_to_braket_circuit(circuit: QuantumCircuit) -> Circuit:
    """Return a Braket quantum circuit from a Qiskit quantum circuit.
     Args:
            circuit (QuantumCircuit): Qiskit Quantum Cricuit

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
    circuits: List[QuantumCircuit],
) -> Iterable[Circuit]:
    """Converts all Qiskit circuits to Braket circuits.
     Args:
            circuits (List(QuantumCircuit)): Qiskit Quantum Cricuit

    Returns:
        Circuit (Iterable[Circuit]): Braket circuit
    """
    warnings.warn(
        "convert_qiskit_to_braket_circuits() is deprecated and "
        "will be removed in a future release. "
        "Use to_braket() instead.",
        DeprecationWarning,
    )
    for circuit in circuits:
        yield to_braket(circuit)


def to_qiskit(circuit: Circuit) -> QuantumCircuit:
    """Return a Qiskit quantum circuit from a Braket quantum circuit.
     Args:
            circuit (Circuit): Braket Quantum Cricuit

    Returns:
        QuantumCircuit: Qiskit quantum circuit
    """
    if not isinstance(circuit, Circuit):
        raise TypeError(f"Expected a Circuit, got {type(circuit)} instead.")

    qiskit_circuit = QuantumCircuit(circuit.qubit_count)
    qubit_map = {
        int(qubit): index for index, qubit in enumerate(sorted(circuit.qubits))
    }
    dict_param = {}
    for instruction in circuit.instructions:
        gate_name = instruction.operator.name.lower()

        gate_params = []
        if hasattr(instruction.operator, "parameters"):
            for value in instruction.operator.parameters:
                if isinstance(value, FreeParameter):
                    if value.name not in dict_param:
                        dict_param[value.name] = Parameter(value.name)
                    gate_params.append(dict_param[value.name])
                else:
                    gate_params.append(value)

        gate = _create_gate(gate_name, gate_params)

        if instruction.power != 1:
            gate = gate**instruction.power
        if control_qubits := instruction.control:
            ctrl_state = instruction.control_state.as_string[::-1]
            gate = gate.control(len(control_qubits), ctrl_state=ctrl_state)

        target = [qiskit_circuit.qubits[qubit_map[i]] for i in control_qubits]
        target += [qiskit_circuit.qubits[qubit_map[i]] for i in instruction.target]

        qiskit_circuit.append(gate, target)
    qiskit_circuit.measure_all()
    return qiskit_circuit


def _create_gate(
    gate_name: str, gate_params: list[Union[float, Parameter]]
) -> Instruction:
    gate_instance = GATE_NAME_TO_QISKIT_GATE.get(gate_name, None)
    if gate_instance is not None:
        gate_cls = gate_instance.__class__
    else:
        raise TypeError(f'Braket gate "{gate_name}" not supported in Qiskit')
    return gate_cls(*gate_params)


def wrap_circuits_in_verbatim_box(circuits: List[Circuit]) -> Iterable[Circuit]:
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
