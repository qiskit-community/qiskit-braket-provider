"""Util function for provider."""
from collections.abc import Callable, Iterable
from typing import Optional, Union
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
from qiskit.circuit import ControlledGate, Measure, Parameter
import qiskit.circuit.library as qiskit_gates

from qiskit.transpiler import InstructionProperties, Target
from qiskit_braket_provider.exception import QiskitBraketException

BRAKET_TO_QISKIT_NAMES = {
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

_CONTROLLED_GATES_BY_QUBIT_COUNT = {
    1: {"ch", "cs", "csdg", "csx", "crx", "cry", "crz", "ccz"},
    3: {"c3sx"},
}
_ARBITRARY_CONTROLLED_GATES = {"mcx"}

_EPS = 1e-10  # global variable used to chop very small numbers to zero

GATE_NAME_TO_BRAKET_GATE: dict[str, Callable] = {
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

_QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES: dict[str, Callable] = {
    "ch": braket_gates.H,
    "cs": braket_gates.S,
    "csdg": braket_gates.Si,
    "csx": braket_gates.V,
    "ccz": braket_gates.CZ,
    "c3sx": braket_gates.V,
    "mcx": braket_gates.CNot,
    "crx": braket_gates.Rx,
    "cry": braket_gates.Ry,
    "crz": braket_gates.Rz,
}

_TRANSLATABLE_QISKIT_GATE_NAMES = (
    set(GATE_NAME_TO_BRAKET_GATE.keys())
    .union(set(_QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES))
    .union({"measure", "barrier", "reset"})
)

GATE_NAME_TO_QISKIT_GATE: dict[str, Optional[QiskitInstruction]] = {
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


def get_controlled_gateset(max_qubits: Optional[int] = None) -> set[str]:
    """Returns the Qiskit gates expressible as controlled versions of existing Braket gates

    This set can be filtered by the maximum number of control qubits.

    Args:
        max_qubits (Optional[int]): The maximum number of control qubits that can be used to express
            the Qiskit gate as a controlled Braket gate. If `None`, then there is no limit to the
            number of control qubits. Default: `None`.

    Returns:
        set[str]: The names of the controlled gates.
    """
    if max_qubits is None:
        gateset = set().union(*[g for _, g in _CONTROLLED_GATES_BY_QUBIT_COUNT.items()])
        gateset.update(_ARBITRARY_CONTROLLED_GATES)
        return gateset
    return set().union(
        *[g for q, g in _CONTROLLED_GATES_BY_QUBIT_COUNT.items() if q <= max_qubits]
    )


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


def to_braket(
    circuit: QuantumCircuit,
    gateset: Optional[Iterable[str]] = None,
    verbatim: bool = False,
) -> Circuit:
    """Return a Braket quantum circuit from a Qiskit quantum circuit.
     Args:
            circuit (QuantumCircuit): Qiskit quantum circuit
            gateset (Optional[Iterable[str]]): The gateset to transpile to.
                If `None`, the transpiler will use all gates defined in the Braket SDK.
                Default: `None`.
            verbatim (bool): Whether to translate the circuit without any modification, in other
                words without transpiling it. Default: False.

    Returns:
        Circuit: Braket circuit
    """
    gateset = gateset or _TRANSLATABLE_QISKIT_GATE_NAMES
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(f"Expected a QuantumCircuit, got {type(circuit)} instead.")

    braket_circuit = Circuit()
    if not verbatim and not {gate.name for gate, _, _ in circuit.data}.issubset(
        gateset
    ):
        circuit = transpile(circuit, basis_gates=gateset, optimization_level=0)

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
            warnings.warn(
                "The Qiskit circuit contains barrier instructions that are ignored."
            )
        elif gate_name == "reset":
            raise NotImplementedError(
                "reset operation not supported by qiskit to braket adapter"
            )
        else:
            params = _create_free_parameters(operation)
            if (
                isinstance(operation, ControlledGate)
                and operation.ctrl_state != 2**operation.num_ctrl_qubits - 1
            ):
                raise ValueError("Negative control is not supported")
            if gate_name in _QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES:
                gate = _QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES[gate_name](*params)
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
                for gate in GATE_NAME_TO_BRAKET_GATE[gate_name](*params):
                    instruction = Instruction(
                        operator=gate,
                        target=[circuit.find_bit(qubit).index for qubit in qubits],
                    )
                    braket_circuit += instruction

    if circuit.global_phase > _EPS:
        braket_circuit.gphase(circuit.global_phase)

    if verbatim:
        return Circuit(braket_circuit.result_types).add_verbatim_box(
            Circuit(braket_circuit.instructions)
        )

    return braket_circuit


def _create_free_parameters(operation):
    params = operation.params if hasattr(operation, "params") else []
    for i, param in enumerate(params):
        if isinstance(param, Parameter):
            params[i] = FreeParameter(param.name)
    return params


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
    circuits: list[QuantumCircuit],
) -> Iterable[Circuit]:
    """Converts all Qiskit circuits to Braket circuits.

     Args:
            circuits (List(QuantumCircuit)): Qiskit quantum circuit

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
            circuit (Circuit): Braket quantum circuit

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
