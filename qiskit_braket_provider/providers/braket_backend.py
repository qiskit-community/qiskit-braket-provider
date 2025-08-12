"""Amazon Braket backends."""

import datetime
import enum
import logging
import warnings
from abc import ABC
from collections.abc import Iterable
from typing import Optional, Union

from braket.aws import AwsDevice, AwsQuantumTask
from braket.aws.queue_information import QueueDepthInfo
from braket.circuits import Circuit
from braket.device_schema import DeviceActionType
from braket.devices import Device, LocalSimulator
from braket.program_sets import ProgramSet
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2, Options, Provider, QubitProperties

from .. import version
from ..exception import QiskitBraketException
from .adapter import (
    aws_device_to_target,
    gateset_from_properties,
    local_simulator_to_target,
    native_angle_restrictions,
    native_gate_connectivity,
    native_gate_set,
    to_braket,
)
from .braket_quantum_task import BraketQuantumTask

logger = logging.getLogger(__name__)


_TASK_ID_DIVIDER = ";"


class BraketBackend(BackendV2, ABC):
    """BraketBackend."""

    def __repr__(self):
        return f"BraketBackend[{self.name}]"

    @property
    def _device(self) -> Device:
        raise NotImplementedError

    def _validate_meas_level(self, meas_level: Union[enum.Enum, int]):
        if isinstance(meas_level, enum.Enum):
            meas_level = meas_level.value
        if meas_level != 2:
            raise QiskitBraketException(
                f"Device {self.name} only supports classified measurement "
                f"results, received meas_level={meas_level}."
            )

    def get_gateset(self, native=False) -> Optional[set[str]]:
        """Get the gate set of the device.

        Args:
            native (bool): Whether to return the device's native gates.
        """
        if native:
            return native_gate_set(self._device.properties)
        else:
            action = self._device.properties.action.get(DeviceActionType.OPENQASM)
            return gateset_from_properties(action) if action else None


class BraketLocalBackend(BraketBackend):
    """BraketLocalBackend."""

    def __init__(self, name: str = "default", **fields):
        """BraketLocalBackend for executing circuits locally.

        Example:
            >>> device = LocalSimulator()                    #Local State Vector Simulator
            >>> device = LocalSimulator("default")           #Local State Vector Simulator
            >>> device = LocalSimulator(name="default")      #Local State Vector Simulator
            >>> device = LocalSimulator(name="braket_sv")    #Local State Vector Simulator
            >>> device = LocalSimulator(name="braket_dm")    #Local Density Matrix Simulator

        Args:
            name: name of backend
            **fields: extra fields
        """
        super().__init__(name=name, **fields)
        self.backend_name = name
        self._local_device = LocalSimulator(backend=self.backend_name)
        self._target = local_simulator_to_target(self._local_device)
        self.status = self._local_device.status

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options()

    @property
    def dtm(self) -> float:
        raise NotImplementedError(
            f"System time resolution of output signals is not supported by {self.name}."
        )

    @property
    def meas_map(self) -> list[list[int]]:
        raise NotImplementedError(f"Measurement map is not supported by {self.name}.")

    @property
    def _device(self) -> Device:
        return self._local_device

    def qubit_properties(
        self, qubit: Union[int, list[int]]
    ) -> Union[QubitProperties, list[QubitProperties]]:
        raise NotImplementedError

    def drive_channel(self, qubit: int):
        raise NotImplementedError(f"Drive channel is not supported by {self.name}.")

    def measure_channel(self, qubit: int):
        raise NotImplementedError(f"Measure channel is not supported by {self.name}.")

    def acquire_channel(self, qubit: int):
        raise NotImplementedError(f"Acquire channel is not supported by {self.name}.")

    def control_channel(self, qubits: Iterable[int]):
        raise NotImplementedError(f"Control channel is not supported by {self.name}.")

    def run(
        self, run_input: Union[QuantumCircuit, list[QuantumCircuit]], **options
    ) -> BraketQuantumTask:
        convert_input = (
            [run_input] if isinstance(run_input, QuantumCircuit) else list(run_input)
        )
        verbatim = options.pop("verbatim", False)
        gateset = self.get_gateset() if not verbatim else None
        circuits: list[Circuit] = [
            to_braket(circ, gateset, verbatim) for circ in convert_input
        ]

        shots = options["shots"] if "shots" in options else 1024
        if shots == 0:
            circuits = list(map(lambda x: x.state_vector(), circuits))
        if "meas_level" in options:
            self._validate_meas_level(options["meas_level"])
            del options["meas_level"]
        tasks = []
        try:
            for circuit in circuits:
                task: LocalQuantumTask = self._device.run(
                    task_specification=circuit, shots=shots
                )
                tasks.append(task)

        except Exception as ex:
            logger.error("During creation of tasks an error occurred: %s", ex)
            logger.error("Cancelling all tasks %d!", len(tasks))
            for task in tasks:
                logger.error("Attempt to cancel %s...", task.id)
                task.cancel()
                logger.error("State of %s: %s.", task.id, task.state())
            raise ex

        task_id = _TASK_ID_DIVIDER.join(task.id for task in tasks)

        return BraketQuantumTask(
            task_id=task_id,
            tasks=tasks,
            backend=self,
            shots=shots,
        )


class BraketAwsBackend(BraketBackend):
    """BraketAwsBackend."""

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        arn: Optional[str] = None,
        provider: Provider = None,
        name: str = None,
        description: str = None,
        online_date: datetime.datetime = None,
        backend_version: str = None,
        *,
        device: Optional[AwsDevice] = None,
        **fields,
    ):
        """BraketAwsBackend for executing circuits on Amazon Braket devices.

        Example:
            >>> provider = BraketProvider()
            >>> backend = provider.get_backend("SV1")
            >>> transpiled_circuit = transpile(circuit, backend=backend)
            >>> backend.run(transpiled_circuit, shots=10).result().get_counts()
            {"100": 10, "001": 10}

        Args:
            arn: ARN of the Braket device
            provider: Qiskit provider for this backend
            name: name of backend
            description: description of backend
            online_date: online date
            backend_version: backend version
            device: Braket device instance
            **fields: other arguments
        """
        if not (arn or device):
            raise ValueError("Must specify either arn or device")
        if arn and device:
            raise ValueError("Can only specify one of arn and device")
        super().__init__(
            provider=provider,
            name=name,
            description=description,
            online_date=online_date,
            backend_version=backend_version,
            **fields,
        )
        self._aws_device = AwsDevice(arn) if arn else device
        self._aws_device.aws_session.add_braket_user_agent(
            f"QiskitBraketProvider/{version.__version__}"
        )
        self._target = aws_device_to_target(device=self._aws_device)
        self._supports_program_sets = (
            DeviceActionType.OPENQASM_PROGRAM_SET in self._aws_device.properties.action
        )

    def retrieve_job(self, task_id: str) -> BraketQuantumTask:
        """Return a single job submitted to AWS backend.

        Args:
            task_id: ID of the task to retrieve.

        Returns:
            The job with the given ID.
        """
        task_ids = task_id.split(_TASK_ID_DIVIDER)

        return BraketQuantumTask(
            task_id=task_id,
            backend=self,
            tasks=[AwsQuantumTask(arn=task_id) for task_id in task_ids],
        )

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @property
    def _device(self) -> Device:
        return self._aws_device

    @classmethod
    def _default_options(cls):
        return Options()

    def qubit_properties(
        self, qubit: Union[int, list[int]]
    ) -> Union[QubitProperties, list[QubitProperties]]:
        # TODO: fetch information from device.properties.provider  # pylint: disable=fixme
        raise NotImplementedError

    def queue_depth(self) -> QueueDepthInfo:
        """
        Task queue depth refers to the total number of quantum tasks currently waiting
        to run on a particular device.

        Returns:
            QueueDepthInfo: Instance of the QueueDepth class representing queue depth
            information for quantum jobs and hybrid jobs.
            Queue depth refers to the number of quantum jobs and hybrid jobs queued on a particular
            device. The normal tasks refers to the quantum jobs not submitted via Hybrid Jobs.
            Whereas, the priority tasks refers to the total number of quantum jobs waiting to run
            submitted through Amazon Braket Hybrid Jobs. These tasks run before the normal tasks.
            If the queue depth for normal or priority quantum tasks is greater than 4000, we display
            their respective queue depth as '>4000'. Similarly, for hybrid jobs if there are more
            than 1000 jobs queued on a device, display the hybrid jobs queue depth as '>1000'.
            Additionally, for QPUs if hybrid jobs queue depth is 0, we display information about
            priority and count of the running hybrid job.

        Example:
            Queue depth information for a running hybrid job.
            >>> device = BraketProvider().get_backend("SV1")
            >>> print(device.queue_depth())
            QueueDepthInfo(quantum_tasks={<QueueType.NORMAL: 'Normal'>: '0',
            <QueueType.PRIORITY: 'Priority'>: '1'}, jobs='0 (1 prioritized job(s) running)')

            If more than 4000 quantum jobs queued on a device.
            >>> device = BraketProvider().get_backend("SV1")
            >>> print(device.queue_depth())
            QueueDepthInfo(quantum_tasks={<QueueType.NORMAL: 'Normal'>: '>4000',
            <QueueType.PRIORITY: 'Priority'>: '2000'}, jobs='100')
        """
        return self._device.queue_depth()

    @property
    def dtm(self) -> float:
        raise NotImplementedError(
            f"System time resolution of output signals is not supported by {self.name}."
        )

    @property
    def meas_map(self) -> list[list[int]]:
        raise NotImplementedError(f"Measurement map is not supported by {self.name}.")

    def drive_channel(self, qubit: int):
        raise NotImplementedError(f"Drive channel is not supported by {self.name}.")

    def measure_channel(self, qubit: int):
        raise NotImplementedError(f"Measure channel is not supported by {self.name}.")

    def acquire_channel(self, qubit: int):
        raise NotImplementedError(f"Acquire channel is not supported by {self.name}.")

    def control_channel(self, qubits: Iterable[int]):
        raise NotImplementedError(f"Control channel is not supported by {self.name}.")

    def run(self, run_input, verbatim: bool = False, native: bool = False, **options):
        if isinstance(run_input, QuantumCircuit):
            circuits = [run_input]
        elif isinstance(run_input, list):
            circuits = run_input
        else:
            raise QiskitBraketException(f"Unsupported input type: {type(run_input)}")

        if "meas_level" in options:
            self._validate_meas_level(options["meas_level"])
            del options["meas_level"]

        gateset = self.get_gateset(native) if not verbatim else None
        connectivity = (
            native_gate_connectivity(self._device.properties) if native else None
        )
        angle_restrictions = (
            native_angle_restrictions(self._device.properties) if native else None
        )

        braket_circuits = [
            to_braket(
                circ,
                basis_gates=gateset,
                verbatim=verbatim,
                connectivity=connectivity,
                angle_restrictions=angle_restrictions,
            )
            for circ in circuits
        ]
        shots = options.pop("shots", None)
        return (
            self._run_program_set(braket_circuits, shots, **options)
            if self._supports_program_sets and shots != 0
            else self._run_batch(braket_circuits, shots, **options)
        )

    def _run_program_set(
        self, braket_circuits: list[Circuit], shots: Optional[int], **options
    ):
        program_set = ProgramSet(braket_circuits, shots_per_executable=shots)
        task = self._aws_device.run(program_set, **options)
        return BraketQuantumTask(
            task_id=task.id, tasks=task, backend=self, shots=program_set.total_shots
        )

    def _run_batch(self, braket_circuits: list[Circuit], shots: int, **options):
        batch_task = self._aws_device.run_batch(braket_circuits, shots=shots, **options)
        tasks: list[AwsQuantumTask] = batch_task.tasks
        task_id = _TASK_ID_DIVIDER.join(task.id for task in tasks)
        return BraketQuantumTask(
            task_id=task_id, tasks=tasks, backend=self, shots=shots
        )


class AWSBraketBackend(BraketAwsBackend):
    """AWSBraketBackend."""

    def __init_subclass__(cls, **kwargs):
        """This throws a deprecation warning on subclassing."""
        warnings.warn(
            f"{cls.__name__} is deprecated.", DeprecationWarning, stacklevel=2
        )
        super().__init_subclass__(**kwargs)

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        device: AwsDevice,
        provider: Provider = None,
        name: str = None,
        description: str = None,
        online_date: datetime.datetime = None,
        backend_version: str = None,
        **fields,
    ):
        """This throws a deprecation warning on initialization."""
        warnings.warn(
            f"{self.__class__.__name__} is deprecated. Use BraketAwsBackend instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            device=device,
            provider=provider,
            name=name,
            description=description,
            online_date=online_date,
            backend_version=backend_version,
            **fields,
        )
