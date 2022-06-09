"""AWS Braket backends."""


import datetime
import logging
from abc import ABC
from typing import Iterable, Union, List
import pkg_resources

from braket.aws import AwsDevice, AwsQuantumTaskBatch, AwsQuantumTask
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2, QubitProperties, Options, Provider

from .adapter import (
    aws_device_to_target,
    local_simulator_to_target,
    convert_qiskit_to_braket_circuits,
)
from .braket_job import AWSBraketJob
from ..exception import QiskitBraketException

logger = logging.getLogger(__name__)


TASK_ID_DIVIDER = ";"


class BraketBackend(BackendV2, ABC):
    """BraketBackend."""

    def __repr__(self):
        return f"BraketBackend[{self.name}]"


class BraketLocalBackend(BraketBackend):
    """BraketLocalBackend."""

    def __init__(self, name: str = "default", **fields):
        """AWSBraketLocalBackend for local execution of circuits.

        Example:
            >>> device = LocalSimulator()                         #Local State Vector Simulator
            >>> device = LocalSimulator("default")                #Local State Vector Simulator
            >>> device = LocalSimulator(name="default")        #Local State Vector Simulator
            >>> device = LocalSimulator(name="braket_sv")      #Local State Vector Simulator
            >>> device = LocalSimulator(name="braket_dm")      #Local Density Matrix Simulator

        Args:
            name: name of backend
            **fields: extra fields
        """
        super().__init__(name="sv_simulator", **fields)
        self.backend_name = name
        self._aws_device = LocalSimulator(backend=self.backend_name)
        self._target = local_simulator_to_target(self._aws_device)
        self.status = self._aws_device.status

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
    def meas_map(self) -> List[List[int]]:
        raise NotImplementedError(f"Measurement map is not supported by {self.name}.")

    def qubit_properties(
        self, qubit: Union[int, List[int]]
    ) -> Union[QubitProperties, List[QubitProperties]]:
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
        self, run_input: Union[QuantumCircuit, List[QuantumCircuit]], **options
    ) -> AWSBraketJob:

        convert_input = (
            [run_input] if isinstance(run_input, QuantumCircuit) else list(run_input)
        )
        circuits: List[Circuit] = list(convert_qiskit_to_braket_circuits(convert_input))
        shots = options["shots"] if "shots" in options else 1024
        tasks = []
        try:
            for circuit in circuits:
                task: Union[LocalQuantumTask] = self._aws_device.run(
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

        job_id = TASK_ID_DIVIDER.join(task.id for task in tasks)

        return AWSBraketJob(
            job_id=job_id,
            tasks=tasks,
            backend=self,
            shots=shots,
        )


class AWSBraketBackend(BraketBackend):
    """AWSBraketBackend."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        device: AwsDevice,
        provider: Provider = None,
        name: str = None,
        description: str = None,
        online_date: datetime.datetime = None,
        backend_version: str = None,
        **fields,
    ):
        """AWSBraketBackend for execution circuits against AWS Braket devices.

        Example:
            >>> provider = AWSBraketProvider()
            >>> backend = provider.get_backend("SV1")
            >>> transpiled_circuit = transpile(circuit, backend=backend)
            >>> backend.run(transpiled_circuit, shots=10).result().get_counts()
            {"100": 10, "001": 10}

        Args:
            device: Braket device class
            provider: Qiskit provider for this backend
            name: name of backend
            description: description of backend
            online_date: online date
            backend_version: backend version
            **fields: other arguments
        """
        super().__init__(
            provider=provider,
            name=name,
            description=description,
            online_date=online_date,
            backend_version=backend_version,
            **fields,
        )
        pkg_version = pkg_resources.get_distribution("qiskit-braket-provider").version
        user_agent = f"QiskitBraketProvider/{pkg_version}"
        device.aws_session.add_braket_user_agent(user_agent)
        self._device = device
        self._target = aws_device_to_target(device=device)

    def retrieve_job(self, job_id: str) -> AWSBraketJob:
        """Return a single job submitted to AWS backend.

        Args:
            job_id: ID of the job to retrieve.

        Returns:
            The job with the given ID.
        """
        task_ids = job_id.split(TASK_ID_DIVIDER)

        return AWSBraketJob(
            job_id=job_id,
            backend=self,
            tasks=[AwsQuantumTask(arn=task_id) for task_id in task_ids],
        )

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options()

    def qubit_properties(
        self, qubit: Union[int, List[int]]
    ) -> Union[QubitProperties, List[QubitProperties]]:
        # TODO: fetch information from device.properties.provider  # pylint: disable=fixme
        raise NotImplementedError

    @property
    def dtm(self) -> float:
        raise NotImplementedError(
            f"System time resolution of output signals is not supported by {self.name}."
        )

    @property
    def meas_map(self) -> List[List[int]]:
        raise NotImplementedError(f"Measurement map is not supported by {self.name}.")

    def drive_channel(self, qubit: int):
        raise NotImplementedError(f"Drive channel is not supported by {self.name}.")

    def measure_channel(self, qubit: int):
        raise NotImplementedError(f"Measure channel is not supported by {self.name}.")

    def acquire_channel(self, qubit: int):
        raise NotImplementedError(f"Acquire channel is not supported by {self.name}.")

    def control_channel(self, qubits: Iterable[int]):
        raise NotImplementedError(f"Control channel is not supported by {self.name}.")

    def run(self, run_input, **options):
        if isinstance(run_input, QuantumCircuit):
            circuits = [run_input]
        elif isinstance(run_input, list):
            circuits = run_input
        else:
            raise QiskitBraketException(f"Unsupported input type: {type(run_input)}")

        braket_circuits = list(convert_qiskit_to_braket_circuits(circuits))
        batch_task: AwsQuantumTaskBatch = self._device.run_batch(
            braket_circuits, shots=options.get("shots")
        )
        tasks: List[AwsQuantumTask] = batch_task.tasks
        job_id = TASK_ID_DIVIDER.join(task.id for task in tasks)

        return AWSBraketJob(
            job_id=job_id, tasks=tasks, backend=self, shots=options.get("shots")
        )
