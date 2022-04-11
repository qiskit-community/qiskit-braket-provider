"""AWS Braket backends."""


import logging
import datetime

from abc import ABC

from typing import Iterable, Union, List
from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.tasks.local_quantum_task import LocalQuantumTask
from braket.circuits import Circuit
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2, QubitProperties, Options, Provider
from qiskit.transpiler import Target

from .braket_job import AWSBraketJob

from .transpilation import convert_circuit
from .utils import aws_device_to_target

logger = logging.getLogger(__name__)


class BraketBackend(BackendV2, ABC):
    """BraketBackend."""

    def __repr__(self):
        return f"BraketBackend[{self.name}]"


class BraketLocalBackend(BraketBackend):
    """BraketLocalBackend."""

    def __init__(self, name: str = None, **fields):
        """AWSBraketLocalBackend for local execution of circuits.

        Args:
            name: name of backend
            **fields:
        """
        super().__init__(name, **fields)
        self.backend_name = name
        self._target = Target()
        """
        # device = LocalSimulator()                         #Local State Vector Simulator
        # device = LocalSimulator("default")                #Local State Vector Simulator
        # device = LocalSimulator(backend="default")        #Local State Vector Simulator
        # device = LocalSimulator(backend="braket_sv")      #Local State Vector Simulator
        # device = LocalSimulator(backend="braket_dm")      #Local Density Matrix Simulator
        """
        self._aws_device = LocalSimulator(backend=self.backend_name)
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
        circuits: List[Circuit] = list(convert_circuit(convert_input))
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

        return AWSBraketJob(
            job_id=tasks[0].id,
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
        self._device = device
        self._target = aws_device_to_target(device=device)

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
        pass
