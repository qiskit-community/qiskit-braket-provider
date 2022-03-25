"""AWS Braket backends."""

import logging

from abc import ABC

from braket.devices import LocalSimulator
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit import QuantumCircuit

from .braket_job import AWSBraketJob
from typing import Iterable, Union, List

from braket.circuits import Circuit
from qiskit.providers import BackendV2, QubitProperties, Options
from qiskit.transpiler import Target

from .transpilation import convert_circuit

logger = logging.getLogger(__name__)


class AWSBraketBackend(BackendV2, ABC):
    """AWSBraketBackend."""


class AWSBraketLocalBackend(AWSBraketBackend):
    """AWSBraketLocalBackend."""

    def __init__(
            self,
            name: str = None,
            **fields
    ):
        """AWSBraketLocalBackend for local execution of circuits.

        Args:
            name: name of backend
            **fields:
        """
        super().__init__(
            name,
            **fields
        )
        self.backend_name = name
        self._target = Target()
        '''
        # device = LocalSimulator()                                                     #Local State Vector Simulator
        # device = LocalSimulator("default")                                            #Local State Vector Simulator
        # device = LocalSimulator(backend="default")                                    #Local State Vector Simulator
        # device = LocalSimulator(backend="braket_sv")                                  #Local State Vector Simulator
        # device = LocalSimulator(backend="braket_dm")                                  #Local Density Matrix Simulator
        '''
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

    def run(self, run_input: Union[QuantumCircuit, List[QuantumCircuit]], **options) -> AWSBraketJob:

        convert_input = [run_input] if type(run_input) is QuantumCircuit else [_input for _input in run_input]
        circuits: List[Circuit] = list(convert_circuit(convert_input))
        shots = options["shots"] if "shots" in options else 1024
        tasks = []
        try:
            for circuit in circuits:
                task: Union[LocalQuantumTask] = self._aws_device.run(
                    task_specification=circuit,
                    shots=shots
                )
                tasks.append(task)

        except Exception as ex:
            logger.error(f'During creation of tasks an error occurred: {ex}')
            logger.error(f'Cancelling all tasks {len(tasks)}!')
            for task in tasks:
                logger.error(f'Attempt to cancel {task.id}...')
                task.cancel()
                logger.error(f'State of {task.id}: {task.state()}.')
            raise ex

        return AWSBraketJob(
            job_id=tasks[0].id,  # TODO: if there is 2 circuits what job_id should be returned?
            tasks=tasks,
            backend=self._aws_device,
            shots=shots
        )


class AWSBraketDeviceBackend(AWSBraketBackend):
    """AWSBraketBackend."""

    def __init__(self, **fields):
        """AWSBraketBackend for execution circuits against AWS Braket devices.

        Args:
            **fields:
        """
        super().__init__(**fields)
        self._target = Target()

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        pass

    @classmethod
    def _default_options(cls):
        pass

    @property
    def dtm(self) -> float:
        pass

    @property
    def meas_map(self) -> List[List[int]]:
        pass

    def qubit_properties(
            self, qubit: Union[int, List[int]]
    ) -> Union[QubitProperties, List[QubitProperties]]:
        pass

    def drive_channel(self, qubit: int):
        pass

    def measure_channel(self, qubit: int):
        pass

    def acquire_channel(self, qubit: int):
        pass

    def control_channel(self, qubits: Iterable[int]):
        pass

    def run(self, run_input, **options):
        pass
