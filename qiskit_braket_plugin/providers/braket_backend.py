"""AWS Braket backends."""

import logging

from abc import ABC

from braket.devices import LocalSimulator
from qiskit import QuantumCircuit

from .braket_job import AWSBraketJob
from typing import Iterable, Union, List

from braket.circuits import Circuit
from braket.aws import AwsDevice, AwsQuantumTask, AwsSession
from qiskit.providers import BackendV2, QubitProperties
from qiskit.qobj import QasmQobj
from qiskit.transpiler import Target

from .transpilation import convert_circuit

logger = logging.getLogger(__name__)


class AWSBraketBackend(BackendV2, ABC):
    """AWSBraketBackend."""


class AWSBraketLocalBackend(AWSBraketBackend):
    """AWSBraketLocalBackend."""

    def __init__(self, backend_name: str):
        """AWSBraketLocalBackend for local execution of circuits.

        Args:
            **fields:
        """
        super().__init__(backend_name)
        self.backend_name = backend_name
        self._target = Target()

        '''
        # device = LocalSimulator()                                                     #Local State Vector Simulator
        # device = LocalSimulator("default")                                            #Local State Vector Simulator
        # device = LocalSimulator(backend="default")                                    #Local State Vector Simulator
        # device = LocalSimulator(backend="braket_sv")                                  #Local State Vector Simulator
        # device = LocalSimulator(backend="braket_dm")                                  #Local Density Matrix Simulator
        '''
        self._aws_device = LocalSimulator(backend=self.backend_name)

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

    def run(self, run_input: Union[QuantumCircuit, List[QuantumCircuit]], **options):
        # If we get here, then we can continue with running, else ValueError!

        circuits: List[Circuit] = list(convert_circuit([run_input]))

        #TODO: change
        shots = 1024

        tasks = []
        try:

            for circuit in circuits:
                task = self._aws_device.run(
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

        job = AWSBraketJob(
            # TODO: use correct job_id
            job_id="TODO",
            tasks=tasks,
            backend=self._aws_device,
            circuit=circuit
        )
        return job


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
