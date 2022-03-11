"""AWS Braket backends."""
from abc import ABC
from typing import Iterable, Union, List

from qiskit.providers import BackendV2, QubitProperties
from qiskit.transpiler import Target


class AWSBraketBackend(BackendV2, ABC):
    """AWSBraketBackend."""


class AWSBraketLocalBackend(AWSBraketBackend):
    """AWSBraketLocalBackend."""

    def __init__(self, **fields):
        """AWSBraketLocalBackend for local execution of circuits.

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
