"""AWS Braket backends."""
import datetime
from abc import ABC
from typing import Union, List

from braket.devices import Device
from qiskit.providers import BackendV2, QubitProperties, Provider
from qiskit.transpiler import Target


class AWSBraketBackend(BackendV2, ABC):
    """AWSBraketBackend."""

    def __repr__(self):
        return f"AWSBraketBackend[{self.name}]"


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

    def run(self, run_input, **options):
        pass


class AWSBraketDeviceBackend(AWSBraketBackend):
    """AWSBraketBackend."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        device: Device,
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

    def run(self, run_input, **options):
        pass
