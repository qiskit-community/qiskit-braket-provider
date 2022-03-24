"""AWS Braket backends."""
import datetime
from abc import ABC
from typing import Union, List, Iterable

from braket.aws import AwsDevice
from qiskit.providers import BackendV2, QubitProperties, Provider, Options
from qiskit.transpiler import Target

from qiskit_braket_plugin.providers.utils import aws_device_to_target


class BraketBackend(BackendV2, ABC):
    """BraketBackend."""

    def __repr__(self):
        return f"BraketBackend[{self.name}]"


class BraketLocalBackend(BraketBackend):
    """BraketLocalBackend."""

    def __init__(self, **fields):
        """BraketLocalBackend for local execution of circuits.

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
        raise NotImplementedError(f"Drive channel is not supported by {self.name}.")

    def measure_channel(self, qubit: int):
        raise NotImplementedError(f"Measure channel is not supported by {self.name}.")

    def acquire_channel(self, qubit: int):
        raise NotImplementedError(f"Acquire channel is not supported by {self.name}.")

    def control_channel(self, qubits: Iterable[int]):
        raise NotImplementedError(f"Control channel is not supported by {self.name}.")

    def run(self, run_input, **options):
        pass


class AWSBraketBackend(BraketBackend):
    """BraketBackend."""

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
