"""Amazon Braket provider."""

import warnings

from braket.aws import AwsDevice
from braket.circuits.noise_model import NoiseModel
from braket.device_schema.dwave import DwaveDeviceCapabilities
from braket.device_schema.quera import QueraDeviceCapabilities
from braket.device_schema.xanadu import XanaduDeviceCapabilities
from qiskit.providers import ProviderV1

from .braket_backend import BraketAwsBackend, BraketLocalBackend


class BraketProvider(ProviderV1):
    """BraketProvider class for accessing Amazon Braket backends.

    Example:
        >>> provider = BraketProvider()
        >>> backends = provider.backends()
        >>> backends
        [BraketBackend[Aria 1],
         BraketBackend[Aria 2],
         BraketBackend[Aspen-M-3],
         BraketBackend[Forte 1],
         BraketBackend[Harmony],
         BraketBackend[Lucy],
         BraketBackend[SV1],
         BraketBackend[TN1],
         BraketBackend[dm1]]
    """

    def set_noise_model(self, noise_model: NoiseModel) -> None:
        """Set the noise model of the device.

        Args:
            noise_model (NoiseModel): The Braket noise model to apply to the circuit before
                execution. Noise model can only be added to the devices that support noise
                simulation.
        """
        self._validate_noise_model_support(noise_model)
        self._noise_model = noise_model

    def _validate_noise_model_support(self, noise_model: NoiseModel) -> None:
        if not isinstance(noise_model, NoiseModel):
            raise ValueError(
                "Invalid noise model specified. Should be instance of Braket noise model"
            )

    def backends(self, name=None, **kwargs):
        noise_model = kwargs.pop("noise_model") if "noise_model" in kwargs else None
        if noise_model:
            self.set_noise_model(noise_model)

        if kwargs.get("local"):
            return [
                BraketLocalBackend(name="braket_sv", noise_model=noise_model),
                BraketLocalBackend(name="braket_dm", noise_model=noise_model),
            ]
        names = [name] if name else None
        devices = AwsDevice.get_devices(names=names, **kwargs)
        # filter by supported devices
        # gate models are only supported
        supported_devices = [
            d
            for d in devices
            if not isinstance(
                d.properties,
                (
                    DwaveDeviceCapabilities,
                    XanaduDeviceCapabilities,
                    QueraDeviceCapabilities,
                ),
            )
        ]
        return [
            BraketAwsBackend(
                device=device,
                provider=self,
                name=device.name,
                description=f"AWS Device: {device.provider_name} {device.name}.",
                online_date=device.properties.service.updatedAt,
                backend_version="2",
            )
            for device in supported_devices
        ]


class AWSBraketProvider(BraketProvider):
    """AWSBraketProvider class for accessing Amazon Braket backends."""

    def __init_subclass__(cls, **kwargs):
        """This throws a deprecation warning on subclassing."""
        warnings.warn(
            f"{cls.__name__} is deprecated.", DeprecationWarning, stacklevel=2
        )
        super().__init_subclass__(**kwargs)

    def __init__(self):
        """This throws a deprecation warning on initialization."""
        warnings.warn(
            f"{self.__class__.__name__} is deprecated. Use BraketProvider instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
