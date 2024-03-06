"""AWS Braket provider."""

import warnings

from braket.aws import AwsDevice
from braket.device_schema.dwave import DwaveDeviceCapabilities
from braket.device_schema.quera import QueraDeviceCapabilities
from braket.device_schema.xanadu import XanaduDeviceCapabilities
from qiskit.providers import ProviderV1

from .braket_backend import BraketAwsBackend, BraketLocalBackend


class BraketProvider(ProviderV1):
    """BraketProvider class for accessing AWS Braket backends.

    Example:
        >>> provider = BraketProvider()
        >>> backends = provider.backends()
        >>> backends
        [BraketBackend[Aspen-10],
         BraketBackend[Aspen-11],
         BraketBackend[Aspen-8],
         BraketBackend[Aspen-9],
         BraketBackend[Aspen-M-1],
         BraketBackend[IonQ Device],
         BraketBackend[Lucy],
         BraketBackend[SV1],
         BraketBackend[TN1],
         BraketBackend[dm1]]
    """

    def backends(self, name=None, **kwargs):
        if kwargs.get("local"):
            return [
                BraketLocalBackend(name="braket_sv"),
                BraketLocalBackend(name="braket_dm"),
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
    """AWSBraketProvider class for accessing AWS Braket backends."""

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
