"""AWS Braket provider."""

from braket.aws import AwsDevice
from braket.device_schema.dwave import DwaveDeviceCapabilities
from braket.device_schema.xanadu import XanaduDeviceCapabilities
from qiskit.providers import ProviderV1

from .braket_backend import AWSBraketBackend, BraketLocalBackend


class AWSBraketProvider(ProviderV1):
    """AWSBraketProvider class for accessing AWS Braket backends.

    Example:
        >>> provider = AWSBraketProvider()
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
            return [BraketLocalBackend(name="default")]
        names = [name] if name else None
        devices = AwsDevice.get_devices(names=names, **kwargs)
        # filter by supported devices
        # gate models are only supported
        supported_devices = [
            d
            for d in devices
            if not isinstance(
                d.properties, (DwaveDeviceCapabilities, XanaduDeviceCapabilities)
            )
        ]
        backends = []
        for device in supported_devices:
            backends.append(
                AWSBraketBackend(
                    device=device,
                    provider=self,
                    name=device.name,
                    description=f"AWS Device: {device.provider_name} {device.name}.",
                    online_date=device.properties.service.updatedAt,
                    backend_version="2",
                )
            )
        return backends
