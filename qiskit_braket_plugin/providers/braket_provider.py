"""AWS Braket provider."""

from braket.aws import AwsDevice
from braket.device_schema.dwave import DwaveDeviceCapabilities
from qiskit.providers import ProviderV1

from .braket_backend import AWSBraketBackend


class AWSBraketProvider(ProviderV1):
    """AWSBraketProvider class for accessing AWS Braket backends."""

    def backends(self, name=None, **kwargs):
        names = [name] if name else None
        devices = AwsDevice.get_devices(names=names, **kwargs)
        # filter by supported devices
        # gate models are only supported
        supported_devices = [
            d for d in devices if not isinstance(d.properties, DwaveDeviceCapabilities)
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
