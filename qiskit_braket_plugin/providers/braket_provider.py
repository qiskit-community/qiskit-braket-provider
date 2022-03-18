"""AWS Braket provider."""

from braket.aws import AwsDevice
from qiskit.providers import ProviderV1

from .braket_backend import AWSBraketDeviceBackend


class AWSBraketProvider(ProviderV1):
    """AWSBraketProvider class for accessing AWS Braket backends."""

    def backends(self, name=None, **kwargs):
        names = [name] if name else None
        devices = AwsDevice.get_devices(names=names, **kwargs)
        backends = []
        for device in devices:
            backends.append(
                AWSBraketDeviceBackend(
                    device=device,
                    provider=self,
                    name=device.name,
                    description=f"AWS Device: {device.provider_name} {device.name}.",
                    online_date=device.properties.service.updatedAt,
                    backend_version="2",
                )
            )
        return backends
