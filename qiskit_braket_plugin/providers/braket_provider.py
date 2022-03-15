"""AWS Braket provider."""

from braket.devices import LocalSimulator
from qiskit.providers import ProviderV1
from .braket_backend import AWSBraketLocalBackend

class AWSBraketProvider(ProviderV1):
    """AWSBraketProvider class for accessing AWS Braket backends."""

    def backends(self, name=None, **kwargs):
        #TODO: Logic of what backend should be returned
        return [AWSBraketLocalBackend(backend_name="default")]
