"""AWS Braket provider."""

from qiskit.providers import ProviderV1


class AWSBraketProvider(ProviderV1):
    """AWSBraketProvider class for accessing AWS Braket backends."""

    def backends(self, name=None, **kwargs):
        return []
