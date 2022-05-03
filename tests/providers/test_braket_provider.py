"""Tests for AWS Braket provider."""
import unittest
from unittest import TestCase
from unittest.mock import Mock

from braket.aws import AwsDeviceType
from qiskit import transpile
from qiskit.circuit.random import random_circuit

from qiskit_braket_provider.providers import AWSBraketProvider
from qiskit_braket_provider.providers.braket_backend import (
    BraketBackend,
    AWSBraketBackend,
)
from tests.providers.mocks import (
    MOCK_GATE_MODEL_SIMULATOR_SV,
    MOCK_GATE_MODEL_SIMULATOR_TN,
    SIMULATOR_REGION,
)


class TestAWSBraketProvider(TestCase):
    """Tests AWSBraketProvider."""

    def test_provider_backends(self):
        """Tests provider."""
        mock_session = Mock()
        simulators = [MOCK_GATE_MODEL_SIMULATOR_SV, MOCK_GATE_MODEL_SIMULATOR_TN]
        mock_session.get_device.side_effect = simulators
        mock_session.region = SIMULATOR_REGION
        mock_session.boto_session.region_name = SIMULATOR_REGION
        mock_session.search_devices.return_value = simulators

        provider = AWSBraketProvider()
        backends = provider.backends(
            aws_session=mock_session, types=[AwsDeviceType.SIMULATOR]
        )

        self.assertTrue(len(backends) > 0)
        for backend in backends:
            with self.subTest(f"{backend.name}"):
                self.assertIsInstance(backend, BraketBackend)

    @unittest.skip("Call to external service")
    def test_real_devices(self):
        """Tests real devices."""
        provider = AWSBraketProvider()
        backends = provider.backends()
        self.assertTrue(len(backends) > 0)
        for backend in backends:
            with self.subTest(f"{backend.name}"):
                self.assertIsInstance(backend, AWSBraketBackend)

        online_simulators_backends = provider.backends(
            statuses=["ONLINE"], types=["SIMULATOR"]
        )
        for backend in online_simulators_backends:
            with self.subTest(f"{backend.name}"):
                self.assertIsInstance(backend, AWSBraketBackend)

    @unittest.skip("Call to external service")
    def test_real_device_circuit_execution(self):
        """Tests circuit execution on real device."""
        provider = AWSBraketProvider()
        state_vector_backend = provider.get_backend("SV1")
        circuit = random_circuit(3, 5, seed=42)
        transpiled_circuit = transpile(
            circuit, backend=state_vector_backend, seed_transpiler=42
        )
        result = state_vector_backend.run(transpiled_circuit, shots=10)
        self.assertTrue(result)
