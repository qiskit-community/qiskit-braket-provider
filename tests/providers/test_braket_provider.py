"""Tests for AWS Braket provider."""
from unittest import TestCase
from unittest.mock import Mock, patch

from braket.aws import AwsDevice, AwsDeviceType
from qiskit import transpile
from qiskit.circuit import QuantumCircuit

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

    def setUp(self):
        self.mock_session = Mock()
        simulators = [MOCK_GATE_MODEL_SIMULATOR_SV, MOCK_GATE_MODEL_SIMULATOR_TN]
        self.mock_session.get_device.side_effect = simulators
        self.mock_session.region = SIMULATOR_REGION
        self.mock_session.boto_session.region_name = SIMULATOR_REGION
        self.mock_session.search_devices.return_value = simulators

    def test_provider_backends(self):
        """Tests provider."""
        provider = AWSBraketProvider()
        backends = provider.backends(
            aws_session=self.mock_session, types=[AwsDeviceType.SIMULATOR]
        )

        self.assertTrue(len(backends) > 0)
        for backend in backends:
            with self.subTest(f"{backend.name}"):
                self.assertIsInstance(backend, BraketBackend)

    def test_real_devices(self):
        """Tests real devices."""
        with patch(
            "qiskit_braket_provider.providers.braket_provider.AwsDevice"
        ) as mock_get_devices:
            mock_get_devices.get_devices.return_value = [
                AwsDevice(MOCK_GATE_MODEL_SIMULATOR_SV["deviceArn"], self.mock_session),
                AwsDevice(MOCK_GATE_MODEL_SIMULATOR_TN["deviceArn"], self.mock_session),
            ]
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

    @patch("qiskit_braket_provider.providers.braket_backend.AwsDevice.get_devices")
    def test_qiskit_circuit_transpilation(self, mock_get_devices):
        """Tests qiskit circuit transpilation."""
        mock_get_devices.return_value = [
            AwsDevice(MOCK_GATE_MODEL_SIMULATOR_SV["deviceArn"], self.mock_session)
        ]

        provider = AWSBraketProvider()
        state_vector_backend = provider.get_backend(
            "SV1", aws_session=self.mock_session
        )
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        transpiled_circuit = transpile(
            circuit, backend=state_vector_backend, seed_transpiler=42
        )
        self.assertTrue(transpiled_circuit)
