"""Tests for AWS Braket provider."""

from unittest import TestCase
from unittest.mock import Mock

from braket.aws import AwsDeviceType
from braket.device_schema.rigetti import RigettiDeviceCapabilities
from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
)

from qiskit_braket_plugin.providers import AWSBraketProvider
from qiskit_braket_plugin.providers.braket_backend import AWSBraketBackend

RIGETTI_ARN = "arn:aws:braket:::device/qpu/rigetti/Aspen-10"
SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
TN1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/tn1"

MOCK_GATE_MODEL_QPU_CAPABILITIES_JSON_1 = {
    "braketSchemaHeader": {
        "name": "braket.device_schema.rigetti.rigetti_device_capabilities",
        "version": "1",
    },
    "service": {
        "executionWindows": [
            {
                "executionDay": "Everyday",
                "windowStartHour": "11:00",
                "windowEndHour": "12:00",
            }
        ],
        "shotsRange": [1, 10],
    },
    "action": {
        "braket.ir.jaqcd.program": {
            "actionType": "braket.ir.jaqcd.program",
            "version": ["1"],
            "supportedOperations": ["H"],
        }
    },
    "paradigm": {
        "qubitCount": 30,
        "nativeGateSet": ["ccnot", "cy"],
        "connectivity": {
            "fullyConnected": False,
            "connectivityGraph": {"1": ["2", "3"]},
        },
    },
    "deviceParameters": {},
}

MOCK_GATE_MODEL_QPU_CAPABILITIES_1 = RigettiDeviceCapabilities.parse_obj(
    MOCK_GATE_MODEL_QPU_CAPABILITIES_JSON_1
)

MOCK_GATE_MODEL_QPU_1 = {
    "deviceName": "Aspen-10",
    "deviceType": "QPU",
    "providerName": "provider1",
    "deviceStatus": "OFFLINE",
    "deviceArn": RIGETTI_ARN,
    "deviceCapabilities": MOCK_GATE_MODEL_QPU_CAPABILITIES_1.json(),
}
RIGETTI_REGION = "us-west-1"

MOCK_GATE_MODEL_SIMULATOR_CAPABILITIES_JSON = {
    "braketSchemaHeader": {
        "name": "braket.device_schema.simulators.gate_model_simulator_device_capabilities",
        "version": "1",
    },
    "service": {
        "executionWindows": [
            {
                "executionDay": "Everyday",
                "windowStartHour": "11:00",
                "windowEndHour": "12:00",
            }
        ],
        "shotsRange": [1, 10],
    },
    "action": {
        "braket.ir.jaqcd.program": {
            "actionType": "braket.ir.jaqcd.program",
            "version": ["1"],
            "supportedOperations": ["H"],
        }
    },
    "paradigm": {"qubitCount": 30},
    "deviceParameters": {},
}

MOCK_GATE_MODEL_SIMULATOR_CAPABILITIES = GateModelSimulatorDeviceCapabilities.parse_obj(
    MOCK_GATE_MODEL_SIMULATOR_CAPABILITIES_JSON
)


class TestAWSBraketProvider(TestCase):
    """Tests AWSBraketProvider."""

    def test_provider_backends(self):
        """Tests provider."""
        mock_session = Mock()
        mock_session.get_device.return_value = MOCK_GATE_MODEL_QPU_1
        mock_session.region = RIGETTI_REGION
        mock_session.boto_session.region_name = RIGETTI_REGION
        mock_session.search_devices.return_value = [MOCK_GATE_MODEL_QPU_1]

        provider = AWSBraketProvider()
        backends = provider.backends(
            aws_session=mock_session, types=[AwsDeviceType.SIMULATOR]
        )

        self.assertTrue(len(backends) > 0)
        for backend in backends:
            with self.subTest(f"{backend.name}"):
                self.assertIsInstance(backend, AWSBraketBackend)
