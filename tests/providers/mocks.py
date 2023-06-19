"""Mocks for testing."""

from collections import Counter
import copy

import uuid
import numpy as np
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.task_result import TaskMetadata
from braket.tasks import GateModelQuantumTaskResult
from braket.tasks.local_quantum_task import LocalQuantumTask
from braket.device_schema.rigetti import RigettiDeviceCapabilities


RIGETTI_ARN = "arn:aws:braket:::device/qpu/rigetti/Aspen-10"
RIGETTI_ASPEN_ARN = "arn:aws:braket:::device/qpu/rigetti/Aspen-M-3"
SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
TN1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/tn1"
RIGETTI_REGION = "us-west-1"
SIMULATOR_REGION = "us-west-1"


RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES_JSON = {
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
        "braket.ir.openqasm.program": {
            "actionType": "braket.ir.openqasm.program",
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

RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES = RigettiDeviceCapabilities.parse_obj(
    RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES_JSON
)

RIGETTI_MOCK_GATE_MODEL_QPU = {
    "deviceName": "Aspen-10",
    "deviceType": "QPU",
    "providerName": "provider1",
    "deviceStatus": "OFFLINE",
    "deviceArn": RIGETTI_ARN,
    "deviceCapabilities": RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES.json(),
}

RIGETTI_MOCK_M_3_QPU_CAPABILITIES_JSON = copy.deepcopy(
    RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES_JSON
)
RIGETTI_MOCK_M_3_QPU_CAPABILITIES_JSON["action"]["braket.ir.openqasm.program"][
    "supportedOperations"
] = ["RX", "RZ", "CP", "CZ", "XY"]
RIGETTI_MOCK_M_3_QPU_CAPABILITIES_JSON["paradigm"]["qubitCount"] = 4
RIGETTI_MOCK_M_3_QPU_CAPABILITIES_JSON["paradigm"]["connectivity"][
    "connectivityGraph"
] = {
    "0": ["1", "2", "7"],
    "1": ["0", "2", "7"],
    "2": ["0", "1", "7"],
    "7": ["0", "1", "2"],
}
RIGETTI_MOCK_M_3_QPU_CAPABILITIES = RigettiDeviceCapabilities.parse_obj(
    RIGETTI_MOCK_M_3_QPU_CAPABILITIES_JSON
)

MOCK_RIGETTI_GATE_MODEL_M_3_QPU = {
    "deviceName": "Aspen-M-3",
    "deviceType": "QPU",
    "providerName": "provider1",
    "deviceStatus": "ONLINE",
    "deviceArn": RIGETTI_ASPEN_ARN,
    "deviceCapabilities": RIGETTI_MOCK_M_3_QPU_CAPABILITIES.json(),
}

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

MOCK_GATE_MODEL_SIMULATOR_SV = {
    "deviceName": "sv1",
    "deviceType": "SIMULATOR",
    "providerName": "provider1",
    "deviceStatus": "ONLINE",
    "deviceArn": SV1_ARN,
    "deviceCapabilities": MOCK_GATE_MODEL_SIMULATOR_CAPABILITIES.json(),
}

MOCK_GATE_MODEL_SIMULATOR_TN = {
    "deviceName": "tn1",
    "deviceType": "SIMULATOR",
    "providerName": "provider1",
    "deviceStatus": "ONLINE",
    "deviceArn": TN1_ARN,
    "deviceCapabilities": MOCK_GATE_MODEL_SIMULATOR_CAPABILITIES.json(),
}

MOCK_GATE_MODEL_QUANTUM_TASK_RESULT = GateModelQuantumTaskResult(
    task_metadata=TaskMetadata(
        **{"id": str(uuid.uuid4()), "deviceId": "default", "shots": 3}
    ),
    additional_metadata=None,
    measurements=np.array([[0, 1], [0, 1], [1, 0]]),
    measured_qubits=[0, 1],
    result_types=None,
    values=None,
    measurement_counts=Counter({"01": 2, "10": 1}),
)

MOCK_LOCAL_QUANTUM_TASK = LocalQuantumTask(MOCK_GATE_MODEL_QUANTUM_TASK_RESULT)
