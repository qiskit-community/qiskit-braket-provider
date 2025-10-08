"""Mocks for testing."""

import copy
import enum
import uuid
from collections import Counter

import numpy as np

from braket.device_schema.rigetti import RigettiDeviceCapabilities
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.task_result import ProgramSetTaskResult, TaskMetadata
from braket.tasks import GateModelQuantumTaskResult, ProgramSetQuantumTaskResult
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit_braket_provider.providers.braket_backend import BraketBackend

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
        "nativeGateSet": ["rx", "rz", "cnot"],
        "connectivity": {
            "fullyConnected": False,
            "connectivityGraph": {
                "1": ["2"],
                "2": ["1", "5"],
                "5": ["1", "6"],
            },
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

RIGETTI_MOCK_M_3_QPU_CAPABILITIES_JSON: dict = copy.deepcopy(
    RIGETTI_MOCK_GATE_MODEL_QPU_CAPABILITIES_JSON
)
RIGETTI_MOCK_M_3_QPU_CAPABILITIES_JSON["action"]["braket.ir.openqasm.program"][
    "supportedOperations"
] = ["RX", "RZ", "CP", "CZ", "XY"]
RIGETTI_MOCK_M_3_QPU_CAPABILITIES_JSON["action"]["braket.ir.openqasm.program"][
    "supportedModifiers"
] = [{"name": "ctrl", "max_qubits": 4}]
RIGETTI_MOCK_M_3_QPU_CAPABILITIES_JSON["paradigm"]["qubitCount"] = 4
RIGETTI_MOCK_M_3_QPU_CAPABILITIES_JSON["paradigm"]["connectivity"]["connectivityGraph"] = {
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
            "supportedOperations": ["H", "CNOT"],
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
    task_metadata=TaskMetadata(**{"id": str(uuid.uuid4()), "deviceId": "default", "shots": 3}),
    additional_metadata=None,
    measurements=np.array([[0, 1], [0, 1], [1, 0]]),
    measured_qubits=[0, 1],
    result_types=None,
    values=None,
    measurement_counts=Counter({"01": 2, "10": 1}),
)

MOCK_LOCAL_QUANTUM_TASK = LocalQuantumTask(MOCK_GATE_MODEL_QUANTUM_TASK_RESULT)

MOCK_PROGRAM_RESULT = {
    "braketSchemaHeader": {
        "name": "braket.task_result.program_result",
        "version": "1",
    },
    "executableResults": [
        {
            "braketSchemaHeader": {
                "name": "braket.task_result.program_set_executable_result",
                "version": "1",
            },
            "measurements": [
                [0, 0],
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 1],
                [0, 0],
                [1, 1],
                [1, 0],
                [1, 1],
                [0, 0],
                [1, 1],
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [0, 0],
                [1, 1],
                [0, 0],
            ],
            "measuredQubits": [0, 1],
            "inputsIndex": 0,
        }
    ],
    "source": {
        "braketSchemaHeader": {
            "name": "braket.ir.openqasm.program",
            "version": "1",
        },
        "source": "OPENQASM 3.0;",  # noqa
        "inputs": {"theta": [0.12, 2.1]},
    },
    "additionalMetadata": {
        "simulatorMetadata": {
            "braketSchemaHeader": {
                "name": "braket.task_result.simulator_metadata",
                "version": "1",
            },
            "executionDuration": 50,
        }
    },
}
# pylint: disable-next=no-value-for-parameter
MOCK_PROGRAM_SET_RESULT = ProgramSetQuantumTaskResult.from_object(
    ProgramSetTaskResult(
        **{
            "braketSchemaHeader": {
                "name": "braket.task_result.program_set_task_result",
                "version": "1",
            },
            "programResults": [MOCK_PROGRAM_RESULT] * 2,
            "taskMetadata": {
                "braketSchemaHeader": {
                    "name": "braket.task_result.program_set_task_metadata",
                    "version": "1",
                },
                "id": "TaskID",  # noqa
                "deviceId": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                "requestedShots": 120,
                "successfulShots": 100,
                "programMetadata": [{"executables": [{}]}],
                "deviceParameters": {
                    "braketSchemaHeader": {
                        "name": "braket.device_schema.simulators.gate_model_simulator_device_parameters",
                        "version": "1",
                    },
                    "paradigmParameters": {
                        "braketSchemaHeader": {
                            "name": "braket.device_schema.gate_model_parameters",
                            "version": "1",
                        },
                        "qubitCount": 5,
                        "disableQubitRewiring": False,
                    },
                },
                "createdAt": "2024-10-15T19:06:58.986Z",
                "endedAt": "2024-10-15T19:07:00.382Z",
                "status": "COMPLETED",
                "totalFailedExecutables": 1,
            },
        }
    )
)
MOCK_PROGRAM_SET_QUANTUM_TASK = LocalQuantumTask(MOCK_PROGRAM_SET_RESULT)


class MockBraketBackend(BraketBackend):
    """
    Mock class for BraketBackend.
    """

    @property
    def target(self):
        pass

    @property
    def max_circuits(self):
        pass

    @classmethod
    def _default_options(cls):
        pass

    def run(self, run_input, **kwargs):
        """
        Mock method for run.
        """
        pass


class MockMeasLevelEnum(enum.Enum):
    """
    Mock class for MeasLevelEnum.
    """

    LEVEL_TWO = 2
