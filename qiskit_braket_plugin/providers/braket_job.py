"""AWS Braket job."""
from datetime import datetime

from braket.aws import AwsQuantumTask
from braket.devices import LocalSimulator
from braket.tasks import GateModelQuantumTaskResult
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit.providers import JobV1
from qiskit.providers.models import BackendStatus
from typing import List, Optional, Union

from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData


class AWSBraketJob(JobV1):
    """AWSBraketJob."""

    def __init__(self,
                 job_id: str,
                 backend: 'BraketBackend',
                 tasks: Union[List[QuantumTask]],
                 **metadata: Optional[dict]):
        """AWSBraketJob for local execution of circuits.

              Args:
                  job_id: id of the job
                  backend: Local simulator
                  tasks: Executed tasks
                  **metadata:
              """
        super().__init__(
            backend=backend,
            job_id=job_id,
            metadata=metadata
        )
        self._job_id = job_id
        self._backend = backend
        self._metadata = metadata
        self._tasks = tasks
        self._date_of_creation = datetime.now()

    @property
    def shots(self) -> int:
        # TODO: Shots can be retrieved from tasks metadata
        return self.metadata["metadata"]["shots"] if "shots" in self.metadata["metadata"] else 0

    def submit(self):
        pass

    def result(self, **kwargs) -> Result:

        experiment_results: List[ExperimentResult] = []
        task: AwsQuantumTask

        # For each task the results is get and filled into an ExperimentResult object
        for task in self._tasks:
            result: GateModelQuantumTaskResult = task.result()
            data = ExperimentResultData(
                counts=dict(result.measurement_counts)
            )
            experiment_result = ExperimentResult(
                shots=self.shots,
                success=task.state() == 'COMPLETED',
                status=task.state(),
                data=data
            )
            experiment_results.append(experiment_result)

        return Result(
            backend_name=self._backend,
            backend_version=1,
            job_id=self._job_id,
            qobj_id=0,
            success=self.status(),
            results=experiment_results
        )

    def cancel(self):
        pass

    def status(self):
        status: str = self._backend._aws_device.status
        backend_status: BackendStatus = BackendStatus(
            backend_name=self._backend.name,
            backend_version="",
            operational=False,
            pending_jobs=0,  # TODO
            status_msg=status

        )
        if status == 'ONLINE' or status == 'AVAILABLE':
            backend_status.operational = True
        elif status == 'OFFLINE':
            backend_status.operational = False
        else:
            backend_status.operational = False
        return backend_status
