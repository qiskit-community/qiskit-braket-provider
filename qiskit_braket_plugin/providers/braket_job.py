"""AWS Braket job."""
import os
from datetime import datetime

from typing import List, Optional, Union

from braket.aws import AwsQuantumTask
from braket.tasks import GateModelQuantumTaskResult
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit.providers import JobV1, BackendV2, JobStatus
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from retrying import retry


def retry_if_result_none(result):
    """Retry on result function."""
    return result is None


@retry(
    retry_on_result=retry_if_result_none,
    stop_max_delay=os.environ.get("QISKIT_BRAKET_PROVIDER_MAX_DELAY", 60000),
    wait_fixed=os.environ.get("QISKIT_BRAKET_PROVIDER_WAIT_TIME", 2000),
    wrap_exception=True,
)
def _get_result_from_aws_tasks(
    tasks: Union[List[LocalQuantumTask], List[AwsQuantumTask]], shots: int
) -> Optional[List[ExperimentResult]]:
    """Returns experiment results of AWS tasks.

    Args:
        tasks: AWS Quantum tasks
        shots: number of shots

    Returns:
        List of experiment results.
    """
    experiment_results: List[ExperimentResult] = []

    # For each task the results is get and filled into an ExperimentResult object
    for task in tasks:
        if task.state() in AwsQuantumTask.RESULTS_READY_STATES:
            result: GateModelQuantumTaskResult = task.result()
            counts = {
                k[::-1]: v for k, v in dict(result.measurement_counts).items()
            }  # convert to little-endian
            data = ExperimentResultData(counts=counts, memory=list(result.measurements))
        else:
            return None

        experiment_result = ExperimentResult(
            shots=shots,
            success=True,
            status=task.state(),
            data=data,
        )
        experiment_results.append(experiment_result)

    return experiment_results


class AWSBraketJob(JobV1):
    """AWSBraketJob."""

    def __init__(
        self,
        job_id: str,
        backend: BackendV2,
        tasks: Union[List[LocalQuantumTask], List[AwsQuantumTask]],
        **metadata: Optional[dict]
    ):
        """AWSBraketJob for local execution of circuits.

        Args:
            job_id: id of the job
            backend: Local simulator
            tasks: Executed tasks
            **metadata:
        """
        super().__init__(backend=backend, job_id=job_id, metadata=metadata)
        self._job_id = job_id
        self._backend = backend
        self._metadata = metadata
        self._tasks = tasks
        self._date_of_creation = datetime.now()

    @property
    def shots(self) -> int:
        """Return the number of shots.

        Returns:
            shots: int with the number of shots.
        """
        return (
            self.metadata["metadata"]["shots"]
            if "shots" in self.metadata["metadata"]
            else 0
        )

    def submit(self):
        return

    def result(self) -> Result:
        experiment_results = _get_result_from_aws_tasks(
            tasks=self._tasks, shots=self.shots
        )
        return Result(
            backend_name=self._backend,
            backend_version=self._backend.version,
            job_id=self._job_id,
            qobj_id=0,
            success=self.status() not in AwsQuantumTask.NO_RESULT_TERMINAL_STATES,
            results=experiment_results,
            status=self.status(),
        )

    def cancel(self):
        for task in self._tasks:
            task.cancel()

    def status(self):
        braket_tasks_states = [task.state() for task in self._tasks]

        if "FAILED" in braket_tasks_states:
            status = JobStatus.ERROR
        elif "CANCELLED" in braket_tasks_states:
            status = JobStatus.CANCELLED
        elif all(state == "COMPLETED" for state in braket_tasks_states):
            status = JobStatus.DONE
        else:
            status = JobStatus.RUNNING

        return status
