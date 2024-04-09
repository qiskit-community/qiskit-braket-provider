"""Amazon Braket task."""

from datetime import datetime
from typing import List, Optional, Union

from braket.aws import AwsQuantumTask, AwsQuantumTaskBatch
from braket.aws.queue_information import QuantumTaskQueueInfo
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit.providers import BackendV2, JobStatus, JobV1
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData


def retry_if_result_none(result):
    """Retry on result function."""
    return result is None


def _get_result_from_aws_tasks(
    tasks: Union[List[LocalQuantumTask], List[AwsQuantumTask]],
) -> Optional[List[ExperimentResult]]:
    """Returns experiment results of AWS tasks.

    Args:
        tasks: AWS Quantum tasks
        shots: number of shots

    Returns:
        List of experiment results.
    """
    experiment_results: List[ExperimentResult] = []

    results = AwsQuantumTaskBatch._retrieve_results(
        tasks, AwsQuantumTaskBatch.MAX_CONNECTIONS_DEFAULT
    )

    # For each task we create an ExperimentResult object with the downloaded results.
    for task, result in zip(tasks, results):
        if not result:
            return None

        if result.task_metadata.shots == 0:
            braket_statevector = result.values[
                result._result_types_indices[
                    "{'type': <Type.statevector: 'statevector'>}"
                ]
            ]
            data = ExperimentResultData(
                statevector=Statevector(braket_statevector).reverse_qargs().data,
            )
        else:
            counts = {
                k[::-1]: v for k, v in dict(result.measurement_counts).items()
            }  # convert to little-endian

            data = ExperimentResultData(
                counts=counts,
                memory=[
                    "".join(shot_result[::-1].astype(str))
                    for shot_result in result.measurements
                ],
            )

        experiment_result = ExperimentResult(
            shots=result.task_metadata.shots,
            success=True,
            status=task.state()
            if isinstance(task, LocalQuantumTask)
            else result.task_metadata.status,
            data=data,
        )
        experiment_results.append(experiment_result)

    return experiment_results


class BraketTask(JobV1):
    """BraketTask."""

    def __init__(
        self,
        task_id: str,
        backend: BackendV2,
        tasks: Union[List[LocalQuantumTask], List[AwsQuantumTask]],
        **metadata: Optional[dict],
    ):
        """BraketTask for execution of circuits on Amazon Braket or locally.

        Args:
            task_id: Semicolon-separated IDs of the underlying tasks
            backend: BraketBackend that ran the circuit
            tasks: Executed tasks
            **metadata: Additional metadata
        """
        super().__init__(backend=backend, job_id=task_id, metadata=metadata)
        self._task_id = task_id
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

    def queue_position(self) -> QuantumTaskQueueInfo:
        """
        The queue position details for the quantum job.

        Returns:
            QuantumTaskQueueInfo: Instance of QuantumTaskQueueInfo class
            representing the queue position information for the quantum job.
            The queue_position is only returned when quantum job is not in
            RUNNING/CANCELLING/TERMINAL states, else queue_position is returned as None.
            The normal tasks refers to the quantum jobs not submitted via Hybrid Jobs.
            Whereas, the priority tasks refers to the total number of quantum jobs waiting to run
            submitted through Amazon Braket Hybrid Jobs. These tasks run before the normal tasks.
            If the queue position for normal or priority quantum tasks is greater than 2000,
            we display their respective queue position as '>2000'.

            Note: We don't provide queue information for the LocalQuantumTasks.

        Examples:
            job status = QUEUED and queue position is 2050
            >>> task.queue_position()
            QuantumTaskQueueInfo(queue_type=<QueueType.NORMAL: 'Normal'>,
            queue_position='>2000', message=None)

            job status = COMPLETED
            >>> task.queue_position()
            QuantumTaskQueueInfo(queue_type=<QueueType.NORMAL: 'Normal'>,
            queue_position=None, message='Task is in COMPLETED status. AmazonBraket does
            not show queue position for this status.')
        """
        for task in self._tasks:
            if isinstance(task, LocalQuantumTask):
                raise NotImplementedError(
                    "We don't provide queue information for the LocalQuantumTask."
                )
            return AwsQuantumTask(self.task_id()).queue_position()

    def task_id(self) -> str:
        """Return a unique id identifying the task."""
        return self._task_id

    def result(self) -> Result:
        experiment_results = _get_result_from_aws_tasks(tasks=self._tasks)
        status = self.status(use_cached_value=True)

        return Result(
            backend_name=self._backend,
            backend_version=self._backend.version,
            job_id=self._task_id,
            qobj_id=0,
            success=status not in AwsQuantumTask.NO_RESULT_TERMINAL_STATES,
            results=experiment_results,
            status=status,
        )

    def cancel(self):
        for task in self._tasks:
            task.cancel()

    def status(self, use_cached_value: bool = False):
        braket_tasks_states = [
            task.state()
            if isinstance(task, LocalQuantumTask)
            else task.state(use_cached_value=use_cached_value)
            for task in self._tasks
        ]

        if "FAILED" in braket_tasks_states:
            status = JobStatus.ERROR
        elif "CANCELLED" in braket_tasks_states:
            status = JobStatus.CANCELLED
        elif all(state == "COMPLETED" for state in braket_tasks_states):
            status = JobStatus.DONE
        elif all(state == "RUNNING" for state in braket_tasks_states):
            status = JobStatus.RUNNING
        else:
            status = JobStatus.QUEUED

        return status
