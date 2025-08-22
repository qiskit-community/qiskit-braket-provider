"""Amazon Braket task."""

from datetime import datetime
from typing import List, Union

from qiskit.providers import BackendV2, JobStatus, JobV1
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

from braket.aws import AwsQuantumTask, AwsQuantumTaskBatch
from braket.aws.queue_information import QuantumTaskQueueInfo
from braket.tasks import GateModelQuantumTaskResult, QuantumTask
from braket.tasks.local_quantum_task import LocalQuantumTask

_TASK_STATUS_MAP = {
    "INITIALIZED": JobStatus.INITIALIZING,
    "QUEUED": JobStatus.INITIALIZING,
    "FAILED": JobStatus.ERROR,
    "CANCELLING": JobStatus.CANCELLED,
    "CANCELLED": JobStatus.CANCELLED,
    "COMPLETED": JobStatus.DONE,
    "RUNNING": JobStatus.RUNNING,
}


def retry_if_result_none(result):
    """Retry on result function."""
    return result is None


def _result_from_circuit_task(
    task: Union[LocalQuantumTask, AwsQuantumTask], result: GateModelQuantumTaskResult
) -> ExperimentResult:
    if not result:
        return None

    if result.task_metadata.shots == 0:
        braket_statevector = result.values[
            result._result_types_indices["{'type': <Type.statevector: 'statevector'>}"]
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
            memory=["".join(shot_result[::-1].astype(str)) for shot_result in result.measurements],
        )

    return ExperimentResult(
        shots=result.task_metadata.shots,
        success=True,
        status=(
            task.state() if isinstance(task, LocalQuantumTask) else result.task_metadata.status
        ),
        data=data,
    )


class BraketQuantumTask(JobV1):
    """BraketQuantumTask."""

    def __init__(
        self,
        task_id: str,
        backend: BackendV2,
        tasks: Union[List[LocalQuantumTask], List[AwsQuantumTask], AwsQuantumTask],
        **metadata,
    ):
        """BraketQuantumTask for execution of circuits on Amazon Braket or locally.

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
        return self.metadata["metadata"]["shots"] if "shots" in self.metadata["metadata"] else 0

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
        if isinstance(self._tasks, QuantumTask):
            return self._tasks.queue_position()
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
        tasks = self._tasks
        if isinstance(tasks, QuantumTask):
            # Guaranteed to be program set result
            experiment_results = [
                ExperimentResult(
                    shots=len((executable_result := program_result[0]).measurements),
                    success=True,
                    data=ExperimentResultData(
                        counts=executable_result.counts,
                        memory=[
                            "".join(shot_result[::-1].astype(str))
                            for shot_result in executable_result.measurements
                        ],
                    ),
                )
                for program_result in tasks.result()
            ]
            status = tasks.state()
            return Result(
                backend_name=self._backend.name,
                backend_version=self._backend.version,
                job_id=self._task_id,
                qobj_id=0,
                success=status not in AwsQuantumTask.NO_RESULT_TERMINAL_STATES,
                results=experiment_results,
                status=status,
            )

        experiment_results = [
            _result_from_circuit_task(task, result)
            for task, result in zip(
                tasks,
                AwsQuantumTaskBatch._retrieve_results(
                    tasks, AwsQuantumTaskBatch.MAX_CONNECTIONS_DEFAULT
                ),
            )
        ]
        status = self.status(use_cached_value=True)

        return Result(
            backend_name=self._backend.name,
            backend_version=self._backend.version,
            job_id=self._task_id,
            qobj_id=0,
            success=status not in AwsQuantumTask.NO_RESULT_TERMINAL_STATES,
            results=experiment_results,
            status=status,
        )

    def cancel(self):
        if isinstance(self._tasks, QuantumTask):
            self._tasks.cancel()
        else:
            for task in self._tasks:
                task.cancel()

    def status(self, use_cached_value: bool = False):
        if isinstance(self._tasks, QuantumTask):
            return _TASK_STATUS_MAP[self._tasks.state()]
        braket_tasks_states = [
            (
                task.state()
                if isinstance(task, LocalQuantumTask)
                else task.state(use_cached_value=use_cached_value)
            )
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
