from collections.abc import Callable

from qiskit.primitives import BasePrimitiveJob, PrimitiveResult, PubResult
from qiskit.providers import JobStatus

from braket.tasks import ProgramSetQuantumTaskResult, QuantumTask
from qiskit_braket_provider.providers.braket_quantum_task import _TASK_STATUS_MAP


class BraketPrimitiveTask(BasePrimitiveJob[PrimitiveResult[PubResult], JobStatus]):
    """
    Job class for Braket-native primitives.

    This class wraps a Braket QuantumTask and constructs a PrimitiveResult
    from the ProgramSetQuantumTaskResult.
    """

    def __init__(
        self,
        task: QuantumTask,
        result_translator: Callable[[ProgramSetQuantumTaskResult], PrimitiveResult],
    ):
        """
        Initialize the task.

        Args:
            task (QuantumTask): The Braket QuantumTask
            result_translator (Callable[[ProgramSetQuantumTaskResult], PrimitiveResult]): Function
                to convert the result of the Braket task to a Qiskit primitive result.
        """
        super().__init__(job_id=task.id)
        self._task = task
        self._result_translator = result_translator
        self._result = None

    def result(self) -> PrimitiveResult:
        if self._result is None:
            self._result = self._result_translator(self._task.result())
        return self._result

    def status(self) -> JobStatus:
        return self._get_task_status()

    def cancel(self):
        self._task.cancel()

    def job_id(self) -> str:
        return self._task.id

    def done(self) -> bool:
        return self._get_task_status() == JobStatus.DONE

    def running(self) -> bool:
        return self._get_task_status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        return self._get_task_status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        return self._get_task_status() in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]

    def _get_task_status(self) -> JobStatus:
        return _TASK_STATUS_MAP[self._task.state()]
