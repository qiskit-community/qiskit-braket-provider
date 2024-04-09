"""Deprecated Amazon Braket Qiskit Job classes"""

from typing import List, Optional, Union
from warnings import warn

from braket.aws import AwsQuantumTask
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit.providers import BackendV2

from .braket_task import BraketTask


class AmazonBraketTask(BraketTask):
    """AmazonBraketTask."""

    def __init_subclass__(cls, **kwargs):
        """This throws a deprecation warning on subclassing."""
        warn(f"{cls.__name__} is deprecated.", DeprecationWarning, stacklevel=2)
        super().__init_subclass__(**kwargs)

    def __init__(
        self,
        task_id: str,
        backend: BackendV2,
        tasks: Union[List[LocalQuantumTask], List[AwsQuantumTask]],
        **metadata: Optional[dict],
    ):
        """This throws a deprecation warning on initialization."""
        warn(
            f"{self.__class__.__name__} is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(task_id=task_id, backend=backend, tasks=tasks, **metadata)


class AWSBraketJob(BraketTask):
    """AWSBraketJob."""

    def __init_subclass__(cls, **kwargs):
        """This throws a deprecation warning on subclassing."""
        warn(f"{cls.__name__} is deprecated.", DeprecationWarning, stacklevel=2)
        super().__init_subclass__(**kwargs)

    def __init__(
        self,
        job_id: str,
        backend: BackendV2,
        tasks: Union[List[LocalQuantumTask], List[AwsQuantumTask]],
        **metadata: Optional[dict],
    ):
        """This throws a deprecation warning on initialization."""
        warn(
            f"{self.__class__.__name__} is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(task_id=job_id, backend=backend, tasks=tasks, **metadata)
        self._job_id = job_id
