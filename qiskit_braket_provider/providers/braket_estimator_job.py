from dataclasses import dataclass

import numpy as np
from qiskit.primitives import BasePrimitiveJob, DataBin, PrimitiveResult, PubResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.providers import JobStatus

from braket.tasks import QuantumTask
from qiskit_braket_provider.providers.braket_quantum_task import _TASK_STATUS_MAP


@dataclass
class _PubMetadata:
    num_bindings: int
    binding_to_result_map: dict[int, tuple[tuple[int, ...], int, int]]
    sum_binding_indices: set[int]


@dataclass
class _JobMetadata:
    pubs: list[EstimatorPub]
    pub_metadata: list[_PubMetadata]
    precision: float
    shots: int


class BraketEstimatorJob(BasePrimitiveJob[PrimitiveResult[PubResult], JobStatus]):
    """
    Job class for BraketEstimator.

    This job wraps a Braket QuantumTask and reconstructs PrimitiveResult
    from the ProgramSetQuantumTaskResult.
    """

    def __init__(self, task: QuantumTask, metadata: _JobMetadata):
        """
        Initialize the estimator job.

        Args:
            task (QuantumTask): The Braket QuantumTask
            metadata (_JobMetadata): Metadata needed to reconstruct results, including:
                - circuits: List of QuantumCircuits
                - pub_metadata: List of metadata for each pub
                - precision: Target precision
                - shots: Number of shots used
        """
        super().__init__(job_id=task.id)
        self._task = task
        self._metadata = metadata
        self._result = None

    def result(self) -> PrimitiveResult:
        """
        Get the result of the job.

        Returns:
            PrimitiveResult: PrimitiveResult containing PubResult for each pub.
        """
        if self._result is None:
            self._result = self._reconstruct_results()
        return self._result

    def status(self) -> JobStatus:
        """
        Get the status of the job.

        Returns:
            JobStatus: Job status string.
        """
        return self._get_job_status()

    def cancel(self):
        """Cancel the job."""
        self._task.cancel()

    def job_id(self) -> str:
        """
        Get the job ID.

        Returns:
            str: Job ID string.
        """
        return self._task.id

    def done(self) -> bool:
        """
        Check if the job is done.

        Returns:
            bool: True if the job is done, False otherwise.
        """
        return self._get_job_status() == JobStatus.DONE

    def running(self) -> bool:
        """
        Check if the job is running.

        Returns:
            bool: True if the job is running, False otherwise.
        """
        return self._get_job_status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        """
        Check if the job was cancelled.

        Returns:
            bool: True if the job was cancelled, False otherwise.
        """
        return self._get_job_status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        """
        Check if the job is in a final state.

        Returns:
            bool: True if the job is in a final state, False otherwise.
        """
        return self._get_job_status() in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]

    def _reconstruct_results(self) -> PrimitiveResult:
        """
        Reconstruct PrimitiveResult from Braket task results.

        Returns:
            PrimitiveResult: PrimitiveResult containing PubResult for each pub.
        """
        task_result = self._task.result()
        metadata = self._metadata

        pub_results = []
        binding_offset = 0

        for pub, pub_meta in zip(metadata.pubs, metadata.pub_metadata):
            num_bindings = pub_meta.num_bindings
            broadcast_shape = pub.shape
            binding_map = pub_meta.binding_to_result_map
            sum_binding_indices = pub_meta.sum_binding_indices

            evs = np.zeros(broadcast_shape, dtype=float)
            for local_binding_idx in range(num_bindings):
                binding_result = task_result[binding_offset + local_binding_idx]
                num_observables = len(binding_result.observables)

                for position, obs_idx, param_idx in binding_map[local_binding_idx]:
                    # CircuitBinding returns results organized by parameter sets
                    # For each parameter, we get all observables
                    evs[np.unravel_index(position, broadcast_shape)] = (
                        binding_result.expectation(param_idx)
                        if local_binding_idx in sum_binding_indices
                        else binding_result[param_idx * num_observables + obs_idx].expectation
                    )

            pub_results.append(
                PubResult(
                    DataBin(evs=evs, shape=broadcast_shape),
                    metadata={
                        "target_precision": metadata.precision,
                        "shots": metadata.shots,
                        "circuit_metadata": pub.circuit.metadata,
                    },
                )
            )
            binding_offset += num_bindings

        return PrimitiveResult(pub_results)

    def _get_job_status(self) -> JobStatus:
        return _TASK_STATUS_MAP[self._task.state()]
