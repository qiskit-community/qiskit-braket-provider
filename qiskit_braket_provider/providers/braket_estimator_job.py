"""Job class for BraketEstimator."""

import numpy as np
from qiskit.primitives import BasePrimitiveJob, DataBin, PrimitiveResult, PubResult

from braket.tasks import ProgramSetQuantumTaskResult, QuantumTask


class BraketEstimatorJob(BasePrimitiveJob):
    """
    Job class for BraketEstimatorV2.

    This job wraps a Braket QuantumTask and reconstructs PrimitiveResult
    from the ProgramSetQuantumTaskResult.
    """

    def __init__(self, task: QuantumTask, metadata: dict):
        """
        Initialize the estimator job.

        Args:
            task (QuantumTask): The Braket QuantumTask
            metadata (dict): Metadata needed to reconstruct results, including:
                - pubs: List of EstimatorPub objects
                - pub_metadata: List of metadata dicts for each pub
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
            self._result = self._reconstruct_results(self._task.result())
        return self._result

    def status(self):
        """
        Get the status of the job.

        Returns:
            Job status string.
        """
        return self._task.state()

    def cancel(self):
        """Cancel the job."""
        self._task.cancel()

    def job_id(self) -> str:
        """
        Get the job ID.

        Returns:
            Job ID string.
        """
        return self._task.id

    def done(self) -> bool:
        """
        Check if the job is done.

        Returns:
            True if the job is done, False otherwise.
        """
        state = self._task.state()
        return state in ["COMPLETED", "FAILED", "CANCELLED"]

    def running(self) -> bool:
        """
        Check if the job is running.

        Returns:
            True if the job is running, False otherwise.
        """
        return self._task.state() == "RUNNING"

    def cancelled(self) -> bool:
        """
        Check if the job was cancelled.

        Returns:
            True if the job was cancelled, False otherwise.
        """
        state = self._task.state()
        return state in ["CANCELLED", "CANCELLING"]

    def in_final_state(self) -> bool:
        """
        Check if the job is in a final state.

        Returns:
            True if the job is in a final state, False otherwise.
        """
        state = self._task.state()
        return state in ["COMPLETED", "FAILED", "CANCELLED"]

    def _reconstruct_results(self, task_result: ProgramSetQuantumTaskResult) -> PrimitiveResult:
        """
        Reconstruct PrimitiveResult from Braket task results.

        Args:
            task_result (ProgramSetQuantumTaskResult): The Braket task result.

        Returns:
            PrimitiveResult: PrimitiveResult containing PubResult for each pub.
        """
        metadata = self._metadata
        precision = metadata["precision"]
        shots = metadata["shots"]

        pub_results = []
        binding_offset = 0

        for pub, pub_meta in zip(metadata["pubs"], metadata["pub_metadata"]):
            num_bindings = pub_meta["num_bindings"]
            broadcast_shape = pub_meta["broadcast_shape"]
            binding_map = pub_meta["binding_to_result_map"]

            evs = np.zeros(broadcast_shape, dtype=float)
            for local_binding_idx in range(num_bindings):
                binding_result = task_result[binding_offset + local_binding_idx]
                num_observables = len(binding_result.observables)
                
                for position, obs_idx, param_idx in binding_map[local_binding_idx]:
                    # CircuitBinding returns results organized by parameter sets
                    # For each parameter, we get all observables
                    flat_idx = param_idx * num_observables + obs_idx
                    evs[np.unravel_index(position, broadcast_shape)] = binding_result[flat_idx].expectation

            pub_results.append(
                PubResult(
                    DataBin(evs=evs, shape=broadcast_shape),
                    metadata={
                        "target_precision": precision,
                        "shots": shots,
                        "circuit_metadata": pub.circuit.metadata,
                    },
                )
            )
            binding_offset += num_bindings

        return PrimitiveResult(pub_results)
