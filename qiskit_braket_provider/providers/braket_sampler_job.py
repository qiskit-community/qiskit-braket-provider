"""Job class for BraketSampler."""

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BasePrimitiveJob, BitArray, DataBin, PrimitiveResult, PubResult
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.providers import JobStatus

from braket.tasks import QuantumTask
from qiskit_braket_provider.providers.braket_quantum_task import _TASK_STATUS_MAP


@dataclass
class _JobMetadata:
    pubs: list[SamplerPub]
    parameter_indices: list[tuple[int, ...]]
    shots: int


@dataclass
class _MeasureInfo:
    creg_name: str
    num_bits: int
    num_bytes: int
    start: int


class BraketSamplerJob(BasePrimitiveJob[PrimitiveResult[PubResult], JobStatus]):
    """
    Job class for BraketSampler.

    This job wraps a Braket QuantumTask and reconstructs PrimitiveResult
    from the ProgramSetQuantumTaskResult.
    """

    def __init__(self, task: QuantumTask, metadata: _JobMetadata):
        """
        Initialize the estimator job.

        Args:
            task (QuantumTask): The Braket QuantumTask
            metadata (_JobMetadata): Metadata needed to reconstruct results, including:
                - pubs: List of EstimatorPub objects
                - parameter_indices: List of n-dimensional parameter indices
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

        shots = metadata.shots
        pub_results = []
        for pub_result, pub, indices in zip(
            task_result.entries, metadata.pubs, metadata.parameter_indices
        ):
            circuit = pub.circuit
            meas_info, max_num_bytes = BraketSamplerJob._analyze_circuit(circuit)
            shape = pub.shape
            # measurements = np.zeros(shape + (shots, 1), dtype=float)
            # for index, entry in zip(indices, pub_result.entries):
            #     measurements[index] = np.packbits(entry.measurements)
            memory_array = np.array(
                [executable_result.measurements for executable_result in pub_result]
            )
            arrays = {
                item.creg_name: np.zeros(shape + (shots, item.num_bytes), dtype=np.uint8)
                for item in meas_info
            }
            for samples, index in zip(memory_array, indices):
                for item in meas_info:
                    start = item.start
                    arrays[item.creg_name][index] = np.flip(
                        np.packbits(
                            samples[:, start : start + item.num_bits], axis=1, bitorder="little"
                        ),
                        axis=-1,
                    )
            pub_results.append(
                PubResult(
                    DataBin(
                        **{
                            item.creg_name: BitArray(arrays[item.creg_name], item.num_bits)
                            for item in meas_info
                        },
                        shape=shape,
                    ),
                    metadata={"shots": shots, "circuit_metadata": circuit.metadata},
                )
            )
        return PrimitiveResult(pub_results)

    def _get_job_status(self) -> JobStatus:
        return _TASK_STATUS_MAP[self._task.state()]

    @staticmethod
    def _analyze_circuit(circuit: QuantumCircuit) -> tuple[list[_MeasureInfo], int]:
        """Analyzes the information for each creg in a circuit."""
        meas_info = []
        max_num_bits = 0
        for creg in circuit.cregs:
            num_bits = creg.size
            start = circuit.find_bit(creg[0]).index if num_bits != 0 else 0
            meas_info.append(
                _MeasureInfo(
                    creg_name=creg.name,
                    num_bits=num_bits,
                    num_bytes=BraketSamplerJob._min_bytes(num_bits),
                    start=start,
                )
            )
            max_num_bits = max(max_num_bits, start + num_bits)
        return meas_info, BraketSamplerJob._min_bytes(max_num_bits)

    @staticmethod
    def _min_bytes(num_bits: int) -> int:
        """Return the minimum number of bytes needed to store ``num_bits``."""
        return num_bits // 8 + (num_bits % 8 > 0)
