"""Job class for BraketSampler."""

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BasePrimitiveJob, BitArray, DataBin, PrimitiveResult, PubResult

from braket.tasks import ProgramSetQuantumTaskResult, QuantumTask


@dataclass
class _MeasureInfo:
    creg_name: str
    num_bits: int
    num_bytes: int
    start: int


class BraketSamplerJob(BasePrimitiveJob):
    """
    Job class for BraketSampler.

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
        task_result: ProgramSetQuantumTaskResult = self._task.result()
        if self._result is None:
            metadata = self._metadata
            shots = metadata["shots"]
            pub_results = []
            for pub_result, pub, pub_meta in zip(
                task_result.entries, metadata["pubs"], metadata["pub_metadata"]
            ):
                circuit = pub.circuit
                meas_info, max_num_bytes = BraketSamplerJob._analyze_circuit(circuit)
                shape = pub_meta["shape"]
                indices = pub_meta["indices"]
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
                                samples[:, start:start + item.num_bits], axis=1, bitorder="little"
                            ),
                            axis=-1
                        )
                pub_results.append(
                    PubResult(
                        DataBin(
                            **{
                                item.creg_name: BitArray(arrays[item.creg_name], item.num_bits)
                                for item in meas_info
                            },
                            shape=shape
                        ),
                        metadata={"shots": shots, "circuit_metadata": circuit.metadata},
                    )
                )
            self._result = PrimitiveResult(pub_results)
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
