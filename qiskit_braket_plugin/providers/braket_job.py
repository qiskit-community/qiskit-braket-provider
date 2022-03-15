"""AWS Braket job."""
from datetime import datetime

from braket.aws import AwsQuantumTask
from braket.tasks import GateModelQuantumTaskResult
from qiskit.providers import JobV1
from qiskit.providers.models import BackendStatus
from qiskit.qobj import QasmQobj, QasmQobjExperiment, QasmQobjInstruction
from typing import List, Optional, Dict, Counter

from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData


def _reverse_and_map(bit_string: str, mapping: Dict[int, int]):
    result_bit_string = len(mapping) * ['x']
    for i, c in enumerate(bit_string):
        if i in mapping:
            result_bit_string[mapping[i]] = c
    # qiskit is Little Endian, braket is Big Endian, so we don't do a re-reversed here
    result = "".join(reversed(result_bit_string))
    return result


def map_measurements(counts: Counter, qasm_experiment: QasmQobjExperiment) -> Dict[str, int]:
    # Need to get measure mapping
    instructions: List[QasmQobjInstruction] = [i for i in qasm_experiment.instructions if i.name == 'measure']
    mapping = dict([(q, m) for i in instructions for q, m in zip(i.qubits, i.memory)])
    mapped_counts = [(_reverse_and_map(k, mapping), v) for k, v in counts.items()]
    keys = set(k for k, _ in mapped_counts)
    new_map = [(key, sum([v for k, v in mapped_counts if k == key])) for key in keys]
    return dict(new_map)


class AWSBraketJob(JobV1):
    """AWSBraketJob."""


    _qobj: QasmQobj
    _job_id: str
    _tasks: List[AwsQuantumTask]
    _backend: 'awsbackend.AWSBackend'

    def __init__(self, job_id: str, qobj: QasmQobj, backend: 'awsbackend.AWSBackend', tasks: List[AwsQuantumTask],
                 extra_data: Optional[dict] = None, s3_bucket: str = None) -> None:
        super().__init__(backend, job_id)
        self._aws_device = backend
        self._tasks = tasks
        self._extra_data = extra_data
        self._date_of_creation = datetime.now()
        self._qobj = qobj
        self._job_id = job_id
        self._s3_bucket = s3_bucket
        self.backend = backend

    @property
    def shots(self) -> int:
        return self._qobj.config.shots

    def submit(self):
        pass

    def result(self, **kwargs):
        print(list(kwargs))

        experiment_results: List[ExperimentResult] = []
        task: AwsQuantumTask
        qasm_experiment: QasmQobjExperiment
        for task, qasm_experiment in zip(self._tasks, self._qobj.experiments):
            result: GateModelQuantumTaskResult = task.result()
            counts: Dict[str, int] = map_measurements(result.measurement_counts, qasm_experiment)
            data = ExperimentResultData(
                counts=dict(counts)
            )
            experiment_result = ExperimentResult(
                shots=self.shots,
                success=task.state() == 'COMPLETED',
                header=qasm_experiment.header,
                status=task.state(),
                data=data
            )
            experiment_results.append(experiment_result)
        qiskit_result = Result(
            backend_name=self._backend,
            # TODO fill
            backend_version=1,
            qobj_id=self._qobj.qobj_id,
            job_id=self._job_id,
            success=self.status(),
            results=experiment_results
        )
        return qiskit_result

    def cancel(self):
        pass

    def status(self):
        status: str = self._aws_device.status
        backend_status: BackendStatus = BackendStatus(
            backend_name=self._aws_device.name,
            #TODO fill
            backend_version=1,
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
