"""Tests for AWS Braket job."""

from unittest import TestCase

from qiskit.providers import JobStatus

from qiskit_braket_provider.providers import (
    AmazonBraketTask,
    BraketLocalBackend,
    AWSBraketJob,
)
from tests.providers.mocks import MOCK_LOCAL_QUANTUM_TASK


class TestAmazonBraketTask(TestCase):
    """Tests AmazonBraketTask."""

    def _get_task(self):
        return AmazonBraketTask(
            backend=BraketLocalBackend(name="default"),
            task_id="AwesomeId",
            tasks=[MOCK_LOCAL_QUANTUM_TASK],
            shots=10,
        )

    def test_task(self):
        """Tests task."""
        task = self._get_task()

        self.assertTrue(isinstance(task, AmazonBraketTask))
        self.assertEqual(task.shots, 10)

        self.assertEqual(task.status(), JobStatus.DONE)

    def test_result(self):
        """Tests result."""
        task = self._get_task()

        self.assertEqual(task.result().job_id, "AwesomeId")
        self.assertEqual(task.result().results[0].data.counts, {"01": 1, "10": 2})
        self.assertEqual(task.result().results[0].data.memory, ["10", "10", "01"])
        self.assertEqual(task.result().results[0].status, "COMPLETED")
        self.assertEqual(task.result().results[0].shots, 3)
        self.assertEqual(task.result().get_memory(), ["10", "10", "01"])


class TestAWSBraketJob(TestCase):
    """Tests AWSBraketJob"""

    def _get_job(self):
        return AWSBraketJob(
            backend=BraketLocalBackend(name="default"),
            job_id="AwesomeId",
            tasks=[MOCK_LOCAL_QUANTUM_TASK],
            shots=10,
        )

    def test_AWS_job(self):
        """Tests job."""
        job = self._get_job()

        self.assertTrue(isinstance(job, AWSBraketJob))
        self.assertEqual(job.shots, 10)
        
        self.assertEqual(job.result().job_id, "AwesomeId")
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_AWS_result(self):
        """Tests result."""
        job = self._get_job()

        self.assertEqual(job.result().job_id, "AwesomeId")
        self.assertEqual(job.result().results[0].data.counts, {"01": 1, "10": 2})
        self.assertEqual(job.result().results[0].data.memory, ["10", "10", "01"])
        self.assertEqual(job.result().results[0].status, "COMPLETED")
        self.assertEqual(job.result().results[0].shots, 3)
        self.assertEqual(job.result().get_memory(), ["10", "10", "01"])
