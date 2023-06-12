"""Tests for AWS Braket job."""

from unittest import TestCase

from qiskit.providers import JobStatus

from qiskit_braket_provider.providers import AmazonBraketTask, BraketLocalBackend
from tests.providers.mocks import MOCK_LOCAL_QUANTUM_TASK


class TestAmazonBraketTask(TestCase):
    """Tests AmazonBraketTask."""

    def _get_job(self):
        return AmazonBraketTask(
            backend=BraketLocalBackend(name="default"),
            task_id="AwesomeId",
            tasks=[MOCK_LOCAL_QUANTUM_TASK],
            shots=10,
        )

    def test_job(self):
        """Tests job."""
        job = self._get_job()

        self.assertTrue(isinstance(job, AmazonBraketTask))
        self.assertEqual(job.shots, 10)

        self.assertEqual(job.status(), JobStatus.DONE)

    def test_result(self):
        """Tests result."""
        job = self._get_job()

        self.assertEqual(job.result().task_id, "AwesomeId")
        self.assertEqual(job.result().results[0].data.counts, {"01": 1, "10": 2})
        self.assertEqual(job.result().results[0].data.memory, ["10", "10", "01"])
        self.assertEqual(job.result().results[0].status, "COMPLETED")
        self.assertEqual(job.result().results[0].shots, 3)
        self.assertEqual(job.result().get_memory(), ["10", "10", "01"])
