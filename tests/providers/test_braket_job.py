"""Tests for AWS Braket job."""

from unittest import TestCase

from qiskit_braket_plugin.providers import AWSBraketJob, AWSBraketLocalBackend
from tests.providers.mocks import MOCK_LOCAL_QUANTUM_TASK


class TestAWSBraketJob(TestCase):
    """Tests AWSBraketJob."""

    def test_job(self):
        """Tests job."""

        job = AWSBraketJob(
                backend=AWSBraketLocalBackend(name="default"),
                job_id="AwesomeId",
                tasks=[MOCK_LOCAL_QUANTUM_TASK],
                shots=100
                )

        self.assertTrue(job)

        self.assertTrue(job.result().job_id, "AwesomeId")
        self.assertTrue(job.result().results[0].data.counts, {'00': 1})
        self.assertTrue(job.result().results[0].shots, 100)
        self.assertTrue(job.result().results[0].status, "COMPLETED")

        self.assertTrue(job.status().status_msg, "AVAILABLE")
