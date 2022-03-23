"""Tests for AWS Braket job."""

from unittest import TestCase

from qiskit_braket_plugin.providers import AWSBraketJob, BraketLocalBackend


class TestAWSBraketJob(TestCase):
    """Tests AWSBraketJob."""

    def test_job(self):
        """Tests job."""
        job = AWSBraketJob(BraketLocalBackend(), job_id="AwesomeId")
        self.assertTrue(job)
        self.assertIsNone(job.result())
        self.assertIsNone(job.status())
        self.assertIsNone(job.submit())
        self.assertIsNone(job.cancel())
