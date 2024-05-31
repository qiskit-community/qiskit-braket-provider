"""Tests for Braket jobs"""

from unittest import TestCase

from qiskit_braket_provider.providers.braket_job import AWSBraketJob, AmazonBraketTask


class TestAmazonBraketTask(TestCase):
    """Tests Amazon Braket Task"""

    def test_deprecation_warning_on_init(self):
        with self.assertWarns(DeprecationWarning):
            class SubAmazonBraketTask(AmazonBraketTask):
                pass


class TestAWSBraketJob(TestCase):
    """Tests Amazon Braket Job"""

    def test_deprecation_warning_on_init(self):
        with self.assertWarns(DeprecationWarning):
            class SubAwsBraketJob(AWSBraketJob):
                pass