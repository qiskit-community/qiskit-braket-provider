"""Tests for AWS Braket provider."""


from unittest import TestCase

from qiskit_braket_plugin.providers import AWSBraketProvider


class TestAWSBraketProvider(TestCase):
    """Tests AWSBraketProvider."""

    def test_provider(self):
        """Tests provider."""
        provider = AWSBraketProvider()
        self.assertTrue(provider)
        self.assertIsInstance(provider.backends(), list)
