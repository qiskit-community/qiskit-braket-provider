"""Tests for Qiskti to Braket adapter."""
from unittest import TestCase

from qiskit_braket_provider.providers.adapter import (
    qiskit_to_braket_gate_names_mapping,
    qiskit_gate_names_to_braket_gates,
    qiskit_gate_name_to_braket_gate_mapping,
)


class TestAdapter(TestCase):
    """Tests adapter."""

    def test_mappers(self):
        """Tests mappers."""
        self.assertEqual(
            list(sorted(qiskit_to_braket_gate_names_mapping.keys())),
            list(sorted(qiskit_gate_names_to_braket_gates.keys())),
        )

        self.assertEqual(
            list(sorted(qiskit_to_braket_gate_names_mapping.values())),
            list(sorted(qiskit_gate_name_to_braket_gate_mapping.keys())),
        )
