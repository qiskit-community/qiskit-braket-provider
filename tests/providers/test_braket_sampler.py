"""Tests for BraketSampler."""

from unittest import TestCase

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.primitives import BackendSamplerV2
from qiskit.primitives.containers.sampler_pub import SamplerPub

from qiskit_braket_provider.providers import BraketLocalBackend
from qiskit_braket_provider.providers.braket_sampler import BraketSampler


class TestBraketSampler(TestCase):
    """Tests for BraketSampler."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = BraketLocalBackend()
        self.sampler = BraketSampler(self.backend)

    def test_initialization(self):
        """Test sampler initialization."""
        self.assertIsInstance(self.sampler, BraketSampler)
        self.assertEqual(self.sampler._backend, self.backend)

    def test_different_precisions_raises_error(self):
        """Test that pubs with different shots raise an error."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        with self.assertRaises(ValueError) as context:
            self.sampler.run([(qc, [0, 1, 2, 3], 100), (qc, [0, 1, 2, 3], 200)])

        self.assertIn("same shots", str(context.exception))

    def test_run_local(self):
        """Tests that correct results are returned for circuits with multiple registers"""
        theta = Parameter("θ")

        qreg_a = QuantumRegister(9, "qreg_a")
        qreg_b = QuantumRegister(3, "qreg_b")
        creg_a = ClassicalRegister(2, "creg_a")
        creg_b = ClassicalRegister(10, "creg_b")

        chsh_circuit = QuantumCircuit(qreg_a, qreg_b, creg_a, creg_b)
        chsh_circuit.h(0)
        for i in range(11):
            chsh_circuit.cx(i, i + 1)
        chsh_circuit.ry(theta, 0)
        chsh_circuit.measure_all(add_bits=False)
        parameter_values = np.array(  # shape (3, 6)
            [np.linspace(0, 2 * np.pi, 6), np.linspace(0, np.pi, 6), np.linspace(0, np.pi / 2, 6)]
        )
        pub = SamplerPub.coerce((chsh_circuit, parameter_values))

        data = self.sampler.run([pub]).result()[0].data
        data_backend = BackendSamplerV2(backend=self.backend).run([pub]).result()[0].data
        for reg, reg_backend in [
            (data.creg_a, data_backend.creg_a),
            (data.creg_b, data_backend.creg_b),
        ]:
            for index in np.ndindex(pub.shape):
                bit_array = reg[index]
                counts = bit_array.get_int_counts()
                shots = bit_array.num_shots

                bit_array_backend = reg_backend[index]
                counts_backend = bit_array_backend.get_int_counts()
                shots_backend = bit_array_backend.num_shots

                self.assertEqual(counts.keys(), counts_backend.keys())
                for k, v in counts.items():
                    self.assertTrue(
                        np.isclose(v / shots, counts_backend[k] / shots_backend, rtol=0.3, atol=0.2)
                    )
