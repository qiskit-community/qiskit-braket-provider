"""Tests for BraketSampler."""

from unittest import TestCase

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.primitives import BackendSamplerV2
from qiskit.primitives.containers.sampler_pub import SamplerPub

from braket.circuits import Circuit
from qiskit_braket_provider.providers import BraketLocalBackend
from qiskit_braket_provider.providers.braket_sampler import BraketSampler


class TestBraketSampler(TestCase):
    """Tests for BraketSampler."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = BraketLocalBackend()
        self.sampler = BraketSampler(self.backend)

    def test_program_sets_unsupported(self):
        """Tests that initialization raises a ValueError if program sets aren't supported"""
        backend = BraketLocalBackend()
        backend._supports_program_sets = False
        with self.assertRaises(ValueError):
            BraketSampler(backend)

    def test_different_precisions_raises_error(self):
        """Test that pubs with different shots raise an error."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        with self.assertRaises(ValueError) as context:
            self.sampler.run([(qc, [0, 1, 2, 3], 100), (qc, [0, 1, 2, 3], 200)])

        self.assertIn("same shots", str(context.exception))

    def test_run_local_multiple_registers(self):
        """Tests that correct results are returned for circuits with multiple registers"""
        circuit = QuantumCircuit(
            QuantumRegister(3, "qreg_a"),
            QuantumRegister(9, "qreg_b"),
            ClassicalRegister(10, "creg_a"),
            ClassicalRegister(2, "creg_b"),
        )
        circuit.h(0)
        for i in range(11):
            circuit.cx(i, i + 1)
        circuit.ry(Parameter("θ"), 0)
        circuit.measure_all(add_bits=False)

        num_steps = 6
        pub = (
            circuit,
            np.array(  # shape (3, 6)
                [
                    np.linspace(0, 2 * np.pi, num_steps),
                    np.linspace(0, np.pi, num_steps),
                    np.linspace(np.pi, 2 * np.pi, num_steps),
                ]
            ),
        )
        coerced = SamplerPub.coerce(pub)

        task = self.sampler.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), coerced.size)
        data = task.result()[0].data
        data_backend = BackendSamplerV2(backend=self.backend).run([pub]).result()[0].data
        for reg, reg_backend in [
            (data.creg_a, data_backend.creg_a),
            (data.creg_b, data_backend.creg_b),
        ]:
            for index in np.ndindex(coerced.shape):
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

    def test_run_local_no_parameters(self):
        """Tests that correct results are returned for circuits with no parameters"""
        qc1 = QuantumCircuit(2)
        qc1.id(0)
        qc1.x(0)
        qc1.measure_all()
        qc2 = QuantumCircuit(3)
        qc2.h(0)
        qc2.id(0)
        qc2.x(0)
        qc2.measure_all()

        task = self.sampler.run([(qc1,), (qc2,)])
        program_set = task.program_set
        self.assertEqual(len(program_set), 2)
        self.assertTrue(isinstance(circ, Circuit) for circ in program_set)
        for actual, expected in zip(
            task.result(), BackendSamplerV2(backend=self.backend).run([(qc1,), (qc2,)]).result()
        ):
            bit_array = actual.data.meas
            counts = bit_array.get_int_counts()
            shots = bit_array.num_shots

            bit_array_backend = expected.data.meas
            counts_backend = bit_array_backend.get_int_counts()
            shots_backend = bit_array_backend.num_shots

            self.assertEqual(counts.keys(), counts_backend.keys())
            for k, v in counts.items():
                self.assertTrue(
                    np.isclose(v / shots, counts_backend[k] / shots_backend, rtol=0.3, atol=0.2)
                )
