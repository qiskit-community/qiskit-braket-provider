"""Tests for Braket instructions."""

import unittest

from qiskit_braket_provider.providers.braket_instructions import CCPRx, MeasureFF


class TestIqmExperimentalCapabilities(unittest.TestCase):
    """Tests for Braket instructions."""

    def test_measureff_initialization(self):
        """Test MeasureFF initialization with valid parameters"""
        feedback_key = 1
        measure = MeasureFF(feedback_key)

        self.assertEqual(measure.name, "MeasureFF")
        self.assertEqual(measure.num_qubits, 1)
        self.assertEqual(measure.num_clbits, 0)
        self.assertEqual(measure.params, [feedback_key])

    def test_measureff_equality(self):
        """Test MeasureFF equality comparison"""
        measure1 = MeasureFF(1)
        measure2 = MeasureFF(1)
        measure3 = MeasureFF(2)

        self.assertEqual(measure1, measure2)
        self.assertNotEqual(measure1, measure3)
        self.assertNotEqual(measure1, "not_a_measure")

    def test_measureff_repr(self):
        """Test MeasureFF string representation"""
        measure = MeasureFF(1)
        expected_repr = "MeasureFF(feedback_key=1)"
        self.assertEqual(repr(measure), expected_repr)

    def test_ccprx_initialization(self):
        """Test CCPRx initialization with valid parameters"""
        angle1 = 0.5
        angle2 = 0.7
        feedback_key = 1
        ccprx = CCPRx(angle1, angle2, feedback_key)

        self.assertEqual(ccprx.name, "CCPRx")
        self.assertEqual(ccprx.num_qubits, 1)
        self.assertEqual(ccprx.num_clbits, 0)
        self.assertEqual(ccprx.params, [angle1, angle2, feedback_key])

    def test_ccprx_equality(self):
        """Test CCPRx equality comparison"""
        ccprx1 = CCPRx(0.5, 0.7, 1)
        ccprx2 = CCPRx(0.5, 0.7, 1)
        ccprx3 = CCPRx(0.5, 0.7, 2)
        ccprx4 = CCPRx(0.6, 0.7, 1)

        self.assertEqual(ccprx1, ccprx2)
        self.assertNotEqual(ccprx1, ccprx3)
        self.assertNotEqual(ccprx1, ccprx4)
        self.assertNotEqual(ccprx1, "not_a_ccprx")

    def test_ccprx_repr(self):
        """Test CCPRx string representation"""
        ccprx = CCPRx(0.5, 0.7, 1)
        expected_repr = "CCPRx(0.5, 0.7, feedback_key=1)"
        self.assertEqual(repr(ccprx), expected_repr)


if __name__ == "__main__":
    unittest.main()
