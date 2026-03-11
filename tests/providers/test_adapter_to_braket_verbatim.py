"""Tests for verbatim box support in to_braket() function."""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, BoxOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates
from braket.ir.openqasm import Program

from qiskit_braket_provider.providers.adapter import (
    _extract_verbatim_boxes,
    _restore_verbatim_boxes,
    to_braket,
    to_qiskit,
)


def _make_box_circuit(num_qubits, gates):
    """Create a QuantumCircuit with the given gates applied.

    Args:
        num_qubits: Number of qubits.
        gates: List of (gate_name, qubit_args) tuples.
    """
    qc = QuantumCircuit(num_qubits)
    for gate_name, qubits in gates:
        getattr(qc, gate_name)(*qubits)
    return qc


def _gate_info(braket_circuit):
    """Extract (name, target) list from a Braket circuit."""
    return [(instr.operator.name, instr.target) for instr in braket_circuit.instructions]


# --- TestExtractVerbatimBoxes ---

class TestExtractVerbatimBoxes:
    """Tests for _extract_verbatim_boxes helper function."""

    @pytest.mark.parametrize(
        "inner_gates, expected_gate_names, expected_qubits",
        [
            ([("h", [0]), ("cx", [0, 1])], ["h", "cx"], [0, 1]),
            ([], [], [0, 1]),  # empty verbatim box
        ],
        ids=["single_box_with_gates", "empty_box"],
    )
    def test_verbatim_box_extraction(self, inner_gates, expected_gate_names, expected_qubits):
        inner = _make_box_circuit(2, inner_gates)
        main = QuantumCircuit(2)
        main.append(BoxOp(inner, label="verbatim"), [0, 1])

        modified, boxes = _extract_verbatim_boxes(main, "verbatim")

        assert len(modified.data) == 1
        assert isinstance(modified.data[0].operation, Barrier)
        assert modified.data[0].operation.label == "verbatim"

        assert len(boxes) == 1
        box_circuit, qubit_indices = boxes[0]
        assert [d.operation.name for d in box_circuit.data] == expected_gate_names
        assert qubit_indices == expected_qubits

    def test_multiple_verbatim_boxes_extraction(self):
        inner1 = _make_box_circuit(2, [("h", [0])])
        inner2 = _make_box_circuit(2, [("cx", [0, 1])])

        main = QuantumCircuit(2)
        main.append(BoxOp(inner1, label="verbatim"), [0, 1])
        main.x(1)
        main.append(BoxOp(inner2, label="verbatim"), [0, 1])

        modified, boxes = _extract_verbatim_boxes(main, "verbatim")

        barriers = [i for i in modified.data if isinstance(i.operation, Barrier)]
        assert len(barriers) == 2
        assert all(b.operation.label == "verbatim" for b in barriers)
        assert len([i for i in modified.data if i.operation.name == "x"]) == 1

        assert len(boxes) == 2
        assert boxes[0][0].data[0].operation.name == "h"
        assert boxes[1][0].data[0].operation.name == "cx"

    def test_circuit_without_verbatim_boxes(self):
        main = _make_box_circuit(2, [("h", [0]), ("cx", [0, 1])])
        modified, boxes = _extract_verbatim_boxes(main, "verbatim")

        assert len(modified.data) == 2
        assert [d.operation.name for d in modified.data] == ["h", "cx"]
        assert len(boxes) == 0

    def test_non_verbatim_boxop_not_extracted(self):
        inner = _make_box_circuit(2, [("h", [0])])
        main = QuantumCircuit(2)
        main.append(BoxOp(inner, label="other_label"), [0, 1])

        modified, boxes = _extract_verbatim_boxes(main, "verbatim")

        assert len(boxes) == 0
        assert len(modified.data) == 1
        assert isinstance(modified.data[0].operation, BoxOp)
        assert modified.data[0].operation.label == "other_label"


class TestRestoreVerbatimBoxes:
    """Tests for _restore_verbatim_boxes helper function."""

    def test_single_verbatim_box_restoration(self):
        box_circuit = _make_box_circuit(2, [("h", [0]), ("cx", [0, 1])])
        transpiled = QuantumCircuit(2)
        transpiled.append(Barrier(2, label="verbatim"), [0, 1])

        restored = _restore_verbatim_boxes(transpiled, [(box_circuit, [0, 1])], "verbatim")

        assert len(restored.data) == 2
        assert restored.data[0].operation.name == "h"
        assert restored.data[1].operation.name == "cx"
        assert restored.find_bit(restored.data[0].qubits[0]).index == 0
        assert restored.find_bit(restored.data[1].qubits[0]).index == 0
        assert restored.find_bit(restored.data[1].qubits[1]).index == 1

    def test_multiple_verbatim_boxes_restoration(self):
        box1 = _make_box_circuit(2, [("h", [0])])
        box2 = _make_box_circuit(2, [("cx", [0, 1])])

        transpiled = QuantumCircuit(2)
        transpiled.append(Barrier(2, label="verbatim"), [0, 1])
        transpiled.x(1)
        transpiled.append(Barrier(2, label="verbatim"), [0, 1])

        restored = _restore_verbatim_boxes(
            transpiled, [(box1, [0, 1]), (box2, [0, 1])], "verbatim"
        )

        gate_names = [i.operation.name for i in restored.data]
        assert gate_names == ["h", "x", "cx"]

    @pytest.mark.parametrize(
        "num_barriers, num_boxes, error_match",
        [
            (2, 1, "Found more barriers.*than verbatim boxes"),
            (1, 2, "Found fewer barriers.*than verbatim boxes"),
        ],
        ids=["too_many_barriers", "too_few_barriers"],
    )
    def test_barrier_box_count_mismatch(self, num_barriers, num_boxes, error_match):
        transpiled = QuantumCircuit(2)
        for _ in range(num_barriers):
            transpiled.append(Barrier(2, label="verbatim"), [0, 1])

        boxes = [(_make_box_circuit(2, [("h", [0])]), [0, 1]) for _ in range(num_boxes)]

        with pytest.raises(ValueError, match=error_match):
            _restore_verbatim_boxes(transpiled, boxes, "verbatim")


class TestToBraketIntegration:
    """Integration tests for to_braket with verbatim box support."""

    def test_to_braket_with_single_verbatim_box(self):
        inner = _make_box_circuit(2, [("h", [0]), ("cx", [0, 1])])
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(BoxOp(inner, label="verbatim"), [0, 1])
        qc.y(1)

        bc = to_braket(qc, verbatim=False)
        info = _gate_info(bc)
        names = [n for n, _ in info]

        assert bc.qubit_count == 2
        for expected in ("X", "H", "CNot", "Y"):
            assert expected in names

        indices = {n: next(i for i, (nm, _) in enumerate(info) if nm == n)
                   for n in ("X", "H", "CNot", "Y")}
        assert indices["X"] < indices["H"] < indices["CNot"] < indices["Y"]
        assert info[indices["X"]][1] == [0]
        assert info[indices["H"]][1] == [0]
        assert info[indices["CNot"]][1] == [0, 1]
        assert info[indices["Y"]][1] == [1]

    def test_to_braket_with_multiple_verbatim_boxes(self):
        inner1 = _make_box_circuit(2, [("h", [0])])
        inner2 = _make_box_circuit(2, [("cx", [0, 1])])
        qc = QuantumCircuit(2)
        qc.append(BoxOp(inner1, label="verbatim"), [0, 1])
        qc.x(1)
        qc.append(BoxOp(inner2, label="verbatim"), [0, 1])

        bc = to_braket(qc, verbatim=False)
        info = _gate_info(bc)
        names = [n for n, _ in info]

        assert bc.qubit_count == 2
        indices = {n: next(i for i, (nm, _) in enumerate(info) if nm == n)
                   for n in ("H", "X", "CNot")}
        assert indices["H"] < indices["X"] < indices["CNot"]
        assert info[indices["H"]][1] == [0]
        assert info[indices["X"]][1] == [1]
        assert info[indices["CNot"]][1] == [0, 1]

    def test_to_braket_with_custom_verbatim_box_name(self):
        inner = _make_box_circuit(2, [("h", [0]), ("cx", [0, 1])])
        qc = QuantumCircuit(2)
        qc.append(BoxOp(inner, label="custom_verbatim"), [0, 1])

        bc = to_braket(qc, verbatim=False, verbatim_box_name="custom_verbatim")
        info = _gate_info(bc)
        names = [n for n, _ in info]

        assert bc.qubit_count == 2
        assert "H" in names
        assert "CNot" in names
        h_idx = next(i for i, (n, _) in enumerate(info) if n == "H")
        cnot_idx = next(i for i, (n, _) in enumerate(info) if n == "CNot")
        assert info[h_idx][1] == [0]
        assert info[cnot_idx][1] == [0, 1]
        assert h_idx < cnot_idx

    def test_to_braket_backward_compatibility(self):
        qc = _make_box_circuit(2, [("h", [0]), ("cx", [0, 1])])
        bc = to_braket(qc, verbatim=False)
        info = _gate_info(bc)
        names = [n for n, _ in info]

        assert bc.qubit_count == 2
        assert "H" in names
        assert "CNot" in names
        h_idx = next(i for i, (n, _) in enumerate(info) if n == "H")
        cnot_idx = next(i for i, (n, _) in enumerate(info) if n == "CNot")
        assert info[h_idx][1] == [0]
        assert info[cnot_idx][1] == [0, 1]
        assert h_idx < cnot_idx

    @pytest.mark.parametrize(
        "verbatim, layout_method",
        [
            (True, None),
            (False, None),
            (False, "dense"),
        ],
        ids=["verbatim_true", "trivial_layout", "layout_override"],
    )
    def test_to_braket_verbatim_and_layout_options(self, verbatim, layout_method):
        inner = _make_box_circuit(2, [("h", [0]), ("cx", [0, 1])])
        qc = QuantumCircuit(2)
        qc.append(BoxOp(inner, label="verbatim"), [0, 1])

        kwargs = {"verbatim": verbatim}
        if layout_method:
            kwargs["layout_method"] = layout_method
        bc = to_braket(qc, **kwargs)
        info = _gate_info(bc)
        names = [n for n, _ in info]

        assert bc.qubit_count == 2
        assert "H" in names
        assert "CNot" in names
        h_idx = next(i for i, (n, _) in enumerate(info) if n == "H")
        cnot_idx = next(i for i, (n, _) in enumerate(info) if n == "CNot")
        assert info[h_idx][1] == [0]
        assert info[cnot_idx][1] == [0, 1]
        assert h_idx < cnot_idx

    def test_to_braket_raises_on_pass_manager_with_verbatim_boxes(self):
        inner = _make_box_circuit(2, [("h", [0])])
        qc = QuantumCircuit(2)
        qc.append(BoxOp(inner, label="verbatim"), [0, 1])

        with pytest.raises(ValueError, match="Custom pass_manager is not supported with verbatim boxes"):
            to_braket(qc, verbatim=False, pass_manager=PassManager([Optimize1qGates()]))

    def test_to_braket_raises_on_barrier_labeled_as_verbatim_box(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(Barrier(2, label="verbatim"), [0, 1])
        qc.y(1)

        with pytest.raises(ValueError, match="Cannot have a Barrier labeled with the same label used for verbatim boxes"):
            to_braket(qc, verbatim=False)

    def test_to_braket_with_multiple_circuits_with_verbatim_boxes(self):
        inner1 = _make_box_circuit(2, [("h", [0]), ("cx", [0, 1])])
        qc1 = QuantumCircuit(2)
        qc1.x(0)
        qc1.append(BoxOp(inner1, label="verbatim"), [0, 1])

        inner2 = _make_box_circuit(3, [("h", [0]), ("h", [1]), ("ccx", [0, 1, 2])])
        qc2 = QuantumCircuit(3)
        qc2.append(BoxOp(inner2, label="verbatim"), [0, 1, 2])
        qc2.z(2)

        qc3 = _make_box_circuit(2, [("h", [0]), ("cx", [0, 1])])

        results = to_braket([qc1, qc2, qc3], verbatim=False)

        assert isinstance(results, list)
        assert len(results) == 3
        assert results[0].qubit_count == 2
        assert results[1].qubit_count == 3
        assert results[2].qubit_count == 2


class TestRoundTripConversion:
    """Round-trip conversion tests for Braket → Qiskit → Braket and OpenQASM → Qiskit → Braket."""

    # Shared OpenQASM programs for parameterized round-trip tests
    _SINGLE_BOX_QASM = """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $0;
    cnot $0, $1;
}
x $1;
"""
    _MULTI_BOX_QASM = """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $0;
}
x $1;
#pragma braket verbatim
box {
    cnot $0, $1;
}
"""
    _MIXED_QASM = """
OPENQASM 3.0;
x $0;
#pragma braket verbatim
box {
    h $0;
    cnot $0, $1;
}
y $1;
"""

    @staticmethod
    def _to_qiskit_input(source, use_program):
        """Return either a Program object or raw string for to_qiskit."""
        return Program(source=source) if use_program else source

    @pytest.mark.parametrize("use_program", [True, False], ids=["braket_program", "openqasm"])
    def test_round_trip_single_verbatim_box(self, use_program):
        qc = to_qiskit(self._to_qiskit_input(self._SINGLE_BOX_QASM, use_program))

        box_ops = [i for i in qc.data
                   if hasattr(i.operation, "label") and i.operation.label == "verbatim"]
        assert len(box_ops) == 1

        bc = to_braket(qc, verbatim=False)
        info = _gate_info(bc)
        names = [n for n, _ in info]

        assert bc.qubit_count == 2
        for expected in ("H", "CNot", "X"):
            assert expected in names

        indices = {n: next(i for i, (nm, _) in enumerate(info) if nm == n)
                   for n in ("H", "CNot", "X")}
        assert indices["H"] < indices["X"]
        assert indices["CNot"] < indices["X"]
        assert info[indices["H"]][1] == [0]
        assert info[indices["CNot"]][1] == [0, 1]
        assert info[indices["X"]][1] == [1]

    @pytest.mark.parametrize("use_program", [True, False], ids=["braket_program", "openqasm"])
    def test_round_trip_multiple_verbatim_boxes(self, use_program):
        qc = to_qiskit(self._to_qiskit_input(self._MULTI_BOX_QASM, use_program))

        box_ops = [i for i in qc.data
                   if hasattr(i.operation, "label") and i.operation.label == "verbatim"]
        assert len(box_ops) == 2

        bc = to_braket(qc, verbatim=False)
        info = _gate_info(bc)
        names = [n for n, _ in info]

        assert bc.qubit_count == 2
        indices = {n: next(i for i, (nm, _) in enumerate(info) if nm == n)
                   for n in ("H", "X", "CNot")}
        assert indices["H"] < indices["X"] < indices["CNot"]
        assert info[indices["H"]][1] == [0]
        assert info[indices["X"]][1] == [1]
        assert info[indices["CNot"]][1] == [0, 1]

    @pytest.mark.parametrize("use_program", [True, False], ids=["braket_program", "openqasm"])
    def test_round_trip_custom_verbatim_box_name(self, use_program):
        qasm = """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $0;
    cnot $0, $1;
}
"""
        label = "custom_verbatim" if use_program else "my_custom_verbatim"
        qc = to_qiskit(self._to_qiskit_input(qasm, use_program), verbatim_box_name=label)

        box_ops = [i for i in qc.data
                   if hasattr(i.operation, "label") and i.operation.label == label]
        assert len(box_ops) == 1

        bc = to_braket(qc, verbatim=False, verbatim_box_name=label)
        info = _gate_info(bc)
        names = [n for n, _ in info]

        assert bc.qubit_count == 2
        assert "H" in names
        assert "CNot" in names
        h_idx = next(i for i, (n, _) in enumerate(info) if n == "H")
        cnot_idx = next(i for i, (n, _) in enumerate(info) if n == "CNot")
        assert info[h_idx][1] == [0]
        assert info[cnot_idx][1] == [0, 1]
        assert h_idx < cnot_idx

    @pytest.mark.parametrize("use_program", [True, False], ids=["braket_program", "openqasm"])
    def test_round_trip_mixed_verbatim_and_non_verbatim(self, use_program):
        qc = to_qiskit(self._to_qiskit_input(self._MIXED_QASM, use_program))
        bc = to_braket(qc, verbatim=False)
        info = _gate_info(bc)
        names = [n for n, _ in info]

        assert bc.qubit_count == 2
        assert "H" in names
        assert "CNot" in names

        h_idx = next(i for i, (n, _) in enumerate(info) if n == "H")
        cnot_idx = next(i for i, (n, _) in enumerate(info) if n == "CNot")
        assert info[h_idx][1] == [0]
        assert info[cnot_idx][1] == [0, 1]
        assert h_idx < cnot_idx

    def test_round_trip_multiple_verbatim_boxes_openqasm_3_qubits(self):
        """OpenQASM-specific test with 3 qubits and 2 CNot gates."""
        qasm = """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $0;
}
x $1;
#pragma braket verbatim
box {
    cnot $0, $1;
    cnot $1, $2;
}
"""
        qc = to_qiskit(qasm)
        box_ops = [i for i in qc.data
                   if hasattr(i.operation, "label") and i.operation.label == "verbatim"]
        assert len(box_ops) == 2

        bc = to_braket(qc, verbatim=False)
        info = _gate_info(bc)
        names = [n for n, _ in info]

        assert bc.qubit_count == 3
        h_idx = next(i for i, (n, _) in enumerate(info) if n == "H")
        x_idx = next(i for i, (n, _) in enumerate(info) if n == "X")
        cnot_indices = [i for i, (n, _) in enumerate(info) if n == "CNot"]

        assert h_idx < x_idx
        assert len(cnot_indices) == 2
        assert all(ci > x_idx for ci in cnot_indices)
        assert info[h_idx][1] == [0]
        assert info[x_idx][1] == [1]
        assert info[cnot_indices[0]][1] == [0, 1]
        assert info[cnot_indices[1]][1] == [1, 2]
