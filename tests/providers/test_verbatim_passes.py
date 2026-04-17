"""Tests for ExtractVerbatimBoxes and RestoreVerbatimBoxes transpiler passes."""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, BoxOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_braket_provider.providers.verbatim_passes import (
    ExtractVerbatimBoxes,
    RestoreVerbatimBoxes,
)

VERBATIM_LABEL = "verbatim"
NUM_QUBITS = 2
QUBIT_PAIR = [0, 1]


def _make_circuit(num_qubits: int, gates: list[tuple[str, list[int]]]) -> QuantumCircuit:
    """Create a QuantumCircuit with the given gates applied.

    Args:
        num_qubits: Number of qubits.
        gates: List of (gate_name, qubit_args) tuples.
    """
    qc = QuantumCircuit(num_qubits)
    for gate_name, qubits in gates:
        getattr(qc, gate_name)(*qubits)
    return qc


def _gate_info(circuit: QuantumCircuit) -> list[tuple[str, list[int]]]:
    """Extract (name, qubit_indices) for each instruction in a circuit."""
    return [
        (inst.operation.name, [circuit.find_bit(q).index for q in inst.qubits])
        for inst in circuit.data
    ]


@pytest.fixture
def h_cx_circuit():
    """2-qubit circuit with H on q0 and CX on q0,q1."""
    return _make_circuit(NUM_QUBITS, [("h", [0]), ("cx", [0, 1])])


@pytest.fixture
def h_circuit():
    """2-qubit circuit with H on q0."""
    return _make_circuit(NUM_QUBITS, [("h", [0])])


@pytest.fixture
def cx_circuit():
    """2-qubit circuit with CX on q0,q1."""
    return _make_circuit(NUM_QUBITS, [("cx", [0, 1])])


def test_extract_single_box(h_cx_circuit):
    """Single verbatim BoxOp is replaced with a labeled barrier on the same qubits."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(BoxOp(h_cx_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)

    extract = ExtractVerbatimBoxes()
    result = extract(qc)

    assert _gate_info(result) == [("barrier", QUBIT_PAIR)]
    assert isinstance(result.data[0].operation, Barrier)
    assert result.data[0].operation.label.startswith(VERBATIM_LABEL)

    boxes = extract.property_set["verbatim_boxes"]
    assert len(boxes) == 1
    assert _gate_info(next(iter(boxes.values()))) == [("h", [0]), ("cx", [0, 1])]


def test_extract_multiple_boxes_preserves_interleaved_gates(h_circuit, cx_circuit):
    """Multiple verbatim BoxOps are extracted while non-verbatim gates stay in place."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(BoxOp(h_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)
    qc.x(1)
    qc.append(BoxOp(cx_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)

    extract = ExtractVerbatimBoxes()
    result = extract(qc)

    assert _gate_info(result) == [("barrier", QUBIT_PAIR), ("x", [1]), ("barrier", QUBIT_PAIR)]
    assert all(
        isinstance(result.data[i].operation, Barrier)
        and result.data[i].operation.label.startswith(VERBATIM_LABEL)
        for i in [0, 2]
    )
    assert result.data[0].operation.label != result.data[2].operation.label

    box_list = list(extract.property_set["verbatim_boxes"].values())
    assert _gate_info(box_list[0]) == [("h", [0])]
    assert _gate_info(box_list[1]) == [("cx", [0, 1])]


def test_extract_only_matches_configured_label(h_circuit):
    """BoxOps with non-matching labels are left untouched; custom labels are matched."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(BoxOp(h_circuit, label="other"), QUBIT_PAIR)
    qc.append(BoxOp(h_circuit, label="custom"), QUBIT_PAIR)

    extract = ExtractVerbatimBoxes("custom")
    result = extract(qc)

    assert isinstance(result.data[0].operation, BoxOp)
    assert result.data[0].operation.label == "other"
    assert isinstance(result.data[1].operation, Barrier)
    assert result.data[1].operation.label.startswith("custom")
    assert len(extract.property_set["verbatim_boxes"]) == 1


def test_extract_raises_on_barrier_label_collision():
    """ExtractVerbatimBoxes rejects circuits with barriers using the verbatim label."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.barrier(QUBIT_PAIR, label=VERBATIM_LABEL)

    with pytest.raises(ValueError, match="conflicts with the verbatim box label"):
        ExtractVerbatimBoxes()(qc)


def test_extract_empty_box():
    """An empty verbatim BoxOp produces an empty stashed circuit."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(BoxOp(QuantumCircuit(NUM_QUBITS), label=VERBATIM_LABEL), QUBIT_PAIR)

    extract = ExtractVerbatimBoxes()
    extract(qc)

    assert next(iter(extract.property_set["verbatim_boxes"].values())).data == []


def test_restore_single_box(h_cx_circuit):
    """A labeled barrier is replaced with the exact stashed gate sequence and qubits."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.barrier(QUBIT_PAIR, label=f"{VERBATIM_LABEL}__0")

    restore = RestoreVerbatimBoxes()
    restore.property_set["verbatim_boxes"] = {f"{VERBATIM_LABEL}__0": h_cx_circuit}
    result = restore(qc)

    assert _gate_info(result) == [("h", [0]), ("cx", [0, 1])]


def test_restore_multiple_boxes_in_order(h_circuit, cx_circuit):
    """Multiple barriers are restored in order with interleaved gates preserved."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.barrier(QUBIT_PAIR, label=f"{VERBATIM_LABEL}__0")
    qc.x(1)
    qc.barrier(QUBIT_PAIR, label=f"{VERBATIM_LABEL}__1")

    restore = RestoreVerbatimBoxes()
    restore.property_set["verbatim_boxes"] = {
        f"{VERBATIM_LABEL}__0": h_circuit,
        f"{VERBATIM_LABEL}__1": cx_circuit,
    }
    result = restore(qc)

    assert _gate_info(result) == [("h", [0]), ("x", [1]), ("cx", [0, 1])]


def test_restore_raises_on_count_mismatch(h_circuit):
    """RestoreVerbatimBoxes raises when stashed boxes outnumber barriers."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.barrier(QUBIT_PAIR, label=f"{VERBATIM_LABEL}__0")

    restore = RestoreVerbatimBoxes()
    restore.property_set["verbatim_boxes"] = {
        f"{VERBATIM_LABEL}__0": h_circuit,
        f"{VERBATIM_LABEL}__1": h_circuit,
    }

    with pytest.raises(RuntimeError, match="Internal error"):
        restore(qc)


def test_restore_no_op_when_no_boxes():
    """RestoreVerbatimBoxes is a no-op when property_set has no verbatim boxes."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.h(0)

    restore = RestoreVerbatimBoxes()
    result = restore(qc)

    assert _gate_info(result) == [("h", [0])]


def test_round_trip_preserves_exact_sequence(h_cx_circuit):
    """Extract->restore round-trip produces the exact original gate sequence and qubits."""
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.append(BoxOp(h_cx_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)
    qc.y(2)

    result = PassManager([ExtractVerbatimBoxes(), RestoreVerbatimBoxes()]).run(qc)

    assert _gate_info(result) == [("x", [0]), ("h", [0]), ("cx", [0, 1]), ("y", [2])]


def test_staged_pass_manager_protects_verbatim_decomposes_rest(h_cx_circuit):
    """Verbatim h+cx survive on exact qubits while non-verbatim h is fully decomposed."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.h(0)
    qc.append(BoxOp(h_cx_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)
    qc.h(1)

    pm = generate_preset_pass_manager(
        optimization_level=0,
        basis_gates=["rz", "sx", "cz"],
    )
    pm.pre_init = PassManager([ExtractVerbatimBoxes()])
    pm.post_optimization = PassManager([RestoreVerbatimBoxes()])

    result = pm.run(qc)
    info = _gate_info(result)

    assert ("h", [0]) in info
    assert ("cx", [0, 1]) in info
    assert sum(1 for name, _ in info if name == "h") == 1
    basis = {"rz", "sx", "cz", "h", "cx"}
    assert all(name in basis for name, _ in info)


def test_is_verbatim_label_none():
    """_is_verbatim_label returns False for None labels."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.barrier(QUBIT_PAIR)

    extract = ExtractVerbatimBoxes()
    result = extract(qc)

    assert _gate_info(result) == [("barrier", QUBIT_PAIR)]


def test_restore_skips_non_verbatim_barriers(h_cx_circuit):
    """Non-verbatim barriers are left untouched during restore."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.barrier(QUBIT_PAIR)
    qc.barrier(QUBIT_PAIR, label=f"{VERBATIM_LABEL}__0")

    restore = RestoreVerbatimBoxes()
    restore.property_set["verbatim_boxes"] = {f"{VERBATIM_LABEL}__0": h_cx_circuit}
    result = restore(qc)

    info = _gate_info(result)
    assert info == [("barrier", QUBIT_PAIR), ("h", [0]), ("cx", [0, 1])]


def test_restore_raises_on_unexpected_verbatim_barrier():
    """Barrier with verbatim label not in stashed dict raises."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.barrier(QUBIT_PAIR, label=f"{VERBATIM_LABEL}__5")

    restore = RestoreVerbatimBoxes()
    restore.property_set["verbatim_boxes"] = {f"{VERBATIM_LABEL}__0": QuantumCircuit(NUM_QUBITS)}

    with pytest.raises(RuntimeError, match="Internal error"):
        restore(qc)
