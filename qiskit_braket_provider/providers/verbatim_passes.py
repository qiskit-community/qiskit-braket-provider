"""Transpiler passes for preserving Braket verbatim boxes through compilation.

Braket verbatim boxes (``#pragma braket verbatim``) mark circuit regions that
must reach hardware unmodified. In Qiskit these are represented as
:class:`~qiskit.circuit.BoxOp` nodes with a ``"verbatim"`` label.

Since the Qiskit transpiler has no notion of verbatim semantics, these two
passes bracket the transpilation pipeline:

- :class:`ExtractVerbatimBoxes` (pre-transpilation): swaps each verbatim
  ``BoxOp`` for a labeled :class:`~qiskit.circuit.Barrier` that the
  transpiler will leave untouched, and stashes the original circuits in
  ``property_set["verbatim_boxes"]``.

- :class:`RestoreVerbatimBoxes` (post-transpilation): replaces the labeled
  barriers with the stashed gate sequences.

Both passes must share a ``property_set``, which happens automatically when
they live in the same :class:`~qiskit.transpiler.PassManager` or
:class:`~qiskit.transpiler.StagedPassManager`.
"""

from qiskit.circuit import Barrier, BoxOp
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

_BRAKET_VERBATIM_BOX_NAME = "verbatim"


class ExtractVerbatimBoxes(TransformationPass):
    """Swap verbatim ``BoxOp`` nodes for labeled barriers before transpilation.

    The original box circuits are stashed in ``property_set["verbatim_boxes"]``
    as a list of ``QuantumCircuit`` instances for
    :class:`RestoreVerbatimBoxes` to restore afterwards.

    Args:
        verbatim_box_name: Label used to identify verbatim ``BoxOp`` nodes.
    """

    def __init__(self, verbatim_box_name: str = _BRAKET_VERBATIM_BOX_NAME):
        super().__init__()
        self._verbatim_box_name = verbatim_box_name

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Replace matching ``BoxOp`` nodes with labeled barriers.

        Raises:
            ValueError: If the DAG already contains a barrier whose label
                matches ``verbatim_box_name``.
        """
        for node in dag.op_nodes(Barrier):
            if getattr(node.op, "label", None) == self._verbatim_box_name:
                raise ValueError(
                    f"Circuit contains a Barrier with label '{self._verbatim_box_name}' "
                    "which conflicts with the verbatim box label"
                )

        verbatim_boxes = []
        for node in dag.op_nodes(BoxOp):
            if getattr(node.op, "label", None) != self._verbatim_box_name:
                continue
            verbatim_boxes.append(node.op.blocks[0])
            dag.substitute_node(node, Barrier(len(node.qargs), label=self._verbatim_box_name))

        self.property_set["verbatim_boxes"] = verbatim_boxes
        return dag


class RestoreVerbatimBoxes(TransformationPass):
    """Replace labeled barriers with the original verbatim gate sequences.

    Reads ``property_set["verbatim_boxes"]`` populated by
    :class:`ExtractVerbatimBoxes`.

    Args:
        verbatim_box_name: Label used to identify placeholder barriers.
    """

    def __init__(self, verbatim_box_name: str = _BRAKET_VERBATIM_BOX_NAME):
        super().__init__()
        self._verbatim_box_name = verbatim_box_name

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Splice stashed gate sequences back in place of labeled barriers.

        Raises:
            ValueError: If the number of labeled barriers does not match
                the number of stashed verbatim boxes.
        """
        verbatim_boxes = self.property_set.get("verbatim_boxes", [])
        if not verbatim_boxes:
            return dag

        barrier_nodes = [
            node
            for node in dag.op_nodes(Barrier)
            if getattr(node.op, "label", None) == self._verbatim_box_name
        ]

        if len(barrier_nodes) > len(verbatim_boxes):
            raise ValueError(
                f"Compiler error while processing verbatim boxes. "
                f"Illegal barriers with label '{self._verbatim_box_name}'"
            )
        if len(barrier_nodes) < len(verbatim_boxes):
            raise ValueError(
                f"Compiler error while processing verbatim boxes. "
                f"Expected {len(barrier_nodes)} verbatim boxes, "
                f"but found {len(verbatim_boxes)}."
            )

        for node, box_circuit in zip(barrier_nodes, verbatim_boxes):
            box_dag = circuit_to_dag(box_circuit)
            wires = dict(zip(box_dag.qubits, node.qargs))
            dag.substitute_node_with_dag(node, box_dag, wires=wires)

        return dag
