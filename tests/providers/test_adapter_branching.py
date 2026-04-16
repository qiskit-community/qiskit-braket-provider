"""Tests for branching statement (if/else) support in the Qiskit adapter."""

import pytest
from qiskit.circuit import Clbit, IfElseOp
from qiskit.circuit.library import CXGate, HGate, Measure, XGate, YGate, ZGate
from qiskit.transpiler import Target

from braket.default_simulator.openqasm.parser.openqasm_ast import (
    ArrayLiteral,
    BooleanLiteral,
    BoolType,
    Cast,
    FunctionCall,
    IntegerLiteral,
    UnaryExpression,
    UnaryOperator,
)
from qiskit_braket_provider import to_qiskit
from qiskit_braket_provider.providers.adapter import _compile, _QiskitProgramContext


def _get_if_else_ops(circuit):
    """Extract all IfElseOp instructions from a circuit."""
    return [instr for instr in circuit.data if isinstance(instr.operation, IfElseOp)]


def _get_ops_with_qubits(circuit):
    """Get (gate_name, qubit_indices) tuples for all non-measure ops in a circuit."""
    return [
        (instr.operation.name, [circuit.find_bit(q).index for q in instr.qubits])
        for instr in circuit.data
        if instr.operation.name != "measure"
    ]


@pytest.mark.parametrize(
    "qasm, expected_true_ops, expected_false_ops",
    [
        pytest.param(
            """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0] == 1) {
    h q[1];
} else {
    x q[1];
}
""",
            [("h", [1])],
            [("x", [1])],
            id="if_else_with_single_gates",
        ),
        pytest.param(
            """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0] == 1) {
    h q[1];
    x q[0];
} else {
    y q[0];
    z q[1];
}
""",
            [("h", [1]), ("x", [0])],
            [("y", [0]), ("z", [1])],
            id="if_else_with_multiple_gates_different_qubits",
        ),
        pytest.param(
            """
OPENQASM 3.0;
qubit[3] q;
bit[1] c;
c[0] = measure q[0];
if (c[0] == 0) {
    h q[1];
    x q[2];
} else {
    y q[0];
}
""",
            [("h", [1]), ("x", [2])],
            [("y", [0])],
            id="asymmetric_branches_different_qubits",
        ),
    ],
)
def test_if_else_branch_bodies(qasm, expected_true_ops, expected_false_ops):
    qc = to_qiskit(qasm)
    if_else_ops = _get_if_else_ops(qc)
    assert len(if_else_ops) == 1

    op = if_else_ops[0].operation
    true_body, false_body = op.params

    assert _get_ops_with_qubits(true_body) == expected_true_ops
    assert _get_ops_with_qubits(false_body) == expected_false_ops


def test_if_only_no_else():
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0] == 1) {
    h q[1];
}
"""
    qc = to_qiskit(qasm)
    if_else_ops = _get_if_else_ops(qc)
    assert len(if_else_ops) == 1

    op = if_else_ops[0].operation
    true_body, false_body = op.params

    assert _get_ops_with_qubits(true_body) == [("h", [1])]
    assert false_body is None


@pytest.mark.parametrize(
    "condition_value, expected_value",
    [
        ("1", 1),
        ("0", 0),
    ],
    ids=["condition_1", "condition_0"],
)
def test_condition_value(condition_value, expected_value):
    qasm = f"""
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0] == {condition_value}) {{
    h q[1];
}}
"""
    qc = to_qiskit(qasm)
    op = _get_if_else_ops(qc)[0].operation
    clbit, value = op.condition
    assert isinstance(clbit, Clbit)
    assert qc.clbits.index(clbit) == 0
    assert value == expected_value


def test_condition_references_correct_clbit():
    """When multiple bit variables are declared, the condition should reference the right clbit."""
    qasm = """
OPENQASM 3.0;
qubit[3] q;
bit[2] a;
bit[2] b;
a[0] = measure q[0];
b[1] = measure q[1];
if (b[1] == 1) {
    h q[2];
}
"""
    qc = to_qiskit(qasm)
    instr = _get_if_else_ops(qc)[0]
    op = instr.operation
    clbit, value = op.condition

    # b starts at offset 2 (after a's 2 bits), b[1] is clbit index 3
    assert qc.clbits.index(clbit) == 3
    assert value == 1

    # IfElseOp should span all qubits and clbits
    assert [qc.find_bit(q).index for q in instr.qubits] == [0, 1, 2]
    assert [qc.find_bit(c).index for c in instr.clbits] == [0, 1, 2, 3]

    # Branch body gate targets correct qubit
    true_body = op.params[0]
    assert _get_ops_with_qubits(true_body) == [("h", [2])]


def test_if_else_circuit_dimensions():
    """Branch body circuits should have the same qubit/clbit counts as the main circuit."""
    qasm = """
OPENQASM 3.0;
qubit[3] q;
bit[2] c;
c[0] = measure q[0];
if (c[0] == 1) {
    h q[1];
} else {
    x q[2];
}
"""
    qc = to_qiskit(qasm)
    instr = _get_if_else_ops(qc)[0]
    op = instr.operation
    true_body, false_body = op.params

    assert true_body.num_qubits == qc.num_qubits
    assert true_body.num_clbits == qc.num_clbits
    assert false_body.num_qubits == qc.num_qubits
    assert false_body.num_clbits == qc.num_clbits

    assert _get_ops_with_qubits(true_body) == [("h", [1])]
    assert _get_ops_with_qubits(false_body) == [("x", [2])]

    # IfElseOp spans all qubits/clbits on the main circuit
    assert [qc.find_bit(q).index for q in instr.qubits] == [0, 1, 2]
    assert [qc.find_bit(c).index for c in instr.clbits] == [0, 1]


def test_gates_before_and_after_branch():
    """Gates outside the branch should appear on the main circuit, not inside the branch."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
x q[0];
c[0] = measure q[0];
if (c[0] == 1) {
    h q[1];
}
y q[1];
"""
    qc = to_qiskit(qasm)

    main_ops = [(instr.operation.name, [qc.find_bit(q).index for q in instr.qubits]) for instr in qc.data]
    assert main_ops[0] == ("x", [0])
    assert main_ops[1] == ("measure", [0])
    assert main_ops[2][0] == "if_else"
    assert main_ops[2][1] == [0, 1]
    assert main_ops[3] == ("y", [1])


def test_multiple_branches():
    """Multiple sequential if/else blocks should each produce their own IfElseOp."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
c[0] = measure q[0];
if (c[0] == 1) {
    h q[1];
}
c[1] = measure q[1];
if (c[1] == 1) {
    x q[0];
}
"""
    qc = to_qiskit(qasm)
    if_else_ops = _get_if_else_ops(qc)
    assert len(if_else_ops) == 2

    # First branch: condition on c[0], h on q[1]
    op0 = if_else_ops[0].operation
    assert qc.clbits.index(op0.condition[0]) == 0
    assert _get_ops_with_qubits(op0.params[0]) == [("h", [1])]

    # Second branch: condition on c[1], x on q[0]
    op1 = if_else_ops[1].operation
    assert qc.clbits.index(op1.condition[0]) == 1
    assert _get_ops_with_qubits(op1.params[0]) == [("x", [0])]


def test_for_loop_before_measurement_works():
    """For loops that appear before any measurement should work normally."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
for int[8] i in [0:1] {
    h q[0];
}
c[0] = measure q[0];
"""
    qc = to_qiskit(qasm)
    main_ops = [(instr.operation.name, [qc.find_bit(q).index for q in instr.qubits]) for instr in qc.data]
    assert main_ops[0] == ("h", [0])
    assert main_ops[1] == ("h", [0])
    assert main_ops[2] == ("measure", [0])


def test_for_loop_after_unrelated_measurement_works():
    """A for loop after a measurement on a different qubit should work."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
for int[8] i in [0:1] {
    h q[1];
}
"""
    qc = to_qiskit(qasm)
    main_ops = [(instr.operation.name, [qc.find_bit(q).index for q in instr.qubits]) for instr in qc.data]
    assert main_ops[0] == ("measure", [0])
    assert main_ops[1] == ("h", [1])
    assert main_ops[2] == ("h", [1])


def test_mcm_branch_inside_for_loop():
    """An MCM branching statement inside a for loop should produce IfElseOps."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
for int[8] i in [0:1] {
    if (c[0] == 1) {
        h q[1];
    } else {
        x q[1];
    }
}
"""
    qc = to_qiskit(qasm)
    if_else_ops = _get_if_else_ops(qc)
    assert len(if_else_ops) == 2

    for op_instr in if_else_ops:
        true_body, false_body = op_instr.operation.params
        assert _get_ops_with_qubits(true_body) == [("h", [1])]
        assert _get_ops_with_qubits(false_body) == [("x", [1])]
        assert qc.clbits.index(op_instr.operation.condition[0]) == 0


@pytest.fixture
def mcm_target():
    t = Target(num_qubits=3)
    t.add_instruction(HGate())
    t.add_instruction(XGate())
    t.add_instruction(YGate())
    t.add_instruction(ZGate())
    t.add_instruction(CXGate())
    t.add_instruction(Measure())
    t.add_instruction(IfElseOp, name="if_else")
    return t


@pytest.mark.parametrize(
    "qasm, expected_if_else_count",
    [
        pytest.param(
            """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0] == 1) {
    h q[1];
} else {
    x q[1];
}
""",
            1,
            id="single_if_else",
        ),
        pytest.param(
            """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
c[0] = measure q[0];
if (c[0] == 1) {
    h q[1];
}
c[1] = measure q[1];
if (c[1] == 1) {
    x q[0];
}
""",
            2,
            id="multiple_if_else",
        ),
        pytest.param(
            """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0] == 1) {
    h q[1];
}
""",
            1,
            id="if_only_no_else",
        ),
    ],
)
def test_compile_preserves_if_else_ops(qasm, expected_if_else_count, mcm_target):
    """IfElseOps should survive compilation through _compile."""
    result = _compile(qasm, target=mcm_target)
    compiled_circuit = result.circuits[0]
    if_else_ops = [instr for instr in compiled_circuit.data if isinstance(instr.operation, IfElseOp)]
    assert len(if_else_ops) == expected_if_else_count


def test_compile_if_else_branch_bodies_intact(mcm_target):
    """Branch body gate content and qubit targets should be preserved after compilation."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0] == 1) {
    h q[1];
} else {
    x q[1];
}
"""
    result = _compile(qasm, target=mcm_target)
    compiled_circuit = result.circuits[0]
    op = next(instr for instr in compiled_circuit.data if isinstance(instr.operation, IfElseOp)).operation
    true_body, false_body = op.params

    assert _get_ops_with_qubits(true_body) == [("h", [1])]
    assert _get_ops_with_qubits(false_body) == [("x", [1])]


def test_compile_if_else_condition_preserved(mcm_target):
    """The condition clbit and value should be preserved after compilation."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0] == 1) {
    h q[1];
}
"""
    result = _compile(qasm, target=mcm_target)
    compiled_circuit = result.circuits[0]
    instr = next(i for i in compiled_circuit.data if isinstance(i.operation, IfElseOp))
    op = instr.operation
    clbit, value = op.condition
    assert isinstance(clbit, Clbit)
    assert compiled_circuit.clbits.index(clbit) == 0
    assert value == 1

    # Verify the IfElseOp spans the right qubits on the compiled circuit
    qubit_indices = [compiled_circuit.find_bit(q).index for q in instr.qubits]
    assert 0 in qubit_indices
    assert 1 in qubit_indices


def test_static_true_condition_takes_if_branch():
    """A static true condition should execute only the if block."""
    qasm = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (1 == 1) {
    h q[0];
} else {
    x q[0];
}
c[0] = measure q[0];
"""
    qc = to_qiskit(qasm)
    ops = _get_ops_with_qubits(qc)
    assert ("h", [0]) in ops
    assert ("x", [0]) not in ops


def test_static_false_condition_takes_else_branch():
    """A static false condition should execute only the else block."""
    qasm = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (0 == 1) {
    h q[0];
} else {
    x q[0];
}
c[0] = measure q[0];
"""
    qc = to_qiskit(qasm)
    ops = _get_ops_with_qubits(qc)
    assert ("x", [0]) in ops
    assert ("h", [0]) not in ops


def test_while_loop_before_measurement():
    """A while loop with a static condition should execute normally."""
    qasm = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
int[8] i = 0;
while (i < 2) {
    h q[0];
    i += 1;
}
c[0] = measure q[0];
"""
    qc = to_qiskit(qasm)
    h_count = sum(1 for name, _ in _get_ops_with_qubits(qc) if name == "h")
    assert h_count == 2


def test_for_loop_with_break():
    qasm = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
for int[8] i in {0, 1, 2} {
    h q[0];
    break;
}
c[0] = measure q[0];
"""
    qc = to_qiskit(qasm)
    h_count = sum(1 for name, _ in _get_ops_with_qubits(qc) if name == "h")
    assert h_count == 1


def test_for_loop_with_continue():
    qasm = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
for int[8] i in {0, 1} {
    continue;
    h q[0];
}
c[0] = measure q[0];
"""
    qc = to_qiskit(qasm)
    h_count = sum(1 for name, _ in _get_ops_with_qubits(qc) if name == "h")
    assert h_count == 0


def test_condition_literal_on_lhs():
    """Condition with literal on the left side: if (1 == c[0])."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (1 == c[0]) {
    h q[1];
}
"""
    qc = to_qiskit(qasm)
    op = _get_if_else_ops(qc)[0].operation
    clbit, value = op.condition
    assert qc.clbits.index(clbit) == 0
    assert value == 1


def test_condition_bare_identifier():
    """Condition on a single-bit variable without indexing: if (c == 1)."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit c;
c = measure q[0];
if (c == 1) {
    h q[1];
}
"""
    qc = to_qiskit(qasm)
    op = _get_if_else_ops(qc)[0].operation
    clbit, value = op.condition
    assert qc.clbits.index(clbit) == 0
    assert value == 1
    assert _get_ops_with_qubits(op.params[0]) == [("h", [1])]


def test_while_loop_with_break():
    qasm = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
int[8] i = 0;
while (i < 5) {
    h q[0];
    i += 1;
    break;
}
c[0] = measure q[0];
"""
    qc = to_qiskit(qasm)
    h_count = sum(1 for name, _ in _get_ops_with_qubits(qc) if name == "h")
    assert h_count == 1


def test_while_loop_with_continue():
    qasm = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
int[8] i = 0;
while (i < 2) {
    i += 1;
    continue;
    h q[0];
}
c[0] = measure q[0];
"""
    qc = to_qiskit(qasm)
    h_count = sum(1 for name, _ in _get_ops_with_qubits(qc) if name == "h")
    assert h_count == 0


def test_resolve_clbit_index_unsupported_type():
    ctx = _QiskitProgramContext()
    with pytest.raises(TypeError, match="Unsupported condition operand type"):
        ctx._resolve_clbit_index(IntegerLiteral(value=0))


def test_bare_indexed_identifier_condition():
    """Condition `if (c[0])` should be treated as `if (c[0] == 1)`."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0]) {
    h q[1];
} else {
    x q[1];
}
"""
    qc = to_qiskit(qasm)
    if_else_ops = _get_if_else_ops(qc)
    assert len(if_else_ops) == 1

    op = if_else_ops[0].operation
    clbit, value = op.condition
    assert qc.clbits.index(clbit) == 0
    assert value == 1

    true_body, false_body = op.params
    assert _get_ops_with_qubits(true_body) == [("h", [1])]
    assert _get_ops_with_qubits(false_body) == [("x", [1])]


def test_bare_identifier_condition_mcm():
    """Condition `if (c)` on a single-bit variable should be treated as `if (c == 1)`."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit c;
c = measure q[0];
if (c) {
    h q[1];
}
"""
    qc = to_qiskit(qasm)
    if_else_ops = _get_if_else_ops(qc)
    assert len(if_else_ops) == 1

    op = if_else_ops[0].operation
    clbit, value = op.condition
    assert qc.clbits.index(clbit) == 0
    assert value == 1
    assert _get_ops_with_qubits(op.params[0]) == [("h", [1])]


def test_unsupported_operator_in_condition():
    """Operators other than == should raise TypeError."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0] != 1) {
    h q[1];
}
"""
    with pytest.raises(NotImplementedError, match="Unsupported operator.*Only '==' is supported"):
        to_qiskit(qasm)


def test_unsupported_condition_type():
    """A non-boolean-castable static condition should raise an unsupported condition error."""
    ctx = _QiskitProgramContext()
    gen = ctx.evaluate_condition(ArrayLiteral(values=[BooleanLiteral(value=True)]))
    with pytest.raises(TypeError, match="Unsupported condition in branching statement"):
        next(gen)


def test_evaluate_expression_unary():
    """UnaryExpression should be handled by _evaluate_expression."""
    ctx = _QiskitProgramContext()
    result = ctx._evaluate_expression(UnaryExpression(op=UnaryOperator["!"], expression=BooleanLiteral(value=True)))
    assert result.value is False


def test_evaluate_expression_cast():
    """Cast should be handled by _evaluate_expression."""
    ctx = _QiskitProgramContext()
    result = ctx._evaluate_expression(Cast(type=BoolType(), argument=IntegerLiteral(value=1)))
    assert result.value is True


def test_evaluate_expression_unsupported_type():
    """An unsupported expression type should raise TypeError."""
    ctx = _QiskitProgramContext()
    with pytest.raises(TypeError, match="Cannot evaluate expression of type"):
        ctx._evaluate_expression(FunctionCall(name=None, arguments=[]))


def test_evaluate_expression_list():
    """A list input should evaluate each element."""
    ctx = _QiskitProgramContext()
    result = ctx._evaluate_expression([IntegerLiteral(value=1), IntegerLiteral(value=2)])
    assert len(result) == 2
    assert result[0].value == 1
    assert result[1].value == 2


def test_multi_bit_register_condition_raises():
    """Using a multi-bit register as a bare condition should raise TypeError."""
    qasm = """
OPENQASM 3.0;
qubit[3] q;
bit[3] c;
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
if (c == 3) {
    h q[0];
}
"""
    with pytest.raises(TypeError, match="Multi-bit register.*cannot be used as a single-bit condition"):
        to_qiskit(qasm)


def test_nested_if_else():
    """Nested if/else should produce an IfElseOp inside the outer IfElseOp's true body."""
    qasm = """
OPENQASM 3.0;
qubit[3] q;
bit[2] c;
c[0] = measure q[0];
c[1] = measure q[1];
if (c[0] == 1) {
    if (c[1] == 1) {
        h q[2];
    } else {
        x q[2];
    }
}
"""
    qc = to_qiskit(qasm)
    outer_ops = _get_if_else_ops(qc)
    assert len(outer_ops) == 1

    outer = outer_ops[0].operation
    true_body, false_body = outer.params
    assert false_body is None

    # The outer true body should contain a nested IfElseOp
    inner_ops = _get_if_else_ops(true_body)
    assert len(inner_ops) == 1

    inner = inner_ops[0].operation
    inner_true, inner_false = inner.params
    assert _get_ops_with_qubits(inner_true) == [("h", [2])]
    assert _get_ops_with_qubits(inner_false) == [("x", [2])]


def test_physical_qubit_inside_branch_expands_circuit():
    """A physical qubit reference inside a branch should expand the main circuit."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0] == 1) {
    h $4;
}
"""
    qc = to_qiskit(qasm)
    assert qc.num_qubits == 5
    op = _get_if_else_ops(qc)[0].operation
    true_body = op.params[0]
    assert _get_ops_with_qubits(true_body) == [("h", [4])]


def test_mcm_while_loop_not_supported():
    """A while loop conditioned on a measurement result is not yet supported."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit c;
c = measure q[0];
while (c == 0) {
    x q[1];
    c = measure q[0];
}
"""
    with pytest.raises(ValueError):
        to_qiskit(qasm)


def test_classical_bit_declared_inside_branch_expands_circuit():
    """A classical bit declared inside a branch should expand the main circuit."""
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit c;
c = measure q[0];
if (c == 1) {
    bit d;
    d = measure q[1];
}
"""
    qc = to_qiskit(qasm)
    assert qc.num_clbits == 2
