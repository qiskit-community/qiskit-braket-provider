# Verbatim Pragma Test Results

## Test Overview
Testing how the circuit with `#pragma braket verbatim` performs when calling `to_braket()`.

## Input OpenQASM 3.0
```openqasm
OPENQASM 3.0;
h $0;
#pragma braket verbatim
box {
    cnot $0, $1;
}
```

## Key Findings

### 1. Qiskit to Braket Conversion (via `to_braket()`)
- **Status**: ⚠️ Verbatim pragma NOT preserved
- **Output Format**: JAQCD (JSON-based instruction format)
- **Result**: Standard H and CNOT gates without verbatim wrapping
- **Output**:
  ```
  braketSchemaHeader=BraketSchemaHeader(name='braket.ir.jaqcd.program', version='1')
  instructions=[H(target=0), CNot(control=0, target=1)]
  ```

### 2. Direct Braket Circuit from OpenQASM 3.0
- **Status**: ✅ Verbatim pragma IS preserved
- **Method**: `Circuit().from_ir(oq3_string)`
- **Result**: StartVerbatim and EndVerbatim markers correctly added
- **Output**:
  ```
  T  : │  0  │        1        │  2  │       3       │
        ┌───┐                                         
  q0 : ─┤ H ├───StartVerbatim─────●─────EndVerbatim───
        └───┘         ║           │          ║        
                      ║         ┌─┴─┐        ║        
  q1 : ───────────────╨─────────┤ X ├────────╨────────
                                └───┘                 
  ```

## Analysis

### Why `to_braket()` Doesn't Preserve Verbatim
1. Qiskit circuits don't have native verbatim pragma support
2. The `to_braket()` adapter converts Qiskit gates directly to Braket gates
3. No mechanism exists to mark certain gates as "verbatim" in Qiskit
4. Output uses JAQCD format which doesn't include verbatim markers

### Why Direct OpenQASM 3.0 Works
1. Braket's OpenQASM 3.0 parser recognizes `#pragma braket verbatim`
2. Parser wraps the verbatim box content with StartVerbatim/EndVerbatim instructions
3. These special instructions tell the backend to execute gates without optimization

## Implications for Qiskit-Braket Provider

To support verbatim pragmas in the Qiskit → Braket workflow, we need:

1. **Qiskit Circuit Annotation**: Add metadata to mark gates/subcircuits as verbatim
2. **Adapter Enhancement**: Modify `to_braket()` to detect verbatim annotations
3. **Verbatim Wrapping**: Insert StartVerbatim/EndVerbatim instructions around marked gates
4. **OpenQASM 3.0 Output**: Ensure verbatim sections are preserved when generating OQ3

## Recommended Approach

Based on the design document, implement:
- Custom gate/instruction metadata in Qiskit to mark verbatim sections
- Detection logic in the adapter to identify verbatim-marked operations
- Insertion of Braket's StartVerbatim/EndVerbatim instructions
- Proper OpenQASM 3.0 generation with `#pragma braket verbatim` boxes
