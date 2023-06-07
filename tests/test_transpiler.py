"""Tests for the transpiler"""
import string
from unittest import TestCase

import numpy as np
import qiskit
from qiskit.circuit.library import HGate, PauliEvolutionGate
from qiskit import QuantumCircuit
from qiskit.opflow import I, X, Z
from qiskit.synthesis import SuzukiTrotter
import scipy
from qiskit_braket_provider import BraketLocalBackend, transpile


class TestTranspiler(TestCase):
    """TestTanspiler"""

    def _generate_params(self, name, varnames):
        """
        (Pseudo-)Automatically generate parameters for a given gate
        """
        params = {}
        rot_args = [
            "theta",
            "phi",
            "lam",
            "gamma",
            "v_x",
            "v_y",
            "v_z",
            "scaling",
            "f_x",
            "slope",
            "offset",
        ]

        if name == "EvolvedOperatorAnsatz":
            operators = (Z ^ Z) - 0.1 * (X ^ I)
            params = {"operators": operators, "parameter_prefix": "x"}

        if name == "QAOAAnsatz":
            operators = (Z ^ Z) - 0.1 * (X ^ I)
            params = {"cost_operator": operators, "parameter_prefix": "x"}

        if name == "IntegerComparator":
            params = {"num_state_qubits": 3, "value": 7}

        if name == "Diagonal":
            params["diag"] = np.exp(1j * np.random.rand(4) * np.pi)

        if name == "FourierChecking":
            params = {"f": [1, -1, -1, -1], "g": [1, 1, 1, -1]}

        if name == "GMS":
            params["num_qubits"] = np.random.randint(1, 5)
            params["theta"] = np.random.rand(params["num_qubits"])

        if name == "GroverOperator":
            oracle = QuantumCircuit(2)
            oracle.z(0)
            params = {"oracle": oracle}

        if name in ["GraphState", "HiddenLinearFunction"]:
            params = {"adjacency_matrix": [[1, 0, 0], [0, 1, 1], [0, 1, 1]]}

        if name == "IQP":
            params = {"interactions": [[6, 5, 3], [5, 4, 5], [3, 5, 1]]}

        if name == "PhaseEstimation":
            qc = QuantumCircuit(1)
            qc.h(0)
            params = {"num_evaluation_qubits": 1, "unitary": qc}

        if name == "PauliEvolutionGate":
            operator = (Z ^ Z) - 0.1 * (X ^ I)
            params = {"operator": operator}

        if name == "PiecewiseChebyshev":
            params = {
                "f_x": lambda x: np.arcsin(1 / x),
                "degree": 2,
                "breakpoints": [2, 4],
                "num_state_qubits": 2,
            }

        if name in ["MCMT", "MCMTVChain"]:
            params = {"gate": HGate(), "num_ctrl_qubits": 2, "num_target_qubits": 2}

        if name == "LinearFunction":
            params = {"linear": [[1, 0, 0], [1, 1, 0], [0, 0, 1]]}

        if len(params) > 0:
            return params

        if name == "PauliGate":
            params["label"] = "IXYZ"

        if name == "PermutationGate":
            params["pattern"] = [2, 4, 3, 0, 1]

        if name == "PhaseOracle":
            params["expression"] = "a & b"

        if "parameter_prefix" in varnames:
            params["parameter_prefix"] = "x"

        for ra in rot_args:
            if name in [
                "ExcitationPreserving",
                "DraperQFTAdder",
                "RGQFTMultiplier",
                "QuadraticForm",
            ]:
                break
            if ra in varnames:
                params[ra] = np.random.rand() * 2 * np.pi

        if "domain" in varnames:
            params["domain"] = (-200, 200)
            params["image"] = (-200, 200)

        if "params" in varnames:
            n = np.random.randint(1, 5)
            return {"params": np.ones(2**n) / np.sqrt(2**n)}

        if "unitary" in varnames:
            n = np.random.randint(1, 4)
            params["unitary"] = scipy.stats.unitary_group.rvs(2**n)

        for nqbis in [
            "num_ctrl_qubits",
            "num_qubits",
            "num_state_qubits",
            "num_variable_qubits",
            "feature_dimension",
            "num_evaluation_qubits",
        ]:
            if nqbis in varnames:
                params[nqbis] = np.random.randint(2, 5)

        return params

    def _get_qiskit_gates(self):
        """
        Automatically retrieves and create ALL Qiskit's gates.
        """
        qiskit_gates = {
            attr: None
            for attr in dir(qiskit.circuit.library)
            if attr[0] in string.ascii_uppercase
        }

        for gate_name in qiskit_gates:
            params = self._generate_params(
                gate_name,
                getattr(
                    qiskit.circuit.library, gate_name
                ).__init__.__code__.co_varnames,
            )
            try:
                qiskit_gates[gate_name] = getattr(qiskit.circuit.library, gate_name)(
                    **params
                )
            except Exception:
                continue

        return {k: v for k, v in qiskit_gates.items() if v is not None}

    def test_qiskit_gates_transpilation(self):
        """Test that Qiskit's gates are getting transpiled with no error thrown"""
        local_simulator = BraketLocalBackend()
        qiskit_gates = self._get_qiskit_gates()
        for gate_name, gate in qiskit_gates.items():
            with self.subTest(gate=gate):
                source_circuit = QuantumCircuit(gate.num_qubits, gate.num_qubits)

                # To avoid: Circuit must have instructions to run on a device
                if gate_name in [
                    "Barrier",
                    "IGate",
                    "Measure",
                    "NLocal",
                    "QuadraticForm",
                    "Reset",
                    "TwoLocal",
                    "XOR",
                    "GMS",
                    "Permutation",
                ]:
                    source_circuit.h(0)
                source_circuit.compose(gate, inplace=True)

                # ParameterExpression with unbound parameters
                bind_dict = {}
                for key in source_circuit.parameters:
                    bind_dict[key] = np.random.random()
                source_circuit = source_circuit.bind_parameters(bind_dict)
                transpiled_circuit = transpile(source_circuit)
                local_simulator.run(transpiled_circuit, shots=10).result().get_counts()

    def test_suzukitrotter_transpilation(self):
        """Test that a SuzukiTrotter circuit is getting transpiled with no error thrown"""
        local_simulator = BraketLocalBackend()
        operator = (Z ^ Z) - 0.1 * (X ^ I)
        evo = PauliEvolutionGate(operator, time=0.2)
        source_circuit = SuzukiTrotter().synthesize(evo)
        transpiled_circuit = transpile(source_circuit)
        local_simulator.run(transpiled_circuit, shots=10).result().get_counts()

    def test_preparestate_transpilation(self):
        """
        Test that a Qiskit circuit with state preparation is getting
        transpiled with no error thrown"""
        local_simulator = BraketLocalBackend()
        source_circuit = QuantumCircuit(1)
        source_circuit.prepare_state([1 / np.sqrt(2), -1 / np.sqrt(2)], 0)
        transpiled_circuit = transpile(source_circuit)
        local_simulator.run(transpiled_circuit, shots=10).result().get_counts()
