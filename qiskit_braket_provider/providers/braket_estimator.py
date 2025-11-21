import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from qiskit.primitives import BaseEstimatorV2, EstimatorPubLike
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import Pauli, SparsePauliOp

from braket.circuits import Observable as BraketObservable
from braket.circuits.observables import I, Sum, TensorProduct, X, Y, Z
from braket.program_sets import CircuitBinding, ParameterSets, ProgramSet
from qiskit_braket_provider.providers.adapter import to_braket
from qiskit_braket_provider.providers.braket_backend import BraketBackend
from qiskit_braket_provider.providers.braket_estimator_job import (
    BraketEstimatorJob,
    _JobMetadata,
    _PubMetadata,
)

_PAULI_MAP = {
    "X": X,
    "Y": Y,
    "Z": Z,
}
_DEFAULT_PRECISION = 0.015625


class BraketEstimator(BaseEstimatorV2):
    """
    Efficient Braket implementation of Qiskit's BaseEstimatorV2.

    This estimator converts EstimatorPub objects to Braket ProgramSet objects
    using efficient broadcasting to minimize redundant CircuitBinding objects.
    """

    def __init__(
        self, backend: BraketBackend, *, verbatim: bool = False, optimization_level: int = 0
    ):
        """
        Initialize the Braket estimator.

        Args:
            backend (BraketBackend): The Braket backend to run circuits on.
            verbatim (bool): Whether to translate the circuit without any modification, in other
                words without transpiling it. Default: False.
            optimization_level (int): The optimization level to pass to `qiskit.transpile`. From Qiskit:
                0: no optimization (default) - basic translation, no optimization, trivial layout
                1: light optimization - routing + potential SaberSwap, some gate cancellation and 1Q gate folding
                2: medium optimization - better routing (noise aware) and commutative cancellation
                3: high optimization - gate resynthesis and unitary-breaking passes
        """
        self._backend = backend
        self._verbatim = verbatim
        self._optimization_level = optimization_level

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float = _DEFAULT_PRECISION
    ) -> BraketEstimatorJob:
        """
        Run estimation on the given pubs.

        Args:
            pubs (Iterable[EstimatorPubLike]): An iterable of EstimatorPubLike objects to estimate.
            precision (float): Target precision for expectation value estimates.
                Default: to 0.015625

        Returns:
            BraketEstimatorJob: A job object containing the estimation results.
        """
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        BraketEstimator._validate_pubs(coerced_pubs)

        all_bindings = []
        pub_metadata = []  # Track which bindings belong to which pub

        for pub in coerced_pubs:
            bindings, binding_to_result_map, sum_binding_indices = self._pub_to_circuit_bindings(
                pub
            )
            all_bindings.extend(bindings)
            pub_metadata.append(
                _PubMetadata(
                    num_bindings=len(bindings),
                    binding_to_result_map=binding_to_result_map,
                    sum_binding_indices=sum_binding_indices,
                )
            )

        shots = int(math.ceil(1.0 / precision**2))
        return BraketEstimatorJob(
            self._backend._device.run(ProgramSet(all_bindings, shots_per_executable=shots)),
            _JobMetadata(
                pubs=coerced_pubs,
                pub_metadata=pub_metadata,
                precision=precision,
                shots=shots,
            ),
        )

    @staticmethod
    def _validate_pubs(pubs: list[EstimatorPub]) -> None:
        """
        Validate that pubs meet requirements.

        Args:
            pubs (list[EstimatorPub]): List of EstimatorPub objects to validate.

        Raises:
            ValueError: If pubs have different precisions.
        """
        precisions = {pub.precision for pub in pubs if pub.precision is not None}
        if len(precisions) > 1:
            raise ValueError(f"All pubs must have the same precision, got: {precisions}")

    def _pub_to_circuit_bindings(
        self, pub: EstimatorPub
    ) -> tuple[list[CircuitBinding], dict[int, tuple[tuple[int, ...], int, int]], set[int]]:
        """
        Convert an EstimatorPub to a list of CircuitBindings.

        Since a CircuitBinding only takes one-dimensional parameter and observable arrays,
        multiple CircuitBindings are necessary to capture all the data in an EstimatorPub,
        whose parameter values and observables can take any broadcastable shapes.

        Each broadcasted (parameter values, observable) pair appears in at most one CircuitBinding.

        Args:
            pub (EstimatorPub): The EstimatorPub to convert.

        Returns:
            tuple[list[CircuitBinding], dict[int, tuple[tuple[int, ...], int, int]], set[int]]:
            The circuit bindings, pub shape, a map of binding index to array positions,
            and the indices of bindings with Pauli sum observables.
        """
        backend = self._backend
        target = backend.target
        qubit_labels = backend.qubit_labels

        observables = np.asarray(pub.observables)
        param_values = pub.parameter_values

        observables_broadcast, param_indices_broadcast = (
            np.broadcast_arrays(
                observables,
                np.fromiter(np.ndindex(shape := param_values.shape), dtype=object).reshape(shape),
            )
            if param_values.shape != ()
            else (observables, np.empty(observables.shape, dtype=object))
        )

        obs_keys = {BraketEstimator._make_obs_key(obs): obs for obs in observables.flatten()}

        # Group parameter sets with the same observable
        obs_groups = defaultdict(list)
        for position, (param_indices, obs) in enumerate(
            zip(param_indices_broadcast.flatten(), observables_broadcast.flatten())
        ):
            obs_groups[BraketEstimator._make_obs_key(obs)].append((position, param_indices))

        bindings = []
        binding_to_result_map = {}
        sum_binding_indices = set()
        processed_obs_keys = set()

        for obs_key, pairs in obs_groups.items():
            if obs_key in processed_obs_keys:
                continue

            param_indices = frozenset(pi for _, pi in pairs)

            # Find other observables with the same parameter sets to complete the Cartesian product
            matching_obs_keys = [
                ok
                for ok, prs in obs_groups.items()
                if (
                    frozenset(pi for _, pi in prs) == param_indices and ok not in processed_obs_keys
                )
            ]
            processed_obs_keys.update(matching_obs_keys)
            param_idx_map = {pk: idx for idx, pk in enumerate(param_indices)}

            circuit = to_braket(
                pub.circuit, target=target, verbatim=self._verbatim, qubit_labels=qubit_labels
            )
            observables = BraketEstimator._translate_observables(
                [obs_keys[ok] for ok in matching_obs_keys]
            )
            parameter_sets = (
                BraketEstimator._translate_parameters([param_values[pi] for pi in param_indices])
                if param_values.shape != ()
                else None
            )
            binding_idx = len(bindings)
            monomials = []
            for ok, observable in zip(matching_obs_keys, observables):
                if isinstance(observable, Sum):
                    bindings.append(
                        CircuitBinding(circuit, input_sets=parameter_sets, observables=observable)
                    )
                    # Map each position in the broadcast to its location in the binding result
                    binding_to_result_map[binding_idx] = [
                        (position, None, param_idx_map[pi]) for position, pi in obs_groups[ok]
                    ]
                    sum_binding_indices.add(binding_idx)
                    binding_idx += 1
                else:
                    monomials.append((ok, observable))

            if monomials:
                bindings.append(
                    CircuitBinding(
                        circuit,
                        input_sets=parameter_sets,
                        observables=[obs for _, obs in monomials],
                    )
                )
                # Map each position in the broadcast to its location in the binding result
                obs_idx_map = {ok: idx for idx, (ok, _) in enumerate(monomials)}
                binding_to_result_map[len(bindings) - 1] = [
                    (position, obs_idx_map[ok], param_idx_map[pi])
                    for ok, _ in monomials
                    for position, pi in obs_groups[ok]
                ]
        return bindings, binding_to_result_map, sum_binding_indices

    @staticmethod
    def _make_obs_key(obs_val: SparsePauliOp | dict[str, float]) -> str:
        """Create a hashable key for observable values.

        Args:
            obs_val (SparsePauliOp | dict[str, float]): A SparsePauliOp observable
                or dict representation

        Returns:
            str: A string representation that can be used as a dictionary key
        """
        return str(sorted(obs_val.items())) if isinstance(obs_val, dict) else str(obs_val)

    @staticmethod
    def _translate_parameters(param_list: list[BindingsArray]) -> ParameterSets:
        """
        Translate parameter values to Braket ParameterSets.

        Args:
            param_list (list[BindingsArray]): List of parameter value arrays.

        Returns:
            ParameterSets: Braket ParameterSets object.
        """
        data = defaultdict(list)
        for bindings_array in param_list:
            for k, v in bindings_array.data.items():
                for param, val in zip(k, v):
                    data[param].append(val)
        return ParameterSets(data)

    @staticmethod
    def _translate_observables(obs_list: list) -> list[BraketObservable]:
        """
        Translate a list of Qiskit observables to Braket observables.

        Args:
            obs_list (list): List of observables from an ObservablesArray.
                Each observable is a dict mapping Pauli strings to coefficients.

        Returns:
            list[BraketObservable]: List of Braket Observable objects (one per input observable).
        """
        result = []
        for obs_dict in obs_list:
            sp = SparsePauliOp.from_list(obs_dict.items())
            result.append(
                Sum(
                    [
                        BraketEstimator._pauli_to_observable(pauli, np.real(coeff))
                        for pauli, coeff in zip(sp.paulis, sp.coeffs)
                    ]
                )
                if len(sp) > 1
                else BraketEstimator._pauli_to_observable(sp.paulis[0], np.real(sp.coeffs[0]))
            )
        return result

    @staticmethod
    def _pauli_to_observable(pauli: Pauli, coeff: float = 1.0) -> BraketObservable:
        """
        Translate a single Pauli and a coefficient to a Braket observable.

        Args:
            pauli (Pauli): Pauli observable to translate.
            coeff (float): Coefficient of the Pauli. Default: 1.

        Returns:
            BraketObservable: Corresponding Braket observable
        """
        factors = [
            _PAULI_MAP[pauli_char](i)
            for i, pauli_char in enumerate(reversed(str(pauli)))
            if pauli_char != "I"
        ]
        if not factors:
            return I(0) * coeff  # Still include trivial term so expectation is correct
        return (TensorProduct(factors) if len(factors) > 1 else factors[0]) * coeff
