from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from qiskit.primitives import BaseSamplerV2, SamplerPubLike
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub

from braket.program_sets import CircuitBinding, ParameterSets, ProgramSet
from qiskit_braket_provider.providers.adapter import to_braket
from qiskit_braket_provider.providers.braket_backend import BraketBackend
from qiskit_braket_provider.providers.braket_sampler_job import BraketSamplerJob, _JobMetadata

_DEFAULT_SHOTS = 1024


class BraketSampler(BaseSamplerV2):
    def __init__(
        self,
        backend: BraketBackend,
        *,
        verbatim: bool = False,
        optimization_level: int = 0,
        **options,
    ):
        """
        Initialize the Braket sampler.

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
        self._options = options

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = _DEFAULT_SHOTS
    ) -> BraketSamplerJob:
        """
        Samples circuits with multiple parameter values.
        Args:
            pubs (Iterable[SamplerPubLike]): An iterable of SamplerPubLike objects to sample.
            shots (int): Number of shots to run for each circuit. Default: 1024
        Returns:
            BraketSamplerJob: A job object containing the sample results.
        """
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        pub_shots = BraketSampler._pub_shots(coerced_pubs)
        circuit_bindings = []
        parameter_indices = []
        for pub in coerced_pubs:
            circuit_binding, indices = self._pub_to_circuit_binding(pub)
            circuit_bindings.append(circuit_binding)
            parameter_indices.append(indices)
        shots_per_executable = pub_shots if pub_shots is not None else shots
        return BraketSamplerJob(
            self._backend._device.run(
                ProgramSet(circuit_bindings, shots_per_executable=shots_per_executable),
                **self._options,
            ),
            _JobMetadata(
                pubs=coerced_pubs, parameter_indices=parameter_indices, shots=shots_per_executable
            ),
        )

    @staticmethod
    def _pub_shots(pubs: list[SamplerPub]):
        shots_values = {pub.shots for pub in pubs}
        if len(shots_values) > 1:
            raise ValueError(f"All pubs must have the same shots, got: {shots_values}")
        return list(shots_values)[0]

    def _pub_to_circuit_binding(self, pub: SamplerPub) -> tuple[CircuitBinding, np.ndarray]:
        param_values = pub.parameter_values
        param_indices = np.fromiter(np.ndindex(param_values.shape), dtype=object).flatten()
        parameter_sets = (
            BraketSampler._translate_parameters([param_values[pi] for pi in param_indices])
            if param_values.shape != ()
            else None
        )
        backend = self._backend
        return CircuitBinding(
            to_braket(
                pub.circuit,
                target=backend.target,
                verbatim=self._verbatim,
                qubit_labels=backend.qubit_labels,
                optimization_level=self._optimization_level,
            ),
            input_sets=parameter_sets,
        ), param_indices

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
