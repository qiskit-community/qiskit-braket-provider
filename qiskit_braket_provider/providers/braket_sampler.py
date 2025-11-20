from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from qiskit.primitives import BaseSamplerV2, SamplerPubLike
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub

from braket.program_sets import CircuitBinding, ParameterSets, ProgramSet
from qiskit_braket_provider import to_braket
from qiskit_braket_provider.providers.braket_backend import BraketBackend
from qiskit_braket_provider.providers.braket_sampler_job import BraketSamplerJob

_DEFAULT_SHOTS = 1024


class BraketSampler(BaseSamplerV2):
    def __init__(self, backend: BraketBackend):
        self._backend = backend

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
        pub_metadata = []
        for pub in coerced_pubs:
            circuit_binding, indices = self._pub_to_circuit_binding(pub)
            circuit_bindings.append(circuit_binding)
            pub_metadata.append({"indices": indices, "shape": pub.shape})
        shots_per_executable = pub_shots if pub_shots is not None else shots
        return BraketSamplerJob(
            self._backend._device.run(
                ProgramSet(circuit_bindings, shots_per_executable=shots_per_executable)
            ),
            {"pubs": coerced_pubs, "pub_metadata": pub_metadata, "shots": shots_per_executable},
        )

    @staticmethod
    def _pub_shots(pubs: list[SamplerPub]):
        shots_values = {pub.shots for pub in pubs}
        if len(shots_values) > 1:
            raise ValueError(f"All pubs must have the same shots, got: {shots_values}")
        return list(shots_values)[0]

    @staticmethod
    def _pub_to_circuit_binding(pub: SamplerPub) -> tuple[CircuitBinding, np.ndarray]:
        param_values = pub.parameter_values
        param_indices = np.fromiter(np.ndindex(param_values.shape), dtype=object).flatten()
        parameter_sets = (
            BraketSampler._translate_parameters([param_values[pi] for pi in param_indices])
            if param_values.shape != ()
            else None
        )
        return CircuitBinding(to_braket(pub.circuit), input_sets=parameter_sets), param_indices

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
