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


class BraketSampler(BaseSamplerV2):
    def __init__(self, backend: BraketBackend):
        self._backend = backend

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> BraketSamplerJob:
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        BraketSampler._validate_pubs(coerced_pubs)
        circuit_bindings = []
        pub_metadata = []
        for pub in coerced_pubs:
            circuit_binding, indices = self._pub_to_circuit_binding(pub)
            circuit_bindings.append(circuit_binding)
            pub_metadata.append({"indices": indices, "shape": pub.shape})
        program_set = ProgramSet(
            circuit_bindings,
            shots_per_executable=shots,
        )
        return BraketSamplerJob(
            self._backend._device.run(program_set),
            {"pubs": coerced_pubs, "pub_metadata": pub_metadata, "shots": shots}
        )

    @staticmethod
    def _validate_pubs(pubs: list[SamplerPub]):
        for pub in pubs:
            if pub.shots:
                raise ValueError("Per-pub shots not supported")

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
