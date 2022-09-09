from nasbench.api import NASBench
from nasbench.lib import model_spec as _model_spec
import pickle as p

ModelSpec = _model_spec.ModelSpec


class OutOfDomainError(Exception):
    """Indicates that the requested graph is outside of the search domain."""


class NASBench_(NASBench):
    def __init__(self, dataset_file=None):
        super().__init__(dataset_file=None, seed=None)
        if dataset_file is not None:
            self.data = p.load(open(dataset_file, 'rb'))

    def get_module_hash(self, model_spec):
        return self._hash_spec(model_spec)

    def query(self, model_spec, epochs=108, stop_halfway=False):
        module_hash = self.get_module_hash(model_spec)
        return self.data[f'{epochs}'][module_hash]
