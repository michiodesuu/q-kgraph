from .quantum_states import QuantumStateEncoder
from .unitary_operators import DiagonalUnitary, GivensUnitary, MatrixExpUnitary, build_unitary, UnitaryOperator
from .path_aggregator import PathEnumerator, AmplitudeAggregator, Path

# V4 quaternion components
from .quaternion_states import (
    QuaternionStateEncoder,
    quaternion_inner_product,
    hamilton_product,
    quaternion_normalize,
    quaternion_conjugate,
)
from .quaternion_operator import QuaternionUnitary
from .quantum_logic_lattice import QuantumLogicLattice
