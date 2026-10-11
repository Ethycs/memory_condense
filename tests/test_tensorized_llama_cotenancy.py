"""Memory arithmetic for the tensorized multi-replica screen."""
import pytest

from tools.assay_tensorized_llama_cotenancy import factor_bytes, maximum_rank


def test_two_factor_copies_fit_at_computed_rank():
    rows=columns=3072
    deployed_q4_bytes=5_308_416
    rank=maximum_rank(rows,columns,deployed_q4_bytes,copies=2)
    assert rank==215
    assert factor_bytes(rows,columns,rank,copies=2) <= deployed_q4_bytes
    assert factor_bytes(rows,columns,rank+1,copies=2) > deployed_q4_bytes


@pytest.mark.parametrize('value', [0,-1])
def test_invalid_memory_dimensions_are_rejected(value):
    with pytest.raises(ValueError):
        factor_bytes(8,8,value,copies=2)
    with pytest.raises(ValueError):
        maximum_rank(8,8,value,copies=2)
