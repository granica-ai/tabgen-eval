"""Exercise the actual Triton bounds logic on CPU; never initialize CUDA."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import tabgen_eval

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kernel", ["histogram", "mapping", "indicators"])
def test_malformed_category_indices_are_memory_safe_in_triton_interpreter(kernel):
    pytest.importorskip("triton")
    pytest.importorskip("torch")
    program = """
import sys
import torch
from tabgen_eval.execution.torch.cuda.triton.preprocessing import _hist, _map, _indicators
def forbidden(*a, **k):
    raise AssertionError('Test attempted CUDA initialization')
torch.cuda._lazy_init = forbidden
kind = sys.argv[1]
if kind == 'histogram':
    codes = torch.tensor([[0,-1],[1,0],[2**63-1,-2]],dtype=torch.int64)
    domain = torch.tensor([0,3,5],dtype=torch.int64)
    storage = torch.full((7,), 12345, dtype=torch.int64)
    output = storage[1:-1]; output.zero_()
    errors = torch.empty(1,dtype=torch.int32)
    _hist[(1,)](codes,domain,output,errors,3,2,128)
    assert output.tolist() == [0,1,1,1,1]
    assert errors.item() == 2
    assert storage[0].item() == storage[-1].item() == 12345
elif kind == 'mapping':
    codes = torch.tensor([[2**63-1,-2],[0,1]],dtype=torch.int64)
    domain = torch.tensor([0,3,6],dtype=torch.int64)
    mapping = torch.tensor([-1,0,1,-1,0,1],dtype=torch.int64)
    features = torch.tensor([0,2,4],dtype=torch.int64)
    output = torch.empty_like(codes)
    dummy = torch.empty(2,dtype=torch.int64)
    _map[(1,1)](codes,domain,mapping,features,output,dummy,dummy,2,2,True,False,256,8)
    assert output.tolist() == [[-1,-1],[0,3]]
else:
    codes = torch.tensor([[0,-1],[3,2**63-1],[4,-2]],dtype=torch.int64)
    storage = torch.full((20,), 12345., dtype=torch.float64)
    output = storage[1:-1].reshape(3,6); output.zero_()
    errors = torch.empty(1,dtype=torch.int32)
    _indicators[(1,)](codes,output,errors,3,2,2,6,128)
    expected = torch.zeros_like(output); expected[0,2] = 1; expected[1,5] = 1
    torch.testing.assert_close(output,expected)
    assert errors.item() == 3
    assert storage[0].item() == storage[-1].item() == 12345
assert not torch.cuda.is_initialized()
"""
    source = Path(tabgen_eval.__file__).resolve().parents[1]
    environment = dict(
        os.environ,
        TRITON_INTERPRET="1",
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(source),
        OMP_NUM_THREADS="2",
    )
    subprocess.run(
        [sys.executable, "-c", program, kernel],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
