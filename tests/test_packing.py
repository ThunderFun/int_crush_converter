"""Tests for quant.packing — INT4 pack/unpack roundtrip."""

import torch
import pytest

from converter.packing import (
    pack_int4,
    unpack_int4,
    validate_int4_range,
    INT4PackingError,
)


class TestValidateInt4Range:
    def test_valid_values(self):
        validate_int4_range(torch.tensor([-8, -1, 0, 7]))

    def test_out_of_range_low(self):
        with pytest.raises(INT4PackingError):
            validate_int4_range(torch.tensor([-9]))

    def test_out_of_range_high(self):
        with pytest.raises(INT4PackingError):
            validate_int4_range(torch.tensor([8]))


class TestPackRoundtrip:
    def test_roundtrip_1d(self):
        values = torch.tensor([-8, -1, 0, 1, 7, -3, 5, 0], dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, K=values.shape[0])
        assert torch.equal(values, unpacked)

    def test_roundtrip_2d(self):
        values = torch.randint(-8, 8, (4, 32), dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, K=32)
        assert torch.equal(values, unpacked)

    def test_roundtrip_odd_length(self):
        values = torch.tensor([-8, 0, 7], dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, K=3)
        assert torch.equal(values, unpacked)

    def test_packed_shape(self):
        values = torch.randint(-8, 8, (4, 33), dtype=torch.int8)
        packed = pack_int4(values)
        assert packed.shape == (4, 17)  # ceil(33/2) = 17

    def test_all_min_values(self):
        values = torch.full((2, 16), -8, dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, K=16)
        assert torch.equal(values, unpacked)

    def test_all_max_values(self):
        values = torch.full((2, 16), 7, dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, K=16)
        assert torch.equal(values, unpacked)
