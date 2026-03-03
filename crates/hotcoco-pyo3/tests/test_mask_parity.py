"""Parity tests: hotcoco.mask vs pycocotools.mask.

Every function in hotcoco.mask that has a pycocotools equivalent must produce
identical output (same types, same values).
"""

import numpy as np
import pycocotools.mask as pm
import pytest

from hotcoco import mask as hm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mask_2d():
    """(10, 10) Fortran-order binary mask with a 3×4 rectangle."""
    m = np.zeros((10, 10), dtype=np.uint8, order="F")
    m[2:5, 3:7] = 1
    return m


@pytest.fixture()
def mask_3d():
    """(10, 10, 2) Fortran-order binary mask with two distinct regions."""
    m = np.zeros((10, 10, 2), dtype=np.uint8, order="F")
    m[2:5, 3:7, 0] = 1
    m[0:2, 0:3, 1] = 1
    return m


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


class TestEncode:
    def test_2d_fortran(self, mask_2d):
        rp = pm.encode(mask_2d)
        rh = hm.encode(mask_2d)
        assert rp["counts"] == rh["counts"]
        assert list(rp["size"]) == list(rh["size"])

    def test_3d_fortran(self, mask_3d):
        rp = pm.encode(mask_3d)
        rh = hm.encode(mask_3d)
        assert len(rp) == len(rh) == 2
        for i, (a, b) in enumerate(zip(rp, rh)):
            assert a["counts"] == b["counts"], f"slice {i}"
            assert list(a["size"]) == list(b["size"]), f"slice {i}"

    def test_2d_c_order(self, mask_2d):
        """C-order arrays should produce the same RLE as Fortran-order."""
        m_c = np.ascontiguousarray(mask_2d)
        rp = pm.encode(np.asfortranarray(m_c))
        rh = hm.encode(m_c)
        assert rp["counts"] == rh["counts"]

    def test_3d_c_order(self, mask_3d):
        """C-order 3D arrays should produce the same RLE as Fortran-order."""
        m_c = np.ascontiguousarray(mask_3d)
        rp = pm.encode(np.asfortranarray(m_c))
        rh = hm.encode(m_c)
        assert len(rp) == len(rh)
        for i, (a, b) in enumerate(zip(rp, rh)):
            assert a["counts"] == b["counts"], f"slice {i}"

    def test_return_type_2d(self, mask_2d):
        rh = hm.encode(mask_2d)
        assert isinstance(rh, dict)
        assert isinstance(rh["counts"], bytes)
        assert isinstance(rh["size"], list)

    def test_return_type_3d(self, mask_3d):
        rh = hm.encode(mask_3d)
        assert isinstance(rh, list)
        assert all(isinstance(r, dict) for r in rh)

    def test_all_zeros(self):
        m = np.zeros((5, 5), dtype=np.uint8, order="F")
        rp = pm.encode(m)
        rh = hm.encode(m)
        assert rp["counts"] == rh["counts"]

    def test_all_ones(self):
        m = np.ones((5, 5), dtype=np.uint8, order="F")
        rp = pm.encode(m)
        rh = hm.encode(m)
        assert rp["counts"] == rh["counts"]


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


class TestDecode:
    def test_single_roundtrip(self, mask_2d):
        rle = pm.encode(mask_2d)
        dp = pm.decode(rle)
        dh = hm.decode(rle)
        np.testing.assert_array_equal(dp, dh)
        assert dh.flags.f_contiguous

    def test_list_roundtrip(self, mask_3d):
        rles = pm.encode(mask_3d)
        dp = pm.decode(rles)
        dh = hm.decode(rles)
        np.testing.assert_array_equal(dp, dh)
        assert dh.flags.f_contiguous

    def test_shape_2d(self, mask_2d):
        rle = pm.encode(mask_2d)
        dh = hm.decode(rle)
        assert dh.shape == (10, 10)

    def test_shape_3d(self, mask_3d):
        rles = pm.encode(mask_3d)
        dh = hm.decode(rles)
        assert dh.shape == (10, 10, 2)


# ---------------------------------------------------------------------------
# area
# ---------------------------------------------------------------------------


class TestArea:
    def test_single(self, mask_2d):
        rle = pm.encode(mask_2d)
        assert pm.area(rle) == hm.area(rle)

    def test_list(self, mask_3d):
        rles = pm.encode(mask_3d)
        np.testing.assert_array_equal(pm.area(rles), hm.area(rles))

    def test_return_type_single(self, mask_2d):
        rle = pm.encode(mask_2d)
        a = hm.area(rle)
        assert isinstance(a, (int, np.integer))

    def test_return_type_list(self, mask_3d):
        rles = pm.encode(mask_3d)
        a = hm.area(rles)
        assert isinstance(a, np.ndarray)
        assert a.dtype == np.uint32


# ---------------------------------------------------------------------------
# toBbox
# ---------------------------------------------------------------------------


class TestToBbox:
    def test_single(self, mask_2d):
        rle = pm.encode(mask_2d)
        np.testing.assert_array_equal(pm.toBbox(rle), hm.toBbox(rle))

    def test_list(self, mask_3d):
        rles = pm.encode(mask_3d)
        np.testing.assert_array_equal(pm.toBbox(rles), hm.toBbox(rles))

    def test_snake_case_alias(self, mask_2d):
        rle = pm.encode(mask_2d)
        np.testing.assert_array_equal(hm.to_bbox(rle), hm.toBbox(rle))

    def test_return_type_single(self, mask_2d):
        rle = pm.encode(mask_2d)
        b = hm.toBbox(rle)
        assert isinstance(b, np.ndarray)
        assert b.shape == (4,)

    def test_return_type_list(self, mask_3d):
        rles = pm.encode(mask_3d)
        b = hm.toBbox(rles)
        assert isinstance(b, np.ndarray)
        assert b.shape == (2, 4)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_union(self, mask_3d):
        rles = pm.encode(mask_3d)
        mp = pm.merge(rles, intersect=False)
        mh = hm.merge(rles, intersect=False)
        assert mp["counts"] == mh["counts"]

    def test_intersection(self, mask_3d):
        rles = pm.encode(mask_3d)
        mp = pm.merge(rles, intersect=True)
        mh = hm.merge(rles, intersect=True)
        assert mp["counts"] == mh["counts"]


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------


class TestIou:
    def test_self_iou(self, mask_3d):
        rles = pm.encode(mask_3d)
        iou_p = pm.iou(rles, rles, [False, False])
        iou_h = hm.iou(rles, rles, [False, False])
        np.testing.assert_allclose(iou_p, iou_h, atol=1e-10)

    def test_return_type(self, mask_3d):
        rles = pm.encode(mask_3d)
        iou_h = hm.iou(rles, rles, [False, False])
        assert isinstance(iou_h, np.ndarray)
        assert iou_h.shape == (2, 2)
        assert iou_h.dtype == np.float64


# ---------------------------------------------------------------------------
# frPyObjects
# ---------------------------------------------------------------------------


class TestFrPyObjects:
    def test_polygon_list(self):
        poly = [[3.0, 2.0, 7.0, 2.0, 7.0, 5.0, 3.0, 5.0]]
        rp = pm.frPyObjects(poly, 10, 10)
        rh = hm.frPyObjects(poly, 10, 10)
        assert isinstance(rp, list) and isinstance(rh, list)
        assert len(rp) == len(rh) == 1
        assert rp[0]["counts"] == rh[0]["counts"]

    def test_single_rle_dict(self):
        rle_dict = {"size": [10, 10], "counts": [2, 3, 95]}
        rp = pm.frPyObjects(rle_dict, 10, 10)
        rh = hm.frPyObjects(rle_dict, 10, 10)
        assert isinstance(rp, dict) and isinstance(rh, dict)
        assert rp["counts"] == rh["counts"]

    def test_multiple_polygons(self):
        polys = [
            [3.0, 2.0, 7.0, 2.0, 7.0, 5.0, 3.0, 5.0],
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
        ]
        rp = pm.frPyObjects(polys, 10, 10)
        rh = hm.frPyObjects(polys, 10, 10)
        assert len(rp) == len(rh) == 2
        for i, (a, b) in enumerate(zip(rp, rh)):
            assert a["counts"] == b["counts"], f"polygon {i}"


# ---------------------------------------------------------------------------
# Roundtrip: encode → decode → encode
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_encode_decode_encode(self, mask_2d):
        """encode → decode → encode should produce the same RLE."""
        rle1 = hm.encode(mask_2d)
        decoded = hm.decode(rle1)
        rle2 = hm.encode(decoded)
        assert rle1["counts"] == rle2["counts"]

    def test_large_mask(self):
        """Larger mask for stress testing."""
        rng = np.random.RandomState(42)
        m = (rng.rand(480, 640) > 0.5).astype(np.uint8)
        m_f = np.asfortranarray(m)
        rp = pm.encode(m_f)
        rh = hm.encode(m_f)
        assert rp["counts"] == rh["counts"]
        np.testing.assert_array_equal(pm.decode(rp), hm.decode(rh))
        assert pm.area(rp) == hm.area(rh)
        np.testing.assert_array_equal(pm.toBbox(rp), hm.toBbox(rh))
