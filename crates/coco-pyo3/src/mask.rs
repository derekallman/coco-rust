use coco_core::mask as rmask;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::{py_to_rle, rle_to_py};

#[pyfunction]
#[pyo3(signature = (mask, h, w))]
pub fn encode(py: Python<'_>, mask: PyReadonlyArray2<u8>, h: u32, w: u32) -> PyResult<PyObject> {
    let shape = mask.shape();
    if shape[0] != h as usize || shape[1] != w as usize {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mask shape must match (h, w)",
        ));
    }
    // Convert row-major numpy to column-major for coco_core
    let mut col_major = vec![0u8; (h as usize) * (w as usize)];
    let mask_slice = mask.as_slice()?;
    for y in 0..h as usize {
        for x in 0..w as usize {
            col_major[y + h as usize * x] = mask_slice[y * w as usize + x];
        }
    }
    let rle = rmask::encode(&col_major, h, w);
    rle_to_py(py, &rle)
}

#[pyfunction]
pub fn decode(py: Python<'_>, rle: &Bound<'_, PyDict>) -> PyResult<Py<PyArray2<u8>>> {
    let rle = py_to_rle(rle)?;
    let col_major = rmask::decode(&rle);
    let h = rle.h as usize;
    let w = rle.w as usize;
    let arr = unsafe { PyArray2::new(py, [h, w], false) };
    unsafe {
        let ptr: *mut u8 = arr.as_raw_array_mut().as_mut_ptr();
        for y in 0..h {
            for x in 0..w {
                *ptr.add(y * w + x) = col_major[y + h * x];
            }
        }
    }
    Ok(arr.unbind())
}

#[pyfunction]
pub fn area(rle: &Bound<'_, PyDict>) -> PyResult<u64> {
    let rle = py_to_rle(rle)?;
    Ok(rmask::area(&rle))
}

#[pyfunction]
pub fn to_bbox(rle: &Bound<'_, PyDict>) -> PyResult<Vec<f64>> {
    let rle = py_to_rle(rle)?;
    Ok(rmask::to_bbox(&rle).to_vec())
}

#[pyfunction]
#[pyo3(signature = (rles, intersect = false))]
pub fn merge(py: Python<'_>, rles: Vec<Bound<'_, PyDict>>, intersect: bool) -> PyResult<PyObject> {
    let rles: Vec<coco_core::Rle> = rles.iter().map(py_to_rle).collect::<PyResult<_>>()?;
    let result = rmask::merge(&rles, intersect);
    rle_to_py(py, &result)
}

#[pyfunction]
pub fn iou(
    dt: Vec<Bound<'_, PyDict>>,
    gt: Vec<Bound<'_, PyDict>>,
    iscrowd: Vec<bool>,
) -> PyResult<Vec<Vec<f64>>> {
    let dt_rles: Vec<coco_core::Rle> = dt.iter().map(py_to_rle).collect::<PyResult<_>>()?;
    let gt_rles: Vec<coco_core::Rle> = gt.iter().map(py_to_rle).collect::<PyResult<_>>()?;
    Ok(rmask::iou(&dt_rles, &gt_rles, &iscrowd))
}

#[pyfunction]
pub fn bbox_iou(
    dt: Vec<[f64; 4]>,
    gt: Vec<[f64; 4]>,
    iscrowd: Vec<bool>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(rmask::bbox_iou(&dt, &gt, &iscrowd))
}

#[pyfunction]
pub fn fr_poly(py: Python<'_>, xy: Vec<f64>, h: u32, w: u32) -> PyResult<PyObject> {
    let rle = rmask::fr_poly(&xy, h, w);
    rle_to_py(py, &rle)
}

#[pyfunction]
pub fn fr_bbox(py: Python<'_>, bb: [f64; 4], h: u32, w: u32) -> PyResult<PyObject> {
    let rle = rmask::fr_bbox(&bb, h, w);
    rle_to_py(py, &rle)
}

#[pyfunction]
pub fn rle_to_string(rle: &Bound<'_, PyDict>) -> PyResult<String> {
    let rle = py_to_rle(rle)?;
    Ok(rmask::rle_to_string(&rle))
}

#[pyfunction]
pub fn rle_from_string(py: Python<'_>, s: &str, h: u32, w: u32) -> PyResult<PyObject> {
    let rle = rmask::rle_from_string(s, h, w);
    rle_to_py(py, &rle)
}
