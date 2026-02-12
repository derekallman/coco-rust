use std::path::Path;

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod convert;
mod mask;

use convert::{annotation_to_py, category_to_py, image_to_py, py_to_annotation, rle_to_py};

// ---------------------------------------------------------------------------
// COCO
// ---------------------------------------------------------------------------

#[pyclass(name = "COCO")]
struct PyCOCO {
    inner: coco_core::COCO,
}

impl Clone for PyCOCO {
    fn clone(&self) -> Self {
        PyCOCO {
            inner: coco_core::COCO::from_dataset(self.inner.dataset.clone()),
        }
    }
}

#[pymethods]
impl PyCOCO {
    #[new]
    #[pyo3(signature = (annotation_file=None))]
    fn new(annotation_file: Option<&str>) -> PyResult<Self> {
        match annotation_file {
            Some(path) => {
                let inner = coco_core::COCO::new(Path::new(path))
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
                Ok(PyCOCO { inner })
            }
            None => {
                let inner = coco_core::COCO::from_dataset(coco_core::Dataset {
                    info: None,
                    images: vec![],
                    annotations: vec![],
                    categories: vec![],
                    licenses: vec![],
                });
                Ok(PyCOCO { inner })
            }
        }
    }

    #[pyo3(signature = (img_ids=vec![], cat_ids=vec![], area_rng=None, iscrowd=None))]
    fn get_ann_ids(
        &self,
        img_ids: Vec<u64>,
        cat_ids: Vec<u64>,
        area_rng: Option<[f64; 2]>,
        iscrowd: Option<bool>,
    ) -> Vec<u64> {
        self.inner
            .get_ann_ids(&img_ids, &cat_ids, area_rng, iscrowd)
    }

    #[pyo3(signature = (cat_nms=vec![], sup_nms=vec![], cat_ids=vec![]))]
    fn get_cat_ids(
        &self,
        cat_nms: Vec<String>,
        sup_nms: Vec<String>,
        cat_ids: Vec<u64>,
    ) -> Vec<u64> {
        let cat_nms_ref: Vec<&str> = cat_nms.iter().map(|s| s.as_str()).collect();
        let sup_nms_ref: Vec<&str> = sup_nms.iter().map(|s| s.as_str()).collect();
        self.inner.get_cat_ids(&cat_nms_ref, &sup_nms_ref, &cat_ids)
    }

    #[pyo3(signature = (img_ids=vec![], cat_ids=vec![]))]
    fn get_img_ids(&self, img_ids: Vec<u64>, cat_ids: Vec<u64>) -> Vec<u64> {
        self.inner.get_img_ids(&img_ids, &cat_ids)
    }

    fn load_anns(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<PyObject> {
        let anns = self.inner.load_anns(&ids);
        let list = PyList::new(
            py,
            anns.iter()
                .map(|a| annotation_to_py(py, a))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        Ok(list.into_any().unbind())
    }

    fn load_cats(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<PyObject> {
        let cats = self.inner.load_cats(&ids);
        let list = PyList::new(
            py,
            cats.iter()
                .map(|c| category_to_py(py, c))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        Ok(list.into_any().unbind())
    }

    fn load_imgs(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<PyObject> {
        let imgs = self.inner.load_imgs(&ids);
        let list = PyList::new(
            py,
            imgs.iter()
                .map(|i| image_to_py(py, i))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        Ok(list.into_any().unbind())
    }

    fn load_res(&self, res_file: &str) -> PyResult<PyCOCO> {
        let inner = self
            .inner
            .load_res(Path::new(res_file))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(PyCOCO { inner })
    }

    fn ann_to_rle(&self, py: Python<'_>, ann: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let annotation = py_to_annotation(ann)?;
        match self.inner.ann_to_rle(&annotation) {
            Some(rle) => rle_to_py(py, &rle),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "Could not convert annotation to RLE (image not found?)",
            )),
        }
    }

    fn ann_to_mask<'py>(
        &self,
        py: Python<'py>,
        ann: &Bound<'py, PyDict>,
    ) -> PyResult<Py<PyArray2<u8>>> {
        let annotation = py_to_annotation(ann)?;
        let rle = self.inner.ann_to_rle(&annotation).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Could not convert annotation to RLE (image not found?)",
            )
        })?;
        let col_major = coco_core::mask::decode(&rle);
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

    // camelCase aliases for pycocotools compatibility
    #[pyo3(name = "getAnnIds")]
    #[pyo3(signature = (img_ids=vec![], cat_ids=vec![], area_rng=None, iscrowd=None))]
    fn get_ann_ids_camel(
        &self,
        img_ids: Vec<u64>,
        cat_ids: Vec<u64>,
        area_rng: Option<[f64; 2]>,
        iscrowd: Option<bool>,
    ) -> Vec<u64> {
        self.get_ann_ids(img_ids, cat_ids, area_rng, iscrowd)
    }

    #[pyo3(name = "getCatIds")]
    #[pyo3(signature = (cat_nms=vec![], sup_nms=vec![], cat_ids=vec![]))]
    fn get_cat_ids_camel(
        &self,
        cat_nms: Vec<String>,
        sup_nms: Vec<String>,
        cat_ids: Vec<u64>,
    ) -> Vec<u64> {
        self.get_cat_ids(cat_nms, sup_nms, cat_ids)
    }

    #[pyo3(name = "getImgIds")]
    #[pyo3(signature = (img_ids=vec![], cat_ids=vec![]))]
    fn get_img_ids_camel(&self, img_ids: Vec<u64>, cat_ids: Vec<u64>) -> Vec<u64> {
        self.get_img_ids(img_ids, cat_ids)
    }

    #[pyo3(name = "loadAnns")]
    fn load_anns_camel(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<PyObject> {
        self.load_anns(py, ids)
    }

    #[pyo3(name = "loadCats")]
    fn load_cats_camel(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<PyObject> {
        self.load_cats(py, ids)
    }

    #[pyo3(name = "loadImgs")]
    fn load_imgs_camel(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<PyObject> {
        self.load_imgs(py, ids)
    }

    #[pyo3(name = "loadRes")]
    fn load_res_camel(&self, res_file: &str) -> PyResult<PyCOCO> {
        self.load_res(res_file)
    }

    #[pyo3(name = "annToRLE")]
    fn ann_to_rle_camel(&self, py: Python<'_>, ann: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        self.ann_to_rle(py, ann)
    }

    #[pyo3(name = "annToMask")]
    fn ann_to_mask_camel<'py>(
        &self,
        py: Python<'py>,
        ann: &Bound<'py, PyDict>,
    ) -> PyResult<Py<PyArray2<u8>>> {
        self.ann_to_mask(py, ann)
    }

    #[getter]
    fn dataset(&self, py: Python<'_>) -> PyResult<PyObject> {
        let ds = &self.inner.dataset;
        let dict = PyDict::new(py);

        let images = PyList::new(
            py,
            ds.images
                .iter()
                .map(|i| image_to_py(py, i))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        let annotations = PyList::new(
            py,
            ds.annotations
                .iter()
                .map(|a| annotation_to_py(py, a))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        let categories = PyList::new(
            py,
            ds.categories
                .iter()
                .map(|c| category_to_py(py, c))
                .collect::<PyResult<Vec<_>>>()?,
        )?;

        dict.set_item("images", images)?;
        dict.set_item("annotations", annotations)?;
        dict.set_item("categories", categories)?;

        Ok(dict.into_any().unbind())
    }
}

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

#[pyclass(name = "Params")]
#[derive(Clone)]
struct PyParams {
    inner: coco_core::Params,
}

#[pymethods]
impl PyParams {
    #[new]
    #[pyo3(signature = (iou_type="bbox"))]
    fn new(iou_type: &str) -> PyResult<Self> {
        let iou = parse_iou_type(iou_type)?;
        Ok(PyParams {
            inner: coco_core::Params::new(iou),
        })
    }

    #[getter]
    fn iou_type(&self) -> &str {
        match self.inner.iou_type {
            coco_core::IouType::Bbox => "bbox",
            coco_core::IouType::Segm => "segm",
            coco_core::IouType::Keypoints => "keypoints",
        }
    }

    #[setter]
    fn set_iou_type(&mut self, val: &str) -> PyResult<()> {
        self.inner.iou_type = parse_iou_type(val)?;
        Ok(())
    }

    #[getter]
    fn img_ids(&self) -> Vec<u64> {
        self.inner.img_ids.clone()
    }

    #[setter]
    fn set_img_ids(&mut self, val: Vec<u64>) {
        self.inner.img_ids = val;
    }

    #[getter]
    fn cat_ids(&self) -> Vec<u64> {
        self.inner.cat_ids.clone()
    }

    #[setter]
    fn set_cat_ids(&mut self, val: Vec<u64>) {
        self.inner.cat_ids = val;
    }

    #[getter]
    fn iou_thrs(&self) -> Vec<f64> {
        self.inner.iou_thrs.clone()
    }

    #[setter]
    fn set_iou_thrs(&mut self, val: Vec<f64>) {
        self.inner.iou_thrs = val;
    }

    #[getter]
    fn rec_thrs(&self) -> Vec<f64> {
        self.inner.rec_thrs.clone()
    }

    #[setter]
    fn set_rec_thrs(&mut self, val: Vec<f64>) {
        self.inner.rec_thrs = val;
    }

    #[getter]
    fn max_dets(&self) -> Vec<usize> {
        self.inner.max_dets.clone()
    }

    #[setter]
    fn set_max_dets(&mut self, val: Vec<usize>) {
        self.inner.max_dets = val;
    }

    #[getter]
    fn area_rng(&self) -> Vec<[f64; 2]> {
        self.inner.area_rng.clone()
    }

    #[setter]
    fn set_area_rng(&mut self, val: Vec<[f64; 2]>) {
        self.inner.area_rng = val;
    }

    #[getter]
    fn area_rng_lbl(&self) -> Vec<String> {
        self.inner.area_rng_lbl.clone()
    }

    #[setter]
    fn set_area_rng_lbl(&mut self, val: Vec<String>) {
        self.inner.area_rng_lbl = val;
    }

    #[getter]
    fn use_cats(&self) -> bool {
        self.inner.use_cats
    }

    #[setter]
    fn set_use_cats(&mut self, val: bool) {
        self.inner.use_cats = val;
    }

    #[getter]
    fn kpt_oks_sigmas(&self) -> Vec<f64> {
        self.inner.kpt_oks_sigmas.clone()
    }

    #[setter]
    fn set_kpt_oks_sigmas(&mut self, val: Vec<f64>) {
        self.inner.kpt_oks_sigmas = val;
    }

    // camelCase aliases for pycocotools compatibility
    #[getter(iouType)]
    fn iou_type_camel(&self) -> &str {
        self.iou_type()
    }

    #[setter(iouType)]
    fn set_iou_type_camel(&mut self, val: &str) -> PyResult<()> {
        self.set_iou_type(val)
    }

    #[getter(imgIds)]
    fn img_ids_camel(&self) -> Vec<u64> {
        self.img_ids()
    }

    #[setter(imgIds)]
    fn set_img_ids_camel(&mut self, val: Vec<u64>) {
        self.set_img_ids(val);
    }

    #[getter(catIds)]
    fn cat_ids_camel(&self) -> Vec<u64> {
        self.cat_ids()
    }

    #[setter(catIds)]
    fn set_cat_ids_camel(&mut self, val: Vec<u64>) {
        self.set_cat_ids(val);
    }

    #[getter(iouThrs)]
    fn iou_thrs_camel(&self) -> Vec<f64> {
        self.iou_thrs()
    }

    #[setter(iouThrs)]
    fn set_iou_thrs_camel(&mut self, val: Vec<f64>) {
        self.set_iou_thrs(val);
    }

    #[getter(recThrs)]
    fn rec_thrs_camel(&self) -> Vec<f64> {
        self.rec_thrs()
    }

    #[setter(recThrs)]
    fn set_rec_thrs_camel(&mut self, val: Vec<f64>) {
        self.set_rec_thrs(val);
    }

    #[getter(maxDets)]
    fn max_dets_camel(&self) -> Vec<usize> {
        self.max_dets()
    }

    #[setter(maxDets)]
    fn set_max_dets_camel(&mut self, val: Vec<usize>) {
        self.set_max_dets(val);
    }

    #[getter(areaRng)]
    fn area_rng_camel(&self) -> Vec<[f64; 2]> {
        self.area_rng()
    }

    #[setter(areaRng)]
    fn set_area_rng_camel(&mut self, val: Vec<[f64; 2]>) {
        self.set_area_rng(val);
    }

    #[getter(areaRngLbl)]
    fn area_rng_lbl_camel(&self) -> Vec<String> {
        self.area_rng_lbl()
    }

    #[setter(areaRngLbl)]
    fn set_area_rng_lbl_camel(&mut self, val: Vec<String>) {
        self.set_area_rng_lbl(val);
    }

    #[getter(useCats)]
    fn use_cats_camel(&self) -> bool {
        self.use_cats()
    }

    #[setter(useCats)]
    fn set_use_cats_camel(&mut self, val: bool) {
        self.set_use_cats(val);
    }
}

fn parse_iou_type(s: &str) -> PyResult<coco_core::IouType> {
    match s {
        "bbox" => Ok(coco_core::IouType::Bbox),
        "segm" => Ok(coco_core::IouType::Segm),
        "keypoints" => Ok(coco_core::IouType::Keypoints),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown iou_type: '{}'. Expected 'bbox', 'segm', or 'keypoints'",
            s
        ))),
    }
}

// ---------------------------------------------------------------------------
// COCOeval
// ---------------------------------------------------------------------------

#[pyclass(name = "COCOeval")]
struct PyCOCOeval {
    inner: coco_core::COCOeval,
}

#[pymethods]
impl PyCOCOeval {
    #[new]
    fn new(coco_gt: &PyCOCO, coco_dt: &PyCOCO, iou_type: &str) -> PyResult<Self> {
        let iou = parse_iou_type(iou_type)?;
        let inner = coco_core::COCOeval::new(coco_gt.clone().inner, coco_dt.clone().inner, iou);
        Ok(PyCOCOeval { inner })
    }

    fn evaluate(&mut self) {
        self.inner.evaluate();
    }

    fn accumulate(&mut self) {
        self.inner.accumulate();
    }

    fn summarize(&mut self) {
        self.inner.summarize();
    }

    #[getter]
    fn params(&self) -> PyParams {
        PyParams {
            inner: self.inner.params.clone(),
        }
    }

    #[setter]
    fn set_params(&mut self, params: &PyParams) {
        self.inner.params = params.inner.clone();
    }

    #[getter]
    fn stats(&self) -> Option<Vec<f64>> {
        self.inner.stats.clone()
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Patch `sys.modules` so that `from pycocotools.coco import COCO` etc.
/// transparently use coco-rust.
#[pyfunction]
fn init_as_pycocotools(py: Python<'_>) -> PyResult<()> {
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    let coco_rs = py.import("coco_rs")?;
    let mask_mod = coco_rs.getattr("mask")?;
    modules.set_item("pycocotools", &coco_rs)?;
    modules.set_item("pycocotools.coco", &coco_rs)?;
    modules.set_item("pycocotools.cocoeval", &coco_rs)?;
    modules.set_item("pycocotools.mask", &mask_mod)?;
    Ok(())
}

#[pymodule]
fn coco_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCOCO>()?;
    m.add_class::<PyCOCOeval>()?;
    m.add_class::<PyParams>()?;
    m.add_function(wrap_pyfunction!(init_as_pycocotools, m)?)?;

    // mask submodule
    let mask_mod = PyModule::new(py, "mask")?;
    mask_mod.add_function(wrap_pyfunction!(mask::encode, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::decode, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::area, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::to_bbox, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::merge, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::iou, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::bbox_iou, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::fr_poly, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::fr_bbox, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::rle_to_string, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::rle_from_string, &mask_mod)?)?;
    m.add_submodule(&mask_mod)?;

    Ok(())
}
