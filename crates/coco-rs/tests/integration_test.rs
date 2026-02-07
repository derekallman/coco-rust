use std::path::PathBuf;

use coco_rs::params::IouType;
use coco_rs::{COCOeval, COCO};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn test_load_gt() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");
    assert_eq!(coco.dataset.images.len(), 3);
    assert_eq!(coco.dataset.annotations.len(), 5);
    assert_eq!(coco.dataset.categories.len(), 2);
}

#[test]
fn test_load_res() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");
    assert_eq!(coco_dt.dataset.annotations.len(), 7);
    // All annotations should have scores
    for ann in &coco_dt.dataset.annotations {
        assert!(ann.score.is_some());
    }
}

#[test]
fn test_bbox_evaluation_runs() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    coco_eval.evaluate();
    coco_eval.accumulate();

    let eval = coco_eval.eval.as_ref().expect("Accumulate should set eval");

    // Verify dimensions
    assert_eq!(eval.t, 10); // IoU thresholds
    assert_eq!(eval.r, 101); // recall thresholds
    assert_eq!(eval.k, 2); // categories
    assert_eq!(eval.a, 4); // area ranges
    assert_eq!(eval.m, 3); // max_dets

    // The precision array should have valid values (not all -1)
    let has_valid = eval.precision.iter().any(|&v| v >= 0.0);
    assert!(has_valid, "Should have some valid precision values");

    // Check that recall is non-negative for at least some entries
    let has_recall = eval.recall.iter().any(|&v| v >= 0.0);
    assert!(has_recall, "Should have some valid recall values");

    // For perfect matches at IoU=0.5 (dt bboxes closely match gt),
    // we should get high AP values
    // At IoU=0.5, our detections are good matches
    let ap_50_idx = eval.precision_idx(0, 0, 0, 0, 2); // t=0 (IoU=0.5), r=0, k=0 (cat), a=0 (all), m=2 (maxDet=100)
    let ap_50 = eval.precision[ap_50_idx];
    assert!(
        ap_50 > 0.0,
        "AP@0.5 for category 'cat' should be positive, got {}",
        ap_50
    );
}

#[test]
fn test_get_ann_ids_filtering() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");

    // Filter by image
    let ids = coco.get_ann_ids(&[1], &[], None, None);
    assert_eq!(ids.len(), 2);

    // Filter by category
    let ids = coco.get_ann_ids(&[], &[1], None, None);
    assert_eq!(ids.len(), 3); // 3 annotations with cat_id=1

    // Filter by both
    let ids = coco.get_ann_ids(&[2], &[1], None, None);
    assert_eq!(ids.len(), 2); // img 2 has 2 cat_id=1 annotations

    // Filter by area range
    let ids = coco.get_ann_ids(&[], &[], Some([500.0, 2000.0]), None);
    assert_eq!(ids.len(), 2); // area 900 and 1600
}

#[test]
fn test_summarize_prints() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    coco_eval.evaluate();
    coco_eval.accumulate();
    // This should print 12 lines without panicking
    coco_eval.summarize();
}
