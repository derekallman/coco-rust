use std::path::PathBuf;

use coco_rs::params::IouType;
use coco_rs::types::{Annotation, Category, Dataset, Image};
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

/// Regression test for the iscrowd-vs-gt_ignore matching bug.
///
/// When a non-crowd GT is area-ignored (area outside the evaluated range),
/// it can be matched once by a detection (making that detection "ignored"),
/// but must NOT be re-matched by additional detections. Only crowd GTs
/// allow re-matching. The bug let area-ignored non-crowd GTs absorb
/// multiple detections as "ignored" instead of counting them as FP,
/// which inflated AP for medium/large area ranges.
#[test]
fn test_area_ignored_gt_does_not_absorb_multiple_detections() {
    // One image, one category, custom area range [500, 1e10].
    // GT_A: bbox [10,10,20,20] area=400, non-crowd → area-ignored (below 500)
    // GT_B: bbox [50,50,100,100] area=10000 → in range
    let gt_dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img1.jpg".into(),
            height: 200,
            width: 200,
            license: None,
            coco_url: None,
            flickr_url: None,
            date_captured: None,
        }],
        annotations: vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 20.0, 20.0]),
                area: Some(400.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([50.0, 50.0, 100.0, 100.0]),
                area: Some(10000.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
            },
        ],
        categories: vec![Category {
            id: 1,
            name: "thing".into(),
            supercategory: None,
            skeleton: None,
            keypoints: None,
        }],
        licenses: vec![],
    };

    // DT1: matches GT_A exactly, area=400 (small), score=0.9
    //       → matches area-ignored GT_A → DT1 is "ignored"
    // DT2: [10,10,25,20] area=500 (in range), overlaps GT_A (IoU≈0.8), score=0.8
    //       With fix: GT_A already matched, not crowd → can't re-match → FP
    //       With bug: GT_A is "ignorable" → re-match → DT2 also "ignored"
    // DT3: matches GT_B perfectly, area=10000 (in range), score=0.7 → TP
    //
    // Crucially, DT2 (the FP) has higher score than DT3 (the TP), so
    // the FP appears before the TP in the precision-recall curve,
    // reducing AP from 1.0 to ~0.5.
    let dt_dataset = Dataset {
        info: None,
        images: gt_dataset.images.clone(),
        annotations: vec![
            Annotation {
                id: 101,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 20.0, 20.0]),
                area: Some(400.0),
                score: Some(0.9),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
            Annotation {
                id: 102,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 25.0, 20.0]),
                area: Some(500.0),
                score: Some(0.8),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
            Annotation {
                id: 103,
                image_id: 1,
                category_id: 1,
                bbox: Some([50.0, 50.0, 100.0, 100.0]),
                area: Some(10000.0),
                score: Some(0.7),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
        ],
        categories: gt_dataset.categories.clone(),
        licenses: vec![],
    };

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    // Custom area range: [500, 1e10] so GT_A (area=400) is area-ignored
    coco_eval.params.area_rng = vec![[500.0, 1e10]];
    coco_eval.params.area_rng_lbl = vec!["custom".into()];
    coco_eval.evaluate();
    coco_eval.accumulate();

    let eval = coco_eval.eval.as_ref().unwrap();
    let m_idx = eval.m - 1;

    // Correct behavior:
    //   Sorted by score: DT1(0.9), DT2(0.8), DT3(0.7)
    //   DT1 matches area-ignored GT_A → DT1 is "ignored".
    //   DT2 overlaps GT_A (IoU≈0.8) but GT_A already matched and not crowd → skip.
    //   DT2 unmatched, area=500 in range → FP.
    //   DT3 matches GT_B → TP.
    //   Non-ignored dets: DT2(FP, score=0.8), DT3(TP, score=0.7).
    //   AP@0.5 ≈ 0.5 (FP before TP in ranking).
    //
    // Buggy behavior (gt_ignore instead of iscrowd):
    //   DT2 re-matches area-ignored GT_A → DT2 also "ignored".
    //   Non-ignored dets: only DT3(TP). AP@0.5 = 1.0.
    let ap_sum: f64 = (0..eval.r)
        .map(|r| {
            let idx = eval.precision_idx(0, r, 0, 0, m_idx);
            let p = eval.precision[idx];
            if p < 0.0 { 0.0 } else { p }
        })
        .sum();
    let ap = ap_sum / eval.r as f64;

    assert!(
        ap < 0.9,
        "AP should be ~0.5 (with FP counted), got {ap:.4}. \
         If AP ≈ 1.0, area-ignored non-crowd GT is incorrectly absorbing multiple detections."
    );
    assert!(
        ap > 0.3,
        "AP should be ~0.5, got {ap:.4}"
    );
}
