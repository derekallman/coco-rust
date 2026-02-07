use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum IouType {
    Bbox,
    Segm,
    Keypoints,
}

#[derive(Debug, Clone)]
pub struct Params {
    pub iou_type: IouType,
    pub img_ids: Vec<u64>,
    pub cat_ids: Vec<u64>,
    pub iou_thrs: Vec<f64>,
    pub rec_thrs: Vec<f64>,
    pub max_dets: Vec<usize>,
    pub area_rng: Vec<[f64; 2]>,
    pub area_rng_lbl: Vec<String>,
    pub use_cats: bool,
    pub kpt_oks_sigmas: Vec<f64>,
}

impl Params {
    pub fn new(iou_type: IouType) -> Self {
        let (max_dets, area_rng, area_rng_lbl) = match iou_type {
            IouType::Keypoints => (
                vec![20],
                vec![
                    [0.0, 1e10],
                    [0.0, 32_f64.powi(2)],
                    [32_f64.powi(2), 96_f64.powi(2)],
                    [96_f64.powi(2), 1e10],
                ],
                vec![
                    "all".into(),
                    "medium".into(),
                    "large".into(),
                    // keypoints uses same labels for consistency
                    "".into(),
                ],
            ),
            _ => (
                vec![1, 10, 100],
                vec![
                    [0.0, 1e10],
                    [0.0, 32_f64.powi(2)],
                    [32_f64.powi(2), 96_f64.powi(2)],
                    [96_f64.powi(2), 1e10],
                ],
                vec![
                    "all".into(),
                    "small".into(),
                    "medium".into(),
                    "large".into(),
                ],
            ),
        };

        // Default OKS sigmas for 17 COCO keypoints
        let kpt_oks_sigmas = vec![
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107,
            0.107, 0.087, 0.087, 0.089, 0.089,
        ];

        let iou_thrs: Vec<f64> = (0..10).map(|i| 0.5 + 0.05 * i as f64).collect();
        let rec_thrs: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();

        Params {
            iou_type,
            img_ids: Vec::new(),
            cat_ids: Vec::new(),
            iou_thrs,
            rec_thrs,
            max_dets,
            area_rng,
            area_rng_lbl,
            use_cats: true,
            kpt_oks_sigmas,
        }
    }
}
