//! COCO evaluation engine — faithful port of `pycocotools/cocoeval.py`.
//!
//! Implements evaluate, accumulate, and summarize for bbox, segm, and keypoint evaluation.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::coco::COCO;
use crate::mask;
use crate::params::{IouType, Params};
use crate::types::Rle;

/// Per-image, per-category evaluation result.
#[derive(Debug, Clone)]
pub struct EvalImg {
    pub image_id: u64,
    pub category_id: u64,
    pub area_rng: [f64; 2],
    pub max_det: usize,
    /// Detection annotation IDs (sorted by score descending, truncated to max_det)
    pub dt_ids: Vec<u64>,
    /// Ground truth annotation IDs (sorted: non-ignored first, then ignored)
    pub gt_ids: Vec<u64>,
    /// Detection matches for each IoU threshold: dt_matches[t][d] = matched gt_id or 0
    pub dt_matches: Vec<Vec<u64>>,
    /// Ground truth matches for each IoU threshold: gt_matches[t][g] = matched dt_id or 0
    pub gt_matches: Vec<Vec<u64>>,
    /// Detection scores
    pub dt_scores: Vec<f64>,
    /// Whether each GT is ignored
    pub gt_ignore: Vec<bool>,
    /// Whether each detection is ignored per IoU threshold
    pub dt_ignore: Vec<Vec<bool>>,
}

/// Accumulated evaluation results.
#[derive(Debug, Clone)]
pub struct AccumulatedEval {
    /// Precision: [T x R x K x A x M]
    pub precision: Vec<f64>,
    /// Recall: [T x K x A x M]
    pub recall: Vec<f64>,
    /// Scores at precision thresholds: [T x R x K x A x M]
    pub scores: Vec<f64>,
    /// Shape parameters
    pub t: usize,
    pub r: usize,
    pub k: usize,
    pub a: usize,
    pub m: usize,
}

impl AccumulatedEval {
    pub fn precision_idx(&self, t: usize, r: usize, k: usize, a: usize, m: usize) -> usize {
        ((((t * self.r + r) * self.k + k) * self.a + a) * self.m) + m
    }

    pub fn recall_idx(&self, t: usize, k: usize, a: usize, m: usize) -> usize {
        (((t * self.k + k) * self.a + a) * self.m) + m
    }

    #[allow(dead_code)]
    fn scores_idx(&self, t: usize, r: usize, k: usize, a: usize, m: usize) -> usize {
        self.precision_idx(t, r, k, a, m)
    }
}

/// The COCO evaluation object.
pub struct COCOeval {
    pub coco_gt: COCO,
    pub coco_dt: COCO,
    pub params: Params,
    pub eval_imgs: Vec<Option<EvalImg>>,
    ious: HashMap<(u64, u64), Vec<Vec<f64>>>,
    pub eval: Option<AccumulatedEval>,
    pub stats: Option<Vec<f64>>,
}

impl COCOeval {
    /// Create a new COCOeval from ground truth and detection COCO objects.
    pub fn new(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self {
        COCOeval {
            coco_gt,
            coco_dt,
            params: Params::new(iou_type),
            eval_imgs: Vec::new(),
            ious: HashMap::new(),
            eval: None,
            stats: None,
        }
    }

    /// Run per-image evaluation.
    pub fn evaluate(&mut self) {
        // Set img_ids and cat_ids if not set
        if self.params.img_ids.is_empty() {
            let mut ids: Vec<u64> = self.coco_gt.dataset.images.iter().map(|i| i.id).collect();
            ids.sort_unstable();
            self.params.img_ids = ids;
        }
        if self.params.cat_ids.is_empty() {
            let mut ids: Vec<u64> = self
                .coco_gt
                .dataset
                .categories
                .iter()
                .map(|c| c.id)
                .collect();
            ids.sort_unstable();
            self.params.cat_ids = ids;
        }

        let cat_ids = if self.params.use_cats {
            self.params.cat_ids.clone()
        } else {
            vec![0] // dummy single category
        };

        // Compute IoUs for all (image, category) pairs in parallel
        let img_ids = self.params.img_ids.clone();
        let pairs: Vec<(u64, u64)> = cat_ids
            .iter()
            .flat_map(|&cat_id| img_ids.iter().map(move |&img_id| (img_id, cat_id)))
            .collect();

        #[allow(clippy::type_complexity)]
        let iou_results: Vec<((u64, u64), Vec<Vec<f64>>)> = pairs
            .par_iter()
            .map(|&(img_id, cat_id)| {
                let iou_matrix = Self::compute_iou_static(
                    &self.coco_gt,
                    &self.coco_dt,
                    &self.params,
                    img_id,
                    cat_id,
                );
                ((img_id, cat_id), iou_matrix)
            })
            .collect();

        self.ious.clear();
        self.ious.reserve(iou_results.len());
        for (key, val) in iou_results {
            self.ious.insert(key, val);
        }

        // Evaluate each (image, category, area_range, max_det) combination in parallel
        let max_det = *self.params.max_dets.last().unwrap_or(&100);
        let area_rngs = self.params.area_rng.clone();

        let mut eval_tuples: Vec<(u64, [f64; 2], u64)> =
            Vec::with_capacity(cat_ids.len() * area_rngs.len() * img_ids.len());
        for &cat_id in &cat_ids {
            for &area_rng in &area_rngs {
                for &img_id in &img_ids {
                    eval_tuples.push((cat_id, area_rng, img_id));
                }
            }
        }

        self.eval_imgs = eval_tuples
            .par_iter()
            .map(|&(cat_id, area_rng, img_id)| {
                Self::evaluate_img_static(
                    &self.coco_gt,
                    &self.coco_dt,
                    &self.params,
                    &self.ious,
                    img_id,
                    cat_id,
                    area_rng,
                    max_det,
                )
            })
            .collect();
    }

    /// Compute the IoU/OKS matrix for a given image and category.
    fn compute_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        params: &Params,
        img_id: u64,
        cat_id: u64,
    ) -> Vec<Vec<f64>> {
        let gt_anns = Self::get_anns_static(coco_gt, params, img_id, cat_id);
        let dt_anns = Self::get_anns_static(coco_dt, params, img_id, cat_id);

        if gt_anns.is_empty() || dt_anns.is_empty() {
            return Vec::new();
        }

        match params.iou_type {
            IouType::Segm => Self::compute_segm_iou_static(coco_gt, coco_dt, dt_anns, gt_anns),
            IouType::Bbox => Self::compute_bbox_iou_static(coco_gt, coco_dt, dt_anns, gt_anns),
            IouType::Keypoints => {
                Self::compute_oks_static(coco_gt, coco_dt, params, dt_anns, gt_anns)
            }
        }
    }

    fn get_anns_static<'a>(coco: &'a COCO, params: &Params, img_id: u64, cat_id: u64) -> &'a [u64] {
        if params.use_cats {
            coco.get_ann_ids_for_img_cat(img_id, cat_id)
        } else {
            coco.get_ann_ids_for_img(img_id)
        }
    }

    fn compute_segm_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
    ) -> Vec<Vec<f64>> {
        let dt_rles: Vec<Rle> = dt_ids
            .iter()
            .filter_map(|&id| {
                let ann = coco_dt.get_ann(id)?;
                coco_dt.ann_to_rle(ann)
            })
            .collect();
        let gt_rles: Vec<Rle> = gt_ids
            .iter()
            .filter_map(|&id| {
                let ann = coco_gt.get_ann(id)?;
                coco_gt.ann_to_rle(ann)
            })
            .collect();

        let iscrowd: Vec<bool> = gt_ids
            .iter()
            .filter_map(|&id| coco_gt.get_ann(id).map(|a| a.iscrowd))
            .collect();

        mask::iou(&dt_rles, &gt_rles, &iscrowd)
    }

    fn compute_bbox_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
    ) -> Vec<Vec<f64>> {
        let dt_bbs: Vec<[f64; 4]> = dt_ids
            .iter()
            .filter_map(|&id| coco_dt.get_ann(id)?.bbox)
            .collect();
        let gt_bbs: Vec<[f64; 4]> = gt_ids
            .iter()
            .filter_map(|&id| coco_gt.get_ann(id)?.bbox)
            .collect();
        let iscrowd: Vec<bool> = gt_ids
            .iter()
            .filter_map(|&id| coco_gt.get_ann(id).map(|a| a.iscrowd))
            .collect();

        mask::bbox_iou(&dt_bbs, &gt_bbs, &iscrowd)
    }

    fn compute_oks_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        params: &Params,
        dt_ids: &[u64],
        gt_ids: &[u64],
    ) -> Vec<Vec<f64>> {
        let sigmas = &params.kpt_oks_sigmas;
        let num_kpts = sigmas.len();
        // vars = (sigmas * 2)**2 = 4 * sigma^2  (matching pycocotools)
        let vars: Vec<f64> = sigmas.iter().map(|s| (2.0 * s).powi(2)).collect();

        let d = dt_ids.len();
        let g = gt_ids.len();
        let mut result = vec![vec![0.0f64; g]; d];

        let gt_anns: Vec<_> = gt_ids
            .iter()
            .filter_map(|&id| coco_gt.get_ann(id))
            .collect();
        let dt_anns: Vec<_> = dt_ids
            .iter()
            .filter_map(|&id| coco_dt.get_ann(id))
            .collect();

        for (j, gt_ann) in gt_anns.iter().enumerate() {
            let gt_kpts = match &gt_ann.keypoints {
                Some(k) => k,
                None => continue,
            };
            let gt_area = gt_ann.area.unwrap_or(0.0) + f64::EPSILON;
            let gt_bbox = gt_ann.bbox.unwrap_or([0.0; 4]);

            // Count visible GT keypoints
            let k1: usize = (0..num_kpts)
                .filter(|&ki| gt_kpts.get(ki * 3 + 2).copied().unwrap_or(0.0) > 0.0)
                .count();

            // Compute ignore region bounds (double the GT bbox)
            let x0 = gt_bbox[0] - gt_bbox[2];
            let x1 = gt_bbox[0] + gt_bbox[2] * 2.0;
            let y0 = gt_bbox[1] - gt_bbox[3];
            let y1 = gt_bbox[1] + gt_bbox[3] * 2.0;

            for (i, dt_ann) in dt_anns.iter().enumerate() {
                let dt_kpts = match &dt_ann.keypoints {
                    Some(k) => k,
                    None => continue,
                };

                // Compute per-keypoint distances
                let mut e_vals: Vec<f64> = Vec::with_capacity(num_kpts);

                for (ki, &var_k) in vars.iter().enumerate().take(num_kpts) {
                    let gx = gt_kpts.get(ki * 3).copied().unwrap_or(0.0);
                    let gy = gt_kpts.get(ki * 3 + 1).copied().unwrap_or(0.0);
                    let xd = dt_kpts.get(ki * 3).copied().unwrap_or(0.0);
                    let yd = dt_kpts.get(ki * 3 + 1).copied().unwrap_or(0.0);

                    let (dx, dy) = if k1 > 0 {
                        (xd - gx, yd - gy)
                    } else {
                        // No visible GT keypoints: measure distance to bbox boundary
                        let dx = 0.0_f64.max(x0 - xd) + 0.0_f64.max(xd - x1);
                        let dy = 0.0_f64.max(y0 - yd) + 0.0_f64.max(yd - y1);
                        (dx, dy)
                    };

                    let e = (dx * dx + dy * dy) / var_k / gt_area / 2.0;
                    e_vals.push(e);
                }

                // Filter to visible keypoints if k1 > 0
                let filtered: Vec<f64> = if k1 > 0 {
                    e_vals
                        .iter()
                        .enumerate()
                        .filter(|&(ki, _)| gt_kpts.get(ki * 3 + 2).copied().unwrap_or(0.0) > 0.0)
                        .map(|(_, &e)| e)
                        .collect()
                } else {
                    e_vals
                };

                if !filtered.is_empty() {
                    let oks: f64 =
                        filtered.iter().map(|&e| (-e).exp()).sum::<f64>() / filtered.len() as f64;
                    result[i][j] = oks;
                }
            }
        }

        result
    }

    /// Evaluate a single image+category combination.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_img_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        params: &Params,
        ious: &HashMap<(u64, u64), Vec<Vec<f64>>>,
        img_id: u64,
        cat_id: u64,
        area_rng: [f64; 2],
        max_det: usize,
    ) -> Option<EvalImg> {
        let gt_ids = Self::get_anns_static(coco_gt, params, img_id, cat_id);
        let dt_ids = Self::get_anns_static(coco_dt, params, img_id, cat_id);

        if gt_ids.is_empty() && dt_ids.is_empty() {
            return None;
        }

        // Load GT annotations, determine ignore flags
        let gt_anns: Vec<_> = gt_ids
            .iter()
            .filter_map(|&id| coco_gt.get_ann(id))
            .collect();
        let is_kp = params.iou_type == IouType::Keypoints;
        let gt_ignore: Vec<bool> = gt_anns
            .iter()
            .map(|ann| {
                let a = ann.area.unwrap_or(0.0);
                let mut ignore = ann.iscrowd || a < area_rng[0] || a > area_rng[1];
                // For keypoints, also ignore GT annotations with num_keypoints == 0
                if is_kp {
                    ignore = ignore || ann.num_keypoints.unwrap_or(0) == 0;
                }
                ignore
            })
            .collect();

        // Sort GT: non-ignored first, then ignored
        let mut gt_order: Vec<usize> = (0..gt_anns.len()).collect();
        gt_order.sort_by_key(|&i| gt_ignore[i] as u8);
        let gt_ignore_sorted: Vec<bool> = gt_order.iter().map(|&i| gt_ignore[i]).collect();
        let gt_iscrowd_sorted: Vec<bool> = gt_order.iter().map(|&i| gt_anns[i].iscrowd).collect();
        let num_gt_not_ignored = gt_ignore_sorted.iter().filter(|&&x| !x).count();

        // Load DT annotations, sort by score descending, limit to max_det
        let mut dt_anns: Vec<_> = dt_ids
            .iter()
            .filter_map(|&id| coco_dt.get_ann(id))
            .collect();
        dt_anns.sort_by(|a, b| {
            b.score
                .unwrap_or(0.0)
                .partial_cmp(&a.score.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if dt_anns.len() > max_det {
            dt_anns.truncate(max_det);
        }

        let dt_scores: Vec<f64> = dt_anns.iter().map(|a| a.score.unwrap_or(0.0)).collect();

        // Determine which DT are ignored by area
        let dt_area_ignore: Vec<bool> = dt_anns
            .iter()
            .map(|ann| {
                let a = ann.area.unwrap_or(0.0);
                a < area_rng[0] || a > area_rng[1]
            })
            .collect();

        // Get IoU matrix
        let iou_matrix = ious.get(&(img_id, cat_id));

        let num_iou_thrs = params.iou_thrs.len();
        let d = dt_anns.len();
        let g = gt_anns.len();

        let mut dt_matches = vec![vec![0u64; d]; num_iou_thrs];
        let mut gt_matches = vec![vec![0u64; g]; num_iou_thrs];
        let mut dt_ignore_flags = vec![vec![false; d]; num_iou_thrs];

        if let Some(iou_mat) = iou_matrix {
            // Precompute remapped IoU matrix indexed by (dt_anns order, gt_order)
            // so we can use direct array indexing instead of HashMap lookups.
            let dt_id_to_iou_idx: HashMap<u64, usize> =
                dt_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            let gt_id_to_iou_idx: HashMap<u64, usize> =
                gt_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

            let iou_reordered: Vec<Vec<f64>> = dt_anns
                .iter()
                .map(|dt_ann| {
                    let dt_idx = dt_id_to_iou_idx.get(&dt_ann.id).copied();
                    gt_order
                        .iter()
                        .map(|&gi_orig| {
                            let gt_idx = gt_id_to_iou_idx.get(&gt_anns[gi_orig].id).copied();
                            match (dt_idx, gt_idx) {
                                (Some(di), Some(gi))
                                    if di < iou_mat.len() && gi < iou_mat[di].len() =>
                                {
                                    iou_mat[di][gi]
                                }
                                _ => 0.0,
                            }
                        })
                        .collect()
                })
                .collect();

            for (t_idx, &iou_thr) in params.iou_thrs.iter().enumerate() {
                for (di, dt_ann) in dt_anns.iter().enumerate() {
                    let mut best_iou = iou_thr; // minimum threshold
                    let mut best_gi: Option<usize> = None;

                    let dt_row = &iou_reordered[di];

                    for (gi_sorted, &iou_val) in dt_row.iter().enumerate() {
                        // Skip already matched non-crowd GTs (pycocotools uses iscrowd,
                        // not the full ignore flag, so crowd GTs can be matched multiple times)
                        if gt_matches[t_idx][gi_sorted] != 0 && !gt_iscrowd_sorted[gi_sorted] {
                            continue;
                        }

                        // Match: iou must meet threshold, prefer non-ignored GT
                        if iou_val < best_iou {
                            continue;
                        }

                        // Prefer non-ignored GT over ignored GT
                        if let Some(prev_gi) = best_gi {
                            let best_ignored = gt_ignore_sorted[prev_gi];
                            let curr_ignored = gt_ignore_sorted[gi_sorted];
                            if !best_ignored && curr_ignored {
                                continue;
                            }
                        }

                        best_iou = iou_val;
                        best_gi = Some(gi_sorted);
                    }

                    if let Some(gi) = best_gi {
                        dt_matches[t_idx][di] = gt_anns[gt_order[gi]].id;
                        gt_matches[t_idx][gi] = dt_ann.id;

                        // DT is ignored if matched to ignored GT
                        dt_ignore_flags[t_idx][di] = gt_ignore_sorted[gi];
                    } else {
                        // Unmatched DT: ignored if area out of range
                        dt_ignore_flags[t_idx][di] = dt_area_ignore[di];
                    }
                }
            }
        }

        // If there are no non-ignored GTs and no non-ignored DTs, skip
        let has_content = num_gt_not_ignored > 0
            || dt_anns
                .iter()
                .enumerate()
                .any(|(di, _)| !dt_area_ignore[di]);
        if !has_content && gt_ids.is_empty() {
            return None;
        }

        Some(EvalImg {
            image_id: img_id,
            category_id: cat_id,
            area_rng,
            max_det,
            dt_ids: dt_anns.iter().map(|a| a.id).collect(),
            gt_ids: gt_order.iter().map(|&i| gt_anns[i].id).collect(),
            dt_matches,
            gt_matches,
            dt_scores,
            gt_ignore: gt_ignore_sorted,
            dt_ignore: dt_ignore_flags,
        })
    }

    /// Accumulate per-image results into precision/recall arrays.
    pub fn accumulate(&mut self) {
        let t = self.params.iou_thrs.len();
        let r = self.params.rec_thrs.len();
        let k = if self.params.use_cats {
            self.params.cat_ids.len()
        } else {
            1
        };
        let a = self.params.area_rng.len();
        let m = self.params.max_dets.len();

        let num_imgs = self.params.img_ids.len();

        // Build flat list of (k_idx, a_idx, m_idx) work items
        let work_items: Vec<(usize, usize, usize)> = (0..k)
            .flat_map(|k_idx| {
                (0..a).flat_map(move |a_idx| (0..m).map(move |m_idx| (k_idx, a_idx, m_idx)))
            })
            .collect();

        // Each work item produces a set of (index, value) writes for precision, recall, scores
        struct AccResult {
            precision_writes: Vec<(usize, f64)>,
            recall_writes: Vec<(usize, f64)>,
            scores_writes: Vec<(usize, f64)>,
        }

        let acc_idx = AccumulatedEval {
            precision: vec![],
            recall: vec![],
            scores: vec![],
            t,
            r,
            k,
            a,
            m,
        };

        let results: Vec<AccResult> = work_items
            .par_iter()
            .map(|&(k_idx, a_idx, m_idx)| {
                let max_det = self.params.max_dets[m_idx];
                let k_actual = if self.params.use_cats { k_idx } else { 0 };

                let mut all_dt_scores: Vec<f64> = Vec::new();
                let mut all_dt_matches: Vec<Vec<u64>> = vec![Vec::new(); t];
                let mut all_dt_ignore: Vec<Vec<bool>> = vec![Vec::new(); t];
                let mut num_gt = 0usize;

                for img_idx in 0..num_imgs {
                    let eval_idx = k_actual * (a * num_imgs) + a_idx * num_imgs + img_idx;
                    if eval_idx >= self.eval_imgs.len() {
                        continue;
                    }
                    let eval_img = match &self.eval_imgs[eval_idx] {
                        Some(e) => e,
                        None => continue,
                    };

                    let nd = eval_img.dt_scores.len().min(max_det);

                    all_dt_scores.extend_from_slice(&eval_img.dt_scores[..nd]);
                    for t_idx in 0..t {
                        all_dt_matches[t_idx].extend_from_slice(&eval_img.dt_matches[t_idx][..nd]);
                        all_dt_ignore[t_idx].extend_from_slice(&eval_img.dt_ignore[t_idx][..nd]);
                    }

                    num_gt += eval_img.gt_ignore.iter().filter(|&&x| !x).count();
                }

                let mut precision_writes = Vec::new();
                let mut recall_writes = Vec::new();
                let mut scores_writes = Vec::new();

                if num_gt == 0 {
                    return AccResult {
                        precision_writes,
                        recall_writes,
                        scores_writes,
                    };
                }

                // Initialize precision and recall to 0
                for t_idx in 0..t {
                    let recall_idx = ((t_idx * k + k_idx) * a + a_idx) * m + m_idx;
                    recall_writes.push((recall_idx, 0.0));
                    for r_idx in 0..r {
                        let p_idx = acc_idx.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        precision_writes.push((p_idx, 0.0));
                        scores_writes.push((p_idx, 0.0));
                    }
                }

                // Sort by score descending
                let mut inds: Vec<usize> = (0..all_dt_scores.len()).collect();
                inds.sort_by(|&a, &b| {
                    all_dt_scores[b]
                        .partial_cmp(&all_dt_scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let nd = inds.len();

                // Hoist sorted_scores outside the threshold loop (identical across thresholds)
                let sorted_scores: Vec<f64> = inds.iter().map(|&i| all_dt_scores[i]).collect();

                // Pre-allocate buffers reused across thresholds
                let mut tp = vec![0.0f64; nd];
                let mut fp = vec![0.0f64; nd];
                let mut rc = vec![0.0f64; nd];
                let mut pr = vec![0.0f64; nd];

                let num_gt_f = num_gt as f64;

                for t_idx in 0..t {
                    // Fill tp/fp from sorted matches/ignore
                    for (out_idx, &src_idx) in inds.iter().enumerate() {
                        if all_dt_ignore[t_idx][src_idx] {
                            tp[out_idx] = 0.0;
                            fp[out_idx] = 0.0;
                        } else if all_dt_matches[t_idx][src_idx] != 0 {
                            tp[out_idx] = 1.0;
                            fp[out_idx] = 0.0;
                        } else {
                            tp[out_idx] = 0.0;
                            fp[out_idx] = 1.0;
                        }
                    }

                    // Cumulative sum
                    for d in 1..nd {
                        tp[d] += tp[d - 1];
                        fp[d] += fp[d - 1];
                    }

                    // Compute recall and precision in-place
                    for d in 0..nd {
                        rc[d] = tp[d] / num_gt_f;
                        let total = tp[d] + fp[d];
                        pr[d] = if total > 0.0 { tp[d] / total } else { 0.0 };
                    }

                    let recall_idx = ((t_idx * k + k_idx) * a + a_idx) * m + m_idx;
                    if nd > 0 {
                        recall_writes.push((recall_idx, rc[nd - 1]));
                    }

                    // Interpolate precision in-place (make monotonically decreasing from right)
                    for d in (0..nd.saturating_sub(1)).rev() {
                        pr[d] = pr[d].max(pr[d + 1]);
                    }

                    // Two-pointer search: both rc and rec_thrs are sorted ascending
                    let mut rc_ptr = 0;
                    for (r_idx, &rec_thr) in self.params.rec_thrs.iter().enumerate() {
                        let p_idx = acc_idx.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        // Advance pointer to first rc >= rec_thr
                        while rc_ptr < nd && rc[rc_ptr] < rec_thr {
                            rc_ptr += 1;
                        }
                        if rc_ptr < nd {
                            precision_writes.push((p_idx, pr[rc_ptr]));
                            scores_writes.push((p_idx, sorted_scores[rc_ptr]));
                        }
                    }
                }

                AccResult {
                    precision_writes,
                    recall_writes,
                    scores_writes,
                }
            })
            .collect();

        // Merge results into output arrays
        let total = t * r * k * a * m;
        let mut precision = vec![-1.0f64; total];
        let mut scores = vec![-1.0f64; total];
        let total_recall = t * k * a * m;
        let mut recall = vec![-1.0f64; total_recall];

        for result in results {
            for (idx, val) in result.precision_writes {
                precision[idx] = val;
            }
            for (idx, val) in result.recall_writes {
                recall[idx] = val;
            }
            for (idx, val) in result.scores_writes {
                scores[idx] = val;
            }
        }

        self.eval = Some(AccumulatedEval {
            precision,
            recall,
            scores,
            t,
            r,
            k,
            a,
            m,
        });
    }

    /// Print the standard 12-line COCO evaluation summary.
    pub fn summarize(&mut self) {
        let eval = match &self.eval {
            Some(e) => e,
            None => {
                eprintln!("Please run evaluate() and accumulate() first.");
                return;
            }
        };

        let is_kp = self.params.iou_type == IouType::Keypoints;

        // Helper to compute a single summary statistic
        let summarize_stat =
            |ap: bool, iou_thr: Option<f64>, area_lbl: &str, max_det: usize| -> f64 {
                let a_idx = self
                    .params
                    .area_rng_lbl
                    .iter()
                    .position(|l| l == area_lbl)
                    .unwrap_or(0);
                let m_idx = self
                    .params
                    .max_dets
                    .iter()
                    .position(|&d| d == max_det)
                    .unwrap_or(0);

                let t_indices: Vec<usize> = if let Some(thr) = iou_thr {
                    self.params
                        .iou_thrs
                        .iter()
                        .enumerate()
                        .filter(|(_, &t)| (t - thr).abs() < 1e-9)
                        .map(|(i, _)| i)
                        .collect()
                } else {
                    (0..eval.t).collect()
                };

                let mut vals = Vec::new();
                for &t_idx in &t_indices {
                    for k_idx in 0..eval.k {
                        if ap {
                            for r_idx in 0..eval.r {
                                let idx = eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                                let v = eval.precision[idx];
                                if v >= 0.0 {
                                    vals.push(v);
                                }
                            }
                        } else {
                            let idx = eval.recall_idx(t_idx, k_idx, a_idx, m_idx);
                            let v = eval.recall[idx];
                            if v >= 0.0 {
                                vals.push(v);
                            }
                        }
                    }
                }

                if vals.is_empty() {
                    -1.0
                } else {
                    vals.iter().sum::<f64>() / vals.len() as f64
                }
            };

        let max_det_default = *self.params.max_dets.last().unwrap_or(&100);
        let max_det_small = if self.params.max_dets.len() >= 3 {
            self.params.max_dets[0]
        } else {
            max_det_default
        };
        let max_det_med = if self.params.max_dets.len() >= 3 {
            self.params.max_dets[1]
        } else {
            max_det_default
        };

        struct MetricDef {
            ap: bool,
            iou_thr: Option<f64>,
            area_lbl: &'static str,
            max_det: usize,
        }

        let metrics_bbox_segm = |max_d: usize, max_d_s: usize, max_d_m: usize| -> Vec<MetricDef> {
            vec![
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: Some(0.5),
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: Some(0.75),
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "small",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "medium",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "large",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "all",
                    max_det: max_d_s,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "all",
                    max_det: max_d_m,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "small",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "medium",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "large",
                    max_det: max_d,
                },
            ]
        };

        let metrics_kp = |max_d: usize| -> Vec<MetricDef> {
            vec![
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: Some(0.5),
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: Some(0.75),
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "medium",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "large",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: Some(0.5),
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: Some(0.75),
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "medium",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "large",
                    max_det: max_d,
                },
            ]
        };

        let metrics = if is_kp {
            metrics_kp(max_det_default)
        } else {
            metrics_bbox_segm(max_det_default, max_det_small, max_det_med)
        };

        let iou_type_str = match self.params.iou_type {
            IouType::Bbox => "bbox",
            IouType::Segm => "segm",
            IouType::Keypoints => "keypoints",
        };

        let mut stats = Vec::with_capacity(metrics.len());

        for m in &metrics {
            let val = summarize_stat(m.ap, m.iou_thr, m.area_lbl, m.max_det);
            stats.push(val);

            let metric_name = if m.ap {
                "Average Precision"
            } else {
                "Average Recall"
            };
            let metric_short = if m.ap { "AP" } else { "AR" };

            let iou_str = match m.iou_thr {
                Some(thr) => format!("{:.2}", thr),
                None if m.ap => "0.50:0.95".to_string(),
                None => "0.50:0.95".to_string(),
            };

            let area_str = m.area_lbl;
            let det_str = m.max_det;

            let val_str = if val < 0.0 {
                format!("{:0.3}", -1.0)
            } else {
                format!("{:0.3}", val)
            };

            println!(
                " {:<18} @[ IoU={:<9} | area={:>6} | maxDets={:>3} ] = {}",
                format!("{} ({})", metric_name, metric_short),
                iou_str,
                area_str,
                det_str,
                val_str
            );
        }

        println!("Eval type: {}", iou_type_str);
        self.stats = Some(stats);
    }
}
