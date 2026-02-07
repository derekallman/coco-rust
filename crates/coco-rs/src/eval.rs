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
#[allow(dead_code)]
struct EvalImg {
    image_id: u64,
    category_id: u64,
    area_rng: [f64; 2],
    max_det: usize,
    /// Detection matches for each IoU threshold: dt_matches[t][d] = matched gt_id or 0
    dt_matches: Vec<Vec<u64>>,
    /// Ground truth matches for each IoU threshold: gt_matches[t][g] = matched dt_id or 0
    gt_matches: Vec<Vec<u64>>,
    /// Detection scores
    dt_scores: Vec<f64>,
    /// Whether each GT is ignored
    gt_ignore: Vec<bool>,
    /// Whether each detection is ignored per IoU threshold
    dt_ignore: Vec<Vec<bool>>,
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
    eval_imgs: Vec<Option<EvalImg>>,
    ious: HashMap<(u64, u64), Vec<Vec<f64>>>,
    pub eval: Option<AccumulatedEval>,
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
            IouType::Segm => Self::compute_segm_iou_static(coco_gt, coco_dt, &dt_anns, &gt_anns),
            IouType::Bbox => Self::compute_bbox_iou_static(coco_gt, coco_dt, &dt_anns, &gt_anns),
            IouType::Keypoints => {
                Self::compute_oks_static(coco_gt, coco_dt, params, &dt_anns, &gt_anns)
            }
        }
    }

    fn get_anns_static(coco: &COCO, params: &Params, img_id: u64, cat_id: u64) -> Vec<u64> {
        if params.use_cats {
            coco.get_ann_ids(&[img_id], &[cat_id], None, None)
        } else {
            coco.get_ann_ids(&[img_id], &[], None, None)
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
        let vars: Vec<f64> = sigmas.iter().map(|s| 2.0 * s * s).collect();

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
            let gt_area = gt_ann.area.unwrap_or(0.0);
            if gt_area == 0.0 {
                continue;
            }

            for (i, dt_ann) in dt_anns.iter().enumerate() {
                let dt_kpts = match &dt_ann.keypoints {
                    Some(k) => k,
                    None => continue,
                };

                let mut oks = 0.0;
                let mut valid = 0;

                for (k, &var_k) in vars.iter().enumerate().take(num_kpts) {
                    let gx = gt_kpts.get(k * 3).copied().unwrap_or(0.0);
                    let gy = gt_kpts.get(k * 3 + 1).copied().unwrap_or(0.0);
                    let gv = gt_kpts.get(k * 3 + 2).copied().unwrap_or(0.0);
                    let dx = dt_kpts.get(k * 3).copied().unwrap_or(0.0);
                    let dy = dt_kpts.get(k * 3 + 1).copied().unwrap_or(0.0);

                    if gv > 0.0 {
                        let dist_sq = (dx - gx).powi(2) + (dy - gy).powi(2);
                        let e = dist_sq / (2.0 * gt_area * var_k);
                        oks += (-e).exp();
                        valid += 1;
                    }
                }

                if valid > 0 {
                    result[i][j] = oks / valid as f64;
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
        let gt_ignore: Vec<bool> = gt_anns
            .iter()
            .map(|ann| {
                let a = ann.area.unwrap_or(0.0);
                ann.iscrowd || a < area_rng[0] || a > area_rng[1]
            })
            .collect();

        // Sort GT: non-ignored first, then ignored
        let mut gt_order: Vec<usize> = (0..gt_anns.len()).collect();
        gt_order.sort_by_key(|&i| gt_ignore[i] as u8);
        let gt_ignore_sorted: Vec<bool> = gt_order.iter().map(|&i| gt_ignore[i]).collect();
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
            // We need to map from sorted DT/GT indices to original indices in iou_mat
            // The iou_mat uses the same ordering as dt_ids / gt_ids (which come from get_anns)
            // We need to find the index in dt_ids/gt_ids for each ann

            let dt_id_to_iou_idx: HashMap<u64, usize> =
                dt_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            let gt_id_to_iou_idx: HashMap<u64, usize> =
                gt_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

            for (t_idx, &iou_thr) in params.iou_thrs.iter().enumerate() {
                for (di, dt_ann) in dt_anns.iter().enumerate() {
                    let mut best_iou = iou_thr; // minimum threshold
                    let mut best_gi: Option<usize> = None;

                    let dt_iou_idx = match dt_id_to_iou_idx.get(&dt_ann.id) {
                        Some(&idx) => idx,
                        None => continue,
                    };

                    if dt_iou_idx >= iou_mat.len() || iou_mat[dt_iou_idx].is_empty() {
                        continue;
                    }

                    for (gi_sorted, &gi_orig) in gt_order.iter().enumerate() {
                        let gt_iou_idx = match gt_id_to_iou_idx.get(&gt_anns[gi_orig].id) {
                            Some(&idx) => idx,
                            None => continue,
                        };

                        // Skip already matched non-crowd GTs
                        if gt_matches[t_idx][gi_sorted] != 0 && !gt_ignore_sorted[gi_sorted] {
                            continue;
                        }

                        if gt_iou_idx >= iou_mat[dt_iou_idx].len() {
                            continue;
                        }

                        let iou_val = iou_mat[dt_iou_idx][gt_iou_idx];

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

                for t_idx in 0..t {
                    let sorted_matches: Vec<u64> =
                        inds.iter().map(|&i| all_dt_matches[t_idx][i]).collect();
                    let sorted_ignore: Vec<bool> =
                        inds.iter().map(|&i| all_dt_ignore[t_idx][i]).collect();

                    let nd = sorted_matches.len();
                    let mut tp = vec![0.0f64; nd];
                    let mut fp = vec![0.0f64; nd];

                    for d in 0..nd {
                        if sorted_ignore[d] {
                            continue;
                        }
                        if sorted_matches[d] != 0 {
                            tp[d] = 1.0;
                        } else {
                            fp[d] = 1.0;
                        }
                    }

                    for d in 1..nd {
                        tp[d] += tp[d - 1];
                        fp[d] += fp[d - 1];
                    }

                    let rc: Vec<f64> = tp.iter().map(|&x| x / num_gt as f64).collect();
                    let pr: Vec<f64> = tp
                        .iter()
                        .zip(fp.iter())
                        .map(|(&tp_v, &fp_v)| {
                            if tp_v + fp_v > 0.0 {
                                tp_v / (tp_v + fp_v)
                            } else {
                                0.0
                            }
                        })
                        .collect();

                    let recall_idx = ((t_idx * k + k_idx) * a + a_idx) * m + m_idx;
                    if !rc.is_empty() {
                        recall_writes.push((recall_idx, *rc.last().unwrap()));
                    }

                    let mut pr_interp = pr.clone();
                    for d in (0..pr_interp.len().saturating_sub(1)).rev() {
                        pr_interp[d] = pr_interp[d].max(pr_interp[d + 1]);
                    }

                    let sorted_scores: Vec<f64> = inds.iter().map(|&i| all_dt_scores[i]).collect();

                    for (r_idx, &rec_thr) in self.params.rec_thrs.iter().enumerate() {
                        let p_idx = acc_idx.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        let pos = rc.iter().position(|&r| r >= rec_thr);
                        if let Some(pos) = pos {
                            precision_writes.push((p_idx, pr_interp[pos]));
                            scores_writes.push((p_idx, sorted_scores[pos]));
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
    pub fn summarize(&self) {
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

        for m in &metrics {
            let val = summarize_stat(m.ap, m.iou_thr, m.area_lbl, m.max_det);

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
    }
}
