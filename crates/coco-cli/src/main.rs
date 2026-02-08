use std::path::PathBuf;

use clap::Parser;
use coco_rs::params::IouType;
use coco_rs::{COCOeval, COCO};

#[derive(Parser)]
#[command(name = "coco-eval")]
#[command(
    about = "COCO evaluation tool — compute AP/AR metrics for object detection, segmentation, and keypoints"
)]
struct Cli {
    /// Path to ground truth annotations JSON file
    #[arg(long)]
    gt: PathBuf,

    /// Path to detection results JSON file
    #[arg(long)]
    dt: PathBuf,

    /// IoU type: bbox, segm, or keypoints
    #[arg(long, default_value = "bbox")]
    iou_type: String,

    /// Filter to specific image IDs (comma-separated)
    #[arg(long, value_delimiter = ',')]
    img_ids: Option<Vec<u64>>,

    /// Filter to specific category IDs (comma-separated)
    #[arg(long, value_delimiter = ',')]
    cat_ids: Option<Vec<u64>>,

    /// Max detections per image (comma-separated, e.g., "1,10,100")
    #[arg(long, value_delimiter = ',')]
    max_dets: Option<Vec<usize>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let iou_type = match cli.iou_type.as_str() {
        "bbox" => IouType::Bbox,
        "segm" => IouType::Segm,
        "keypoints" => IouType::Keypoints,
        other => {
            eprintln!(
                "Unknown IoU type: '{}'. Use bbox, segm, or keypoints.",
                other
            );
            std::process::exit(1);
        }
    };

    eprintln!("Loading ground truth from {:?}...", cli.gt);
    let coco_gt = COCO::new(&cli.gt)?;

    eprintln!("Loading detections from {:?}...", cli.dt);
    let coco_dt = coco_gt.load_res(&cli.dt)?;

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, iou_type);

    if let Some(img_ids) = cli.img_ids {
        coco_eval.params.img_ids = img_ids;
    }
    if let Some(cat_ids) = cli.cat_ids {
        coco_eval.params.cat_ids = cat_ids;
    }
    if let Some(max_dets) = cli.max_dets {
        coco_eval.params.max_dets = max_dets;
    }

    eprintln!("Evaluating...");
    coco_eval.evaluate();

    eprintln!("Accumulating...");
    coco_eval.accumulate();

    coco_eval.summarize();

    // Print machine-readable stats line for parity testing
    if let Some(ref stats) = coco_eval.stats {
        let stats_strs: Vec<String> = stats.iter().map(|v| format!("{:.15}", v)).collect();
        println!("stats: [{}]", stats_strs.join(", "));
    }

    Ok(())
}
