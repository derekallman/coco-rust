use serde::{Deserialize, Deserializer, Serialize};

/// Top-level COCO dataset structure.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Dataset {
    #[serde(default)]
    pub info: Option<Info>,
    #[serde(default)]
    pub images: Vec<Image>,
    #[serde(default)]
    pub annotations: Vec<Annotation>,
    #[serde(default)]
    pub categories: Vec<Category>,
    #[serde(default)]
    pub licenses: Vec<License>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Info {
    #[serde(default)]
    pub year: Option<u32>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub contributor: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub date_created: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Image {
    pub id: u64,
    #[serde(default)]
    pub file_name: String,
    pub height: u32,
    pub width: u32,
    #[serde(default)]
    pub license: Option<u64>,
    #[serde(default)]
    pub coco_url: Option<String>,
    #[serde(default)]
    pub flickr_url: Option<String>,
    #[serde(default)]
    pub date_captured: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Annotation {
    #[serde(default)]
    pub id: u64,
    pub image_id: u64,
    pub category_id: u64,
    #[serde(default)]
    pub bbox: Option<[f64; 4]>,
    #[serde(default)]
    pub area: Option<f64>,
    #[serde(default)]
    pub segmentation: Option<Segmentation>,
    #[serde(default, deserialize_with = "deserialize_iscrowd")]
    pub iscrowd: bool,
    #[serde(default)]
    pub keypoints: Option<Vec<f64>>,
    #[serde(default)]
    pub num_keypoints: Option<u32>,
    /// Detection score (present only in result annotations).
    #[serde(default)]
    pub score: Option<f64>,
}

fn deserialize_iscrowd<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum IsCrowd {
        Bool(bool),
        Int(u8),
    }
    match IsCrowd::deserialize(deserializer)? {
        IsCrowd::Bool(b) => Ok(b),
        IsCrowd::Int(i) => Ok(i != 0),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Segmentation {
    /// Polygon format: list of polygons, each a flat list of [x, y, x, y, ...] coordinates.
    Polygon(Vec<Vec<f64>>),
    /// Compressed RLE format (as stored in COCO JSON results).
    CompressedRle { size: [u32; 2], counts: String },
    /// Uncompressed RLE format.
    UncompressedRle { size: [u32; 2], counts: Vec<u32> },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Category {
    pub id: u64,
    pub name: String,
    #[serde(default)]
    pub supercategory: Option<String>,
    #[serde(default)]
    pub skeleton: Option<Vec<[u32; 2]>>,
    #[serde(default)]
    pub keypoints: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct License {
    #[serde(default)]
    pub id: u64,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
}

/// Run-length encoding for masks.
#[derive(Debug, Clone, PartialEq)]
pub struct Rle {
    pub h: u32,
    pub w: u32,
    /// Run counts: alternating runs of 0s and 1s, starting with 0s.
    pub counts: Vec<u32>,
}
