//! Screen capture, change detection and model-input assembly.
//!
//! Layout / placement decision captured to make the eventual real-world tuning
//! easier to reason about:
//!
//! - The immediate live path no longer round-trips through a lossless PNG:
//!   the primary screen frame is downsampled in memory, a full-screen JPEG is
//!   produced for the default model input, and lightweight grayscale
//!   signatures are derived straight from that same downsample.
//! - Change detection uses two grids: a coarse 4×4 grid for the cheap "did
//!   anything move" decision and a fine 16×9 grid to locate *where*.
//! - A localised change feeds the model a high-density crop taken from the
//!   original (source) resolution instead of the downsampled thumbnail, so
//!   small terminal / chat / table text keeps more real pixels.

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image::{
    codecs::jpeg::JpegEncoder, imageops::FilterType, DynamicImage, GrayImage, ImageBuffer, Luma,
    RgbImage,
};
use xcap::Monitor;

pub const CAPTURE_WIDTH: u32 = 768;
pub const CAPTURE_HEIGHT: u32 = 432;
pub const JPEG_QUALITY: u8 = 78;

pub const COARSE_COLS: u32 = 4;
pub const COARSE_ROWS: u32 = 4;
pub const FINE_COLS: u32 = 16;
pub const FINE_ROWS: u32 = 9;

/// Minimum bit-flips inside one fine cell to count it as changed.
pub const FINE_CELL_THRESHOLD: u32 = 6;
/// Minimum bit-flips inside one coarse cell to count it as changed.
pub const COARSE_CELL_THRESHOLD: u32 = 10;

/// A full frame plus everything needed to build model inputs later.
pub struct CapturedFrame {
    /// Original screen pixels, kept for high-density crops.
    pub source: DynamicImage,
    /// Downsampled RGB thumbnail used as the default full-screen input.
    pub thumb: RgbImage,
    pub thumb_width: u32,
    pub thumb_height: u32,
    /// Per-cell signatures (coarse then fine), computed from the thumbnail.
    pub coarse: Vec<u64>,
    pub fine: Vec<u64>,
}

/// One image part for the OpenAI-compatible vision request.
#[derive(Clone)]
pub struct ImageInput {
    pub mime: &'static str,
    pub base64: String,
}

/// The assembled model input for one recognition round.
#[derive(Clone)]
pub struct ScreenFrame {
    pub images: Vec<ImageInput>,
    /// Set when a crop was produced (`include_full` also present), useful for
    /// benchmark logs to tell full-screen from crop requests apart.
    pub source_rect: Option<(u32, u32, u32, u32)>,
}

impl ScreenFrame {
    pub fn full_screen(&self) -> bool {
        self.source_rect.is_none()
    }
}

/// Captures the primary monitor, downsamples in memory and computes both
/// change-detection grids.  There is intentionally no PNG round-trip here:
/// the JPEG for the model is encoded directly from the in-memory downsample.
///
/// On Windows this uses Graphics Capture (`xcap` `wgc` feature). Combined with
/// `WDA_EXCLUDEFROMCAPTURE` on Baodou's own HWNDs, the pet / bubble / launcher
/// are omitted and the desktop behind them is what the model sees.
pub fn capture_primary() -> Result<CapturedFrame, String> {
    let monitor = Monitor::all()
        .map_err(|e| format!("枚举显示器失败：{e}"))?
        .into_iter()
        .find(|m| m.is_primary().unwrap_or(false))
        .ok_or_else(|| "没有检测到主显示器".to_string())?;
    let image = monitor
        .capture_image()
        .map_err(|e| format!("屏幕采集失败：{e}"))?;

    let source = DynamicImage::ImageRgba8(image);
    let thumb = source
        .resize(CAPTURE_WIDTH, CAPTURE_HEIGHT, FilterType::Triangle)
        .to_rgb8();

    let gray = to_luma(&thumb);
    Ok(CapturedFrame {
        source,
        thumb,
        thumb_width: CAPTURE_WIDTH,
        thumb_height: CAPTURE_HEIGHT,
        coarse: partition_signatures(&gray, COARSE_COLS, COARSE_ROWS),
        fine: partition_signatures(&gray, FINE_COLS, FINE_ROWS),
    })
}

impl CapturedFrame {
    /// Builds the images actually sent to the model for this round.
    ///
    /// - `Full` sends the full-screen thumbnail.
    /// - `Crop(rect)` maps the thumb-space change rectangle back to the
    ///   source resolution, expands it for context, and sends a high-density
    ///   letterboxed crop.
    /// - `FullPlusCrop(rect)` sends the thumbnail plus that crop in one
    ///   request (opt-in; used only when multi-image input is verified stable).
    pub fn build_input(
        &self,
        rect: Option<(u32, u32, u32, u32)>,
        multi_image: bool,
    ) -> Result<ScreenFrame, String> {
        let mut images = Vec::new();

        if let Some(rect) = rect {
            if multi_image {
                images.push(encode_jpeg_image(&self.thumb)?);
            }
            let (x, y, w, h) = self.source_crop_rect(rect);
            let cropped = self.source.crop_imm(x, y, w, h).to_rgb8();
            images.push(encode_jpeg_image(&letterbox(
                &cropped,
                CAPTURE_WIDTH,
                CAPTURE_HEIGHT,
            ))?);
            return Ok(ScreenFrame {
                images,
                source_rect: Some((x, y, w, h)),
            });
        }

        images.push(encode_jpeg_image(&self.thumb)?);
        Ok(ScreenFrame {
            images,
            source_rect: None,
        })
    }

    /// Maps a thumb-space rectangle to the source resolution with surrounding
    /// context, clamped to the frame bounds.
    fn source_crop_rect(&self, rect: (u32, u32, u32, u32)) -> (u32, u32, u32, u32) {
        let (x, y, w, h) = rect;
        let sx = self.source.width() as f64 / self.thumb_width as f64;
        let sy = self.source.height() as f64 / self.thumb_height as f64;
        let mut x0 = (x as f64 * sx).floor() as i64;
        let mut y0 = (y as f64 * sy).floor() as i64;
        let mut x1 = ((x + w) as f64 * sx).ceil() as i64;
        let mut y1 = ((y + h) as f64 * sy).ceil() as i64;

        let side = (x1 - x0).max(y1 - y0) as f64 * 0.3;
        let pad = (side as i64).clamp(48, 640);
        x0 = (x0 - pad).max(0);
        y0 = (y0 - pad).max(0);
        x1 = (x1 + pad).min(self.source.width() as i64);
        y1 = (y1 + pad).min(self.source.height() as i64);
        (x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32)
    }
}

fn to_luma(rgb: &RgbImage) -> GrayImage {
    ImageBuffer::from_fn(rgb.width(), rgb.height(), |x, y| {
        let p = rgb.get_pixel(x, y);
        // ITU-R BT.601 weights; integer math keeps this cheap.
        let luma = ((p[0] as u32 * 77 + p[1] as u32 * 150 + p[2] as u32 * 29) / 256) as u8;
        Luma([luma])
    })
}

/// Computes one 64-bit signature per grid cell.  Each cell is sampled on an
/// 8×8 lattice and each sample is a bit telling whether the pixel is at or
/// above the cell's mean luminance.  Cheap, robust to cursor blink and small
/// text anti-aliasing because it only measures *distribution of bright/dark*.
pub fn partition_signatures(gray: &GrayImage, cols: u32, rows: u32) -> Vec<u64> {
    let w = gray.width();
    let h = gray.height();
    let mut signatures = Vec::with_capacity((cols * rows) as usize);
    for row in 0..rows {
        for col in 0..cols {
            let x0 = col * w / cols;
            let y0 = row * h / rows;
            let x1 = ((col + 1) * w / cols).max(x0 + 1);
            let y1 = ((row + 1) * h / rows).max(y0 + 1);
            signatures.push(cell_signature(gray, x0, y0, x1 - x0, y1 - y0));
        }
    }
    signatures
}

fn cell_signature(gray: &GrayImage, x0: u32, y0: u32, cw: u32, ch: u32) -> u64 {
    let mut samples = [0_u16; 64];
    let mut mean: u32 = 0;
    for sy in 0..8 {
        let py = y0 + (ch.saturating_sub(1)) * sy / 7;
        for sx in 0..8 {
            let px = x0 + (cw.saturating_sub(1)) * sx / 7;
            let value = gray.get_pixel(px, py)[0];
            samples[(sy * 8 + sx) as usize] = u16::from(value);
            mean += u32::from(value);
        }
    }
    mean /= 64;
    let mut signature = 0_u64;
    for (index, sample) in samples.iter().enumerate() {
        if u32::from(*sample) >= mean {
            signature |= 1_u64 << index;
        }
    }
    signature
}

/// Indices of cells whose signature differed from the previous frame by at
/// least `threshold` bits.
pub fn changed_cells(previous: &[u64], current: &[u64], threshold: u32) -> Vec<usize> {
    let max = previous.len().min(current.len());
    (0..max)
        .filter(|index| (previous[*index] ^ current[*index]).count_ones() >= threshold)
        .collect()
}

/// Union bounding box (thumb coordinates) of a set of changed fine cells.
pub fn change_bbox(
    changed: &[usize],
    cols: u32,
    rows: u32,
    width: u32,
    height: u32,
) -> Option<(u32, u32, u32, u32)> {
    if changed.is_empty() {
        return None;
    }
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    for index in changed {
        let col = (*index as u32) % cols;
        let row = (*index as u32) / cols;
        let x0 = col * width / cols;
        let y0 = row * height / rows;
        let x1 = ((col + 1) * width / cols).max(x0 + 1);
        let y1 = ((row + 1) * height / rows).max(y0 + 1);
        min_x = min_x.min(x0);
        min_y = min_y.min(y0);
        max_x = max_x.max(x1);
        max_y = max_y.max(y1);
    }
    Some((min_x, min_y, max_x - min_x, max_y - min_y))
}

/// Fraction of the screen covered by a bounding box.
pub fn area_fraction(rect: &(u32, u32, u32, u32), width: u32, height: u32) -> f64 {
    let area = rect.2 as f64 * rect.3 as f64;
    area / (width as f64 * height as f64)
}

fn encode_jpeg_image(image: &RgbImage) -> Result<ImageInput, String> {
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY)
        .encode_image(image)
        .map_err(|e| format!("图像编码失败：{e}"))?;
    Ok(ImageInput {
        mime: "data:image/jpeg;base64,",
        base64: BASE64.encode(jpeg),
    })
}

/// Scales `frame` into a `width × height` canvas without distortion, centring
/// it on a neutral dark background.  Vision models read letterboxed crops far
/// more reliably than aspect-distorted resizes.
fn letterbox(frame: &RgbImage, width: u32, height: u32) -> RgbImage {
    let scale = (width as f32 / frame.width() as f32).min(height as f32 / frame.height() as f32);
    let new_width = ((frame.width() as f32 * scale).round() as u32).max(1);
    let new_height = ((frame.height() as f32 * scale).round() as u32).max(1);
    let resized = image::imageops::resize(frame, new_width, new_height, FilterType::Triangle);
    let mut canvas = RgbImage::from_pixel(width, height, image::Rgb([16, 16, 16]));
    let ox = (width - new_width) / 2;
    let oy = (height - new_height) / 2;
    image::imageops::overlay(&mut canvas, &resized, ox as i64, oy as i64);
    canvas
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(rgb: &RgbImage) -> GrayImage {
        to_luma(rgb)
    }

    fn solid_rgb(width: u32, height: u32, value: u8) -> RgbImage {
        RgbImage::from_pixel(width, height, image::Rgb([value, value, value]))
    }

    #[test]
    fn stable_frame_has_no_changed_cells() {
        let image = solid_rgb(768, 432, 120);
        let a = &gray(&image);
        let coarse_a = partition_signatures(a, COARSE_COLS, COARSE_ROWS);
        let a2 = &gray(&image);
        let coarse_b = partition_signatures(a2, COARSE_COLS, COARSE_ROWS);
        assert!(changed_cells(&coarse_a, &coarse_b, COARSE_CELL_THRESHOLD).is_empty());
    }

    #[test]
    fn bright_region_counts_as_changed() {
        let mut image = solid_rgb(768, 432, 120);
        for x in 0..96 {
            for y in 0..96 {
                image.put_pixel(x, y, image::Rgb([250, 250, 250]));
            }
        }
        let fine_a = partition_signatures(&gray(&image), FINE_COLS, FINE_ROWS);
        let mut image2 = solid_rgb(768, 432, 120);
        for x in 0..96 {
            for y in 0..96 {
                image2.put_pixel(x + 300, y + 20, image::Rgb([245, 245, 245]));
            }
        }
        let fine_b = partition_signatures(&gray(&image2), FINE_COLS, FINE_ROWS);
        let changed = changed_cells(&fine_a, &fine_b, FINE_CELL_THRESHOLD);
        assert!(!changed.is_empty());
        let bbox = change_bbox(&changed, FINE_COLS, FINE_ROWS, 768, 432).expect("bbox");
        assert!(
            bbox.0 >= 200 && bbox.0 <= 400,
            "crop lands near the moved block"
        );
    }

    #[test]
    fn bbox_covers_lower_right_change() {
        let changed: Vec<usize> = (FINE_COLS * FINE_ROWS - 4..FINE_COLS * FINE_ROWS)
            .map(|index| index as usize)
            .collect();
        let bbox = change_bbox(&changed, FINE_COLS, FINE_ROWS, 768, 432).expect("bbox");
        assert!(bbox.0 + bbox.2 >= 700);
        assert!(bbox.1 + bbox.3 >= 390);
    }
}
