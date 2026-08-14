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
    RgbImage, RgbaImage,
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

/// Cached desktop pixels used to paint over Baodou windows so the model
/// never sees the pet / bubble / launcher. The live grab keeps those
/// windows visible to the user.
pub struct DesktopBackdrop {
    image: RgbaImage,
    origin_x: i32,
    origin_y: i32,
}

impl DesktopBackdrop {
    /// Grabs the primary monitor while `hwnds` are DWM-cloaked. Call this
    /// once before showing the floating window so the cloak never flickers
    /// during the recognition loop.
    pub fn capture_excluding(hwnds: &[isize]) -> Result<Self, String> {
        ensure_exclusion_targets(hwnds)?;
        let monitor = primary_monitor()?;
        let origin_x = monitor.x().unwrap_or(0);
        let origin_y = monitor.y().unwrap_or(0);
        let image = {
            let _cloak = WindowCloak::apply(hwnds)?;
            monitor
                .capture_image()
                .map_err(|e| format!("屏幕采集失败：{e}"))?
        };
        Ok(Self {
            image,
            origin_x,
            origin_y,
        })
    }

    fn covers(&self, width: u32, height: u32) -> bool {
        self.image.width() == width && self.image.height() == height
    }

    /// Paints cached desktop over each visible Baodou window, then stores the
    /// cleaned frame so the next round can refresh pixels that are no longer
    /// covered (window moved / resized / hidden).
    fn refresh_and_paint(&mut self, frame: &mut RgbaImage, hwnds: &[isize]) -> Result<(), String> {
        self.paint_over(frame, hwnds)?;
        self.image.clone_from(frame);
        Ok(())
    }

    /// Copies the cached desktop into each visible Baodou window rectangle.
    fn paint_over(&self, frame: &mut RgbaImage, hwnds: &[isize]) -> Result<(), String> {
        let frame_w = frame.width() as i32;
        let frame_h = frame.height() as i32;
        for hwnd in hwnds.iter().copied().filter(|hwnd| *hwnd != 0) {
            let Some(rect) = visible_window_rect(hwnd)? else {
                continue;
            };
            let x0 = (rect.0 - self.origin_x).max(0);
            let y0 = (rect.1 - self.origin_y).max(0);
            let x1 = (rect.2 - self.origin_x).min(frame_w);
            let y1 = (rect.3 - self.origin_y).min(frame_h);
            if x0 >= x1 || y0 >= y1 {
                continue;
            }
            blit_rect(
                frame,
                &self.image,
                x0 as u32,
                y0 as u32,
                (x1 - x0) as u32,
                (y1 - y0) as u32,
            );
        }
        Ok(())
    }
}

/// Captures the primary monitor and paints `backdrop` over `hwnds` so those
/// windows stay on screen for the user while the model sees the cached desktop
/// behind them. If no valid backdrop exists, the windows are DWM-cloaked for
/// the grab; an unfiltered image is never returned.
pub fn capture_primary_excluding(
    hwnds: &[isize],
    backdrop: Option<&mut DesktopBackdrop>,
) -> Result<CapturedFrame, String> {
    ensure_exclusion_targets(hwnds)?;
    let mut image = grab_primary()?;
    if let Some(backdrop) = backdrop {
        if backdrop.covers(image.width(), image.height()) {
            backdrop.refresh_and_paint(&mut image, hwnds)?;
            return frame_from_rgba(image);
        }
    }

    // Never fall back to an unfiltered frame. A missing/stale backdrop is
    // unusual (startup capture failure or display-mode change), but allowing
    // the raw screenshot through here exposes Baodou's own response to the
    // vision model and creates a self-reinforcing feedback loop.
    let image = grab_primary_excluding(hwnds)?;
    frame_from_rgba(image)
}

fn ensure_exclusion_targets(hwnds: &[isize]) -> Result<(), String> {
    if hwnds.iter().any(|hwnd| *hwnd != 0) {
        Ok(())
    } else {
        Err("无法获取应用窗口句柄，已阻止发送未经遮罩的屏幕截图".into())
    }
}

fn primary_monitor() -> Result<Monitor, String> {
    Monitor::all()
        .map_err(|e| format!("枚举显示器失败：{e}"))?
        .into_iter()
        .find(|m| m.is_primary().unwrap_or(false))
        .ok_or_else(|| "没有检测到主显示器".to_string())
}

fn grab_primary() -> Result<RgbaImage, String> {
    primary_monitor()?
        .capture_image()
        .map_err(|e| format!("屏幕采集失败：{e}"))
}

fn grab_primary_excluding(hwnds: &[isize]) -> Result<RgbaImage, String> {
    let _cloak = WindowCloak::apply(hwnds)?;
    grab_primary()
}

fn frame_from_rgba(image: RgbaImage) -> Result<CapturedFrame, String> {
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

fn blit_rect(dst: &mut RgbaImage, src: &RgbaImage, x: u32, y: u32, width: u32, height: u32) {
    let max_w = dst.width().min(src.width());
    let max_h = dst.height().min(src.height());
    if x >= max_w || y >= max_h {
        return;
    }
    let width = width.min(max_w - x);
    let height = height.min(max_h - y);
    for row in y..y + height {
        let dst_start = ((row * dst.width() + x) * 4) as usize;
        let src_start = ((row * src.width() + x) * 4) as usize;
        let len = (width * 4) as usize;
        dst.as_mut()[dst_start..dst_start + len]
            .copy_from_slice(&src.as_raw()[src_start..src_start + len]);
    }
}

/// Temporarily cloaks HWNDs via `DWMWA_CLOAK`. Used only while seeding
/// [`DesktopBackdrop`] so the first grab does not contain Baodou chrome.
/// Cloaked windows leave DWM composition (and therefore WGC / DXGI frames)
/// but keep their WebView2 swapchain intact; dropping restores them.
struct WindowCloak {
    hwnds: Vec<isize>,
}

impl WindowCloak {
    fn apply(hwnds: &[isize]) -> Result<Self, String> {
        let hwnds: Vec<isize> = hwnds.iter().copied().filter(|hwnd| *hwnd != 0).collect();
        ensure_exclusion_targets(&hwnds)?;
        let mut applied = Vec::with_capacity(hwnds.len());
        for hwnd in &hwnds {
            if let Err(error) = set_cloaked(*hwnd, true) {
                for applied_hwnd in applied.iter().copied() {
                    let _ = set_cloaked(applied_hwnd, false);
                }
                flush_dwm();
                return Err(error);
            }
            applied.push(*hwnd);
        }
        if !hwnds.is_empty() {
            flush_dwm();
        }
        Ok(Self { hwnds })
    }
}

impl Drop for WindowCloak {
    fn drop(&mut self) {
        for hwnd in &self.hwnds {
            let _ = set_cloaked(*hwnd, false);
        }
        if !self.hwnds.is_empty() {
            flush_dwm();
        }
    }
}

#[cfg(windows)]
fn set_cloaked(hwnd: isize, cloak: bool) -> Result<(), String> {
    use windows::Win32::{
        Foundation::HWND,
        Graphics::Dwm::{DwmSetWindowAttribute, DWMWA_CLOAK},
    };

    let value: i32 = i32::from(cloak);
    let handle = HWND(hwnd as *mut core::ffi::c_void);
    unsafe {
        DwmSetWindowAttribute(
            handle,
            DWMWA_CLOAK,
            std::ptr::from_ref(&value).cast(),
            std::mem::size_of::<i32>() as u32,
        )
        .map_err(|error| format!("无法从截图中排除应用窗口：{error}"))
    }
}

#[cfg(not(windows))]
fn set_cloaked(_hwnd: isize, _cloak: bool) -> Result<(), String> {
    Err("当前平台不支持应用窗口截图排除".into())
}

#[cfg(windows)]
fn flush_dwm() {
    use windows::Win32::Graphics::Dwm::DwmFlush;
    unsafe {
        let _ = DwmFlush();
    }
}

#[cfg(not(windows))]
fn flush_dwm() {}

/// Screen-space bounds of a visible, non-minimized HWND as
/// `(left, top, right, bottom)`.
#[cfg(windows)]
fn visible_window_rect(hwnd: isize) -> Result<Option<(i32, i32, i32, i32)>, String> {
    use windows::Win32::{
        Foundation::HWND,
        UI::WindowsAndMessaging::{GetWindowRect, IsIconic, IsWindow, IsWindowVisible},
    };

    let handle = HWND(hwnd as *mut core::ffi::c_void);
    unsafe {
        if !IsWindow(Some(handle)).as_bool() {
            return Err("应用窗口句柄已失效，已阻止发送未经遮罩的屏幕截图".into());
        }
        if !IsWindowVisible(handle).as_bool() || IsIconic(handle).as_bool() {
            return Ok(None);
        }
        let mut rect = windows::Win32::Foundation::RECT::default();
        GetWindowRect(handle, &mut rect)
            .map_err(|error| format!("无法读取应用窗口范围：{error}"))?;
        if rect.right <= rect.left || rect.bottom <= rect.top {
            return Err("应用窗口范围无效，已阻止发送未经遮罩的屏幕截图".into());
        }
        Ok(Some((rect.left, rect.top, rect.right, rect.bottom)))
    }
}

#[cfg(not(windows))]
fn visible_window_rect(_hwnd: isize) -> Result<Option<(i32, i32, i32, i32)>, String> {
    Err("当前平台不支持应用窗口截图排除".into())
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
    fn blit_rect_copies_overlapping_block() {
        let mut dst = RgbaImage::from_pixel(8, 8, image::Rgba([1, 2, 3, 255]));
        let src = RgbaImage::from_pixel(8, 8, image::Rgba([9, 8, 7, 255]));
        blit_rect(&mut dst, &src, 2, 3, 3, 2);
        assert_eq!(dst.get_pixel(2, 3), src.get_pixel(2, 3));
        assert_eq!(dst.get_pixel(4, 4), src.get_pixel(4, 4));
        assert_eq!(*dst.get_pixel(1, 3), image::Rgba([1, 2, 3, 255]));
        assert_eq!(*dst.get_pixel(2, 2), image::Rgba([1, 2, 3, 255]));
    }

    #[test]
    fn capture_exclusion_refuses_an_empty_target_list() {
        assert!(ensure_exclusion_targets(&[]).is_err());
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
