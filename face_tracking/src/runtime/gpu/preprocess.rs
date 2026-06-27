use image::DynamicImage;
use ndarray::Array4;

pub struct PreprocessResult {
    pub tensor: Array4<f32>,
    pub scale: f32,
    pub offset_x: f32,
    pub offset_y: f32,
}

pub fn preprocess(frame: &[u8], width: u32, height: u32) -> Result<PreprocessResult, String> {
    // 1. Convert raw bytes to image
    let img =
        image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(width, height, frame.to_vec())
            .ok_or_else(|| "Failed to create image buffer".to_string())?;

    let dynamic_img = DynamicImage::ImageRgba8(img);

    // 2. Letterbox to 224x224
    let scale = (224.0 / width as f32).min(224.0 / height as f32);
    let new_w = (width as f32 * scale).round() as u32;
    let new_h = (height as f32 * scale).round() as u32;
    let offset_x = (224.0 - new_w as f32) / 2.0;
    let offset_y = (224.0 - new_h as f32) / 2.0;

    let resized = dynamic_img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
    let mut padded = image::DynamicImage::new_rgb8(224, 224).to_rgb8();
    image::imageops::overlay(&mut padded, &resized.to_rgb8(), offset_x as i64, offset_y as i64);

    // 3. Normalize to SCRFD format and convert to NCHW tensor
    let mut array = Array4::<f32>::zeros((1, 3, 224, 224));
    for (x, y, pixel) in padded.enumerate_pixels() {
        // Fix: Engine provides BGRA, so pixel[0] is B, pixel[2] is R.
        let b = (pixel[0] as f32 - 127.5) / 128.0;
        let g = (pixel[1] as f32 - 127.5) / 128.0;
        let r = (pixel[2] as f32 - 127.5) / 128.0;

        array[[0, 0, y as usize, x as usize]] = r;
        array[[0, 1, y as usize, x as usize]] = g;
        array[[0, 2, y as usize, x as usize]] = b;
    }

    Ok(PreprocessResult {
        tensor: array,
        scale,
        offset_x,
        offset_y,
    })
}
