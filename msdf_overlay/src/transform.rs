use crate::layout::{LayoutGlyph, Anchor};

#[derive(Debug, Clone)]
pub struct TransformedGlyph {
    pub char_code: char,
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub uv: [f32; 4],
    pub scale: f32,
    pub rotation: f32,
}

pub fn apply_transform(
    glyphs: &[LayoutGlyph],
    anchor: &Anchor,
    transformed: &mut Vec<TransformedGlyph>,
) {
    transformed.clear();
    let cos_r = anchor.rotation.cos();
    let sin_r = anchor.rotation.sin();

    for g in glyphs {
        // Glyph local center offset relative to origin:
        let cx = g.x + g.width / 2.0;
        let cy = g.y + g.height / 2.0;

        // Rotate the offset:
        let rx = (cx * cos_r - cy * sin_r) * anchor.scale;
        let ry = (cx * sin_r + cy * cos_r) * anchor.scale;

        // Translate by anchor position:
        let px = anchor.position[0] + rx;
        let py = anchor.position[1] + ry;

        transformed.push(TransformedGlyph {
            char_code: g.char_code,
            position: [px, py],
            size: [g.width * anchor.scale, g.height * anchor.scale],
            uv: g.uv,
            scale: anchor.scale,
            rotation: anchor.rotation,
        });
    }
}
