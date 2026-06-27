use crate::atlas;

#[derive(Debug, Clone)]
pub struct LayoutGlyph {
    pub char_code: char,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub uv: [f32; 4],
}

#[derive(Debug, Clone)]
pub struct Anchor {
    pub position: [f32; 2],
    pub rotation: f32,
    pub scale: f32,
}

pub fn compute_layout(chars: &[char], tracking: f32, glyphs: &mut Vec<LayoutGlyph>) {
    glyphs.clear();
    let mut current_x = 0.0;

    for &c in chars {
        if let Some(info) = atlas::lookup(c) {
            glyphs.push(LayoutGlyph {
                char_code: c,
                x: current_x + info.bearing_x,
                // Center typographic baseline around y = 0
                y: -info.bearing_y / 2.0,
                width: info.width,
                height: info.height,
                uv: info.uv,
            });
            current_x += info.advance + tracking;
        }
    }

    // Center-align the glyphs around origin (0, 0)
    if !glyphs.is_empty() {
        let min_x = glyphs.first().unwrap().x;
        let max_x = glyphs.last().unwrap().x + glyphs.last().unwrap().width;
        let width = max_x - min_x;
        let offset_x = -width / 2.0;

        for g in glyphs.iter_mut() {
            g.x += offset_x;
        }
    }
}
