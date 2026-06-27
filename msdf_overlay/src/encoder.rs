use crate::transform::TransformedGlyph;

pub fn encode(glyphs: &[TransformedGlyph], color: [f32; 4], out: &mut Vec<f32>) {
    out.clear();
    out.reserve(glyphs.len() * 12);
    for g in glyphs {
        out.push(g.position[0]);
        out.push(g.position[1]);
        out.push(g.uv[0]);
        out.push(g.uv[1]);
        out.push(g.uv[2]);
        out.push(g.uv[3]);
        out.push(color[0]);
        out.push(color[1]);
        out.push(color[2]);
        out.push(color[3]);
        out.push(g.scale);
        out.push(g.rotation);
    }
}
