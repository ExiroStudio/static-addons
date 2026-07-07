struct Params {
    cell_size: f32,
    dot_radius: f32,
    gamma: f32,
    _pad: f32,
};
@group(2) @binding(0) var<uniform> params: Params;

// Procedural hash function (2D to 1D)
// Returns a deterministic pseudorandom value in [0.0, 1.0]
fn hash21(p: vec2<f32>) -> f32 {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)),
                      dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q.x + q.y) * 43758.5453123);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Convert uv [0..1] to pixel coordinates
    let pixel_coord = in.uv * H.resolution;
    
    // Determine the cell index for the current pixel
    let cell_coord = floor(pixel_coord / params.cell_size);
    
    // Find the UV coordinate for the exact center of this cell
    let cell_center_uv = (cell_coord + vec2<f32>(0.5, 0.5)) * params.cell_size / H.resolution;
    
    // Sample the brightness (luma) of the image strictly at the cell center.
    // Every pixel within this cell will sample the exact same point, 
    // maximizing hardware texture cache hits and keeping LOD derivatives stable.
    let luma = sample_luma(cell_center_uv);
    
    // Apply gamma curve to convert linear brightness into a perceived density.
    // This allows highlights to look organic without immediately clamping to a solid block.
    let density = pow(luma, params.gamma);
    
    // Generate the procedural spatial energy distribution for this cell
    let noise = hash21(cell_coord);
    
    // Occupancy test: a dot is only drawn if the cell's density overcomes its noise threshold
    let is_active = density > noise;
    
    // Hardcoded colors for V1 (Engine schema doesn't yet support passing Color directly to f32 uniform array)
    let foreground = vec3<f32>(1.0, 1.0, 1.0);
    let background = vec3<f32>(0.0, 0.0, 0.0);
    
    if (is_active) {
        // Calculate the physical center of the cell in pixel space
        let cell_center_pixel = (cell_coord + vec2<f32>(0.5, 0.5)) * params.cell_size;
        
        // Distance from current fragment to the cell center
        let dist = distance(pixel_coord, cell_center_pixel);
        
        // Maximum radius of the dot in pixels
        let max_radius = params.cell_size * params.dot_radius;
        
        // Draw the dot using a simple Signed Distance Field with 1-pixel anti-aliasing edge
        let alpha = 1.0 - smoothstep(max_radius - 0.75, max_radius + 0.75, dist);
        
        let color = mix(background, foreground, alpha);
        return vec4<f32>(color, 1.0);
    } else {
        // Inactive cell
        return vec4<f32>(background, 1.0);
    }
}
