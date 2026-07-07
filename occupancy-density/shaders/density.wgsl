struct Params {
    cell_size: f32,
    dot_radius: f32,
    gamma: f32,
    highlight: f32,
    highlight_ceiling: f32,
    shadow: f32,
    shadow_floor: f32,
    _pad: f32,
};
@group(2) @binding(0) var<uniform> params: Params;

// Interleaved Gradient Noise (IGN)
// Provides a pseudo-blue noise high-frequency scatter.
fn ign(cell: vec2<f32>) -> f32 {
    let f = dot(cell, vec2<f32>(0.06711056, 0.00583715));
    return fract(52.9829189 * fract(f));
}

// Spatial Weyl sequence based on Martin Roberts' R2
// Provides highly uniform low-discrepancy distribution.
fn weyl(cell: vec2<f32>) -> f32 {
    return fract(dot(cell, vec2<f32>(0.7548776662466927, 0.5698402909980532)));
}

// V1.1: Blue Noise Inspired Distribution (IGN + R2 Hybrid)
fn get_noise(cell: vec2<f32>) -> f32 {
    return fract(ign(cell) + weyl(cell));
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
    
    // 1. Mid-tone control (gamma curve)
    let percept = pow(luma, params.gamma);
    
    // 2. Soften the transitions (S-Curve for organic gradients)
    let shaped = smoothstep(params.shadow, params.highlight, percept);
    
    // 3. Compress the occupancy range
    // Darkest areas maintain shadow_floor activity (alive, not dead)
    // Brightest areas cap at highlight_ceiling activity (dense, but with breathing holes)
    let density = mix(params.shadow_floor, params.highlight_ceiling, shaped);
    
    // Generate the procedural spatial energy distribution for this cell
    // Using V1.1 Hybrid: R2 (uniformity) + IGN (organic scatter)
    let noise = get_noise(cell_coord);
    
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
