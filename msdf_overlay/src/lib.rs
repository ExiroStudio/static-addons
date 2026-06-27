use std::ffi::c_void;

pub mod atlas;
pub mod parser;
pub mod layout;
pub mod transform;
pub mod encoder;

#[repr(C)]
pub struct HandshakeData {
    pub runtime_family: u32,
    pub handshake_abi: u32,
    pub runtime_abi: u32,
}

pub const RENDER_FAMILY: u32 = 0x52454e44; // "REND"
pub const HANDSHAKE_ABI_VERSION: u32 = 1;
pub const RENDER_ABI_V1: u32 = 0x52414231; // "RAB1"

#[no_mangle]
pub extern "C" fn get_handshake() -> HandshakeData {
    HandshakeData {
        runtime_family: RENDER_FAMILY,
        handshake_abi: HANDSHAKE_ABI_VERSION,
        runtime_abi: RENDER_ABI_V1,
    }
}

#[repr(C)]
pub struct PipelineHost {
    pub engine_ctx: *mut c_void,
    pub publish_instances: unsafe extern "C" fn(
        host: *mut PipelineHost,
        instance_id: *const u8,
        instance_id_len: usize,
        schema_id: u64,
        rows_data: *const f32,
        rows_data_len: usize,
    ),
    pub get_param_f32: unsafe extern "C" fn(
        host: *mut PipelineHost,
        name: *const u8,
        name_len: usize,
        out: *mut f32,
    ) -> u8,
    pub timing: unsafe extern "C" fn(
        host: *mut PipelineHost,
        dt: *mut f32,
        elapsed: *mut f32,
    ),
}

use std::time::{Instant, SystemTime};

pub struct AddonContext {
    pub host: *mut PipelineHost,
    pub cached_expression: String,
    pub last_metadata_check: Instant,
    pub last_modified: Option<SystemTime>,
    // Reusable buffers to avoid allocations on the hot path:
    pub chars: Vec<char>,
    pub glyphs: Vec<layout::LayoutGlyph>,
    pub transformed: Vec<transform::TransformedGlyph>,
    pub encoded: Vec<f32>,
}

fn read_expression_from_file() -> Option<String> {
    if let Ok(content) = std::fs::read_to_string("pipeline.json") {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(pipeline) = json.get("pipeline").and_then(|p| p.as_array()) {
                for node in pipeline {
                    if node.get("addon").and_then(|a| a.as_str()) == Some("msdf-overlay") {
                        if let Some(expr) = node.get("config").and_then(|c| c.get("ascii_expression")).and_then(|e| e.as_str()) {
                            return Some(expr.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

#[no_mangle]
pub extern "C" fn create_instance(host: *mut PipelineHost) -> *mut c_void {
    let initial_expr = read_expression_from_file().unwrap_or_else(|| ">_<".to_string());
    let last_modified = std::fs::metadata("pipeline.json")
        .ok()
        .and_then(|m| m.modified().ok());
    let ctx = Box::new(AddonContext {
        host,
        cached_expression: initial_expr,
        last_metadata_check: Instant::now(),
        last_modified,
        chars: Vec::with_capacity(32),
        glyphs: Vec::with_capacity(32),
        transformed: Vec::with_capacity(32),
        encoded: Vec::with_capacity(384),
    });
    Box::into_raw(ctx) as *mut c_void
}

fn get_param(host: *mut PipelineHost, name: &str) -> f32 {
    let mut val = 0.0;
    unsafe {
        let name_bytes = name.as_bytes();
        let ok = ((*host).get_param_f32)(host, name_bytes.as_ptr(), name_bytes.len(), &mut val);
        if ok != 0 {
            val
        } else {
            0.0
        }
    }
}

fn update_cached_expression(ctx: &mut AddonContext) {
    let now = Instant::now();
    // Only check file metadata at most once every 500ms
    if now.duration_since(ctx.last_metadata_check).as_millis() > 500 {
        ctx.last_metadata_check = now;
        if let Ok(metadata) = std::fs::metadata("pipeline.json") {
            if let Ok(modified) = metadata.modified() {
                if Some(modified) != ctx.last_modified {
                    ctx.last_modified = Some(modified);
                    if let Some(new_expr) = read_expression_from_file() {
                        ctx.cached_expression = new_expr;
                    }
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn update_instance(instance: *mut c_void) {
    if instance.is_null() {
        return;
    }
    let ctx = unsafe { &mut *(instance as *mut AddonContext) };

    // Get parameters
    let scale_mul = get_param(ctx.host, "scale_mul");
    let opacity = get_param(ctx.host, "opacity");

    // Re-check and update ascii expression cached in AddonContext if changed
    update_cached_expression(ctx);

    // Reuse vectors to completely eliminate heap allocation on the hot path
    parser::parse(&ctx.cached_expression, &mut ctx.chars);
    layout::compute_layout(&ctx.chars, 0.0, &mut ctx.glyphs);

    let anchor = layout::Anchor {
        position: [0.0, 0.0],
        rotation: 0.0,
        scale: scale_mul * 0.15,
    };
    transform::apply_transform(&ctx.glyphs, &anchor, &mut ctx.transformed);
    encoder::encode(&ctx.transformed, [1.0, 1.0, 1.0, opacity], &mut ctx.encoded);

    // Publish
    let instance_id = b"msdf-overlay";
    unsafe {
        ((*ctx.host).publish_instances)(
            ctx.host,
            instance_id.as_ptr(),
            instance_id.len(),
            1, // schema_id
            ctx.encoded.as_ptr(),
            ctx.encoded.len(),
        );
    }
}

#[no_mangle]
pub extern "C" fn destroy_instance(instance: *mut c_void) {
    if !instance.is_null() {
        unsafe {
            let _ = Box::from_raw(instance as *mut AddonContext);
        }
    }
}
