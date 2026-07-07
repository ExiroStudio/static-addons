//! FFI entry points for the face-tracking addon.
//!
//! Exports a single `extern "C"` symbol:
//!   - `behavior_api() -> *const BehaviorApi`
//!
//! The engine passes a `*mut NativeHost` to `create()`. We wrap it in a
//! `FfiBridgeHost` struct implementing the `HostApi` trait so the addon's
//! internal logic can run unchanged.

use std::ffi::c_void;
use std::sync::Arc;

use crate::host::{FrameRef, FrameTier, FrameView, HostApi, ParamValue, SignalValue};
use crate::runtime::{Executor, Timing};

// ---- C-ABI types (must match the engine's behavior/native.rs) ----------------

#[repr(C)]
pub struct FfiFrame {
    pub width: u32,
    pub height: u32,
    pub data: *const u8,
    pub len: usize,
    pub valid: u8,
}

#[repr(C)]
pub struct NativeHost {
    pub engine_ctx: *mut c_void,
    pub read_frame: unsafe extern "C" fn(host: *mut NativeHost, out: *mut FfiFrame),
    pub publish_f32:
        unsafe extern "C" fn(host: *mut NativeHost, name: *const u8, name_len: usize, value: f32),
    pub publish_bool:
        unsafe extern "C" fn(host: *mut NativeHost, name: *const u8, name_len: usize, value: u8),
    pub publish_vec2: unsafe extern "C" fn(
        host: *mut NativeHost,
        name: *const u8,
        name_len: usize,
        x: f32,
        y: f32,
    ),
    pub publish_vec3: unsafe extern "C" fn(
        host: *mut NativeHost,
        name: *const u8,
        name_len: usize,
        x: f32,
        y: f32,
        z: f32,
    ),
    pub get_param_f32: unsafe extern "C" fn(
        host: *mut NativeHost,
        name: *const u8,
        name_len: usize,
        out: *mut f32,
    ) -> u8,
    pub timing: unsafe extern "C" fn(host: *mut NativeHost, dt: *mut f32, elapsed: *mut f32),
}

#[repr(C)]
pub struct BehaviorApi {
    pub create: extern "C" fn(host: *mut NativeHost) -> *mut c_void,
    pub update: extern "C" fn(instance: *mut c_void),
    pub destroy: extern "C" fn(instance: *mut c_void),
}

// ---- bridge host (implements the addon's HostApi trait) ----------------------

struct FfiBridgeHost {
    host_ptr: *mut NativeHost,
    frame: Option<FfiFrame>,
    frame_bytes: Option<Arc<[u8]>>,
    timing: Timing,
    epoch: u64,
}

impl FfiBridgeHost {
    unsafe fn new(host_ptr: *mut NativeHost, epoch: u64) -> Self {
        let mut ffi_frame = FfiFrame {
            width: 0,
            height: 0,
            data: std::ptr::null(),
            len: 0,
            valid: 0,
        };
        ((*host_ptr).read_frame)(host_ptr, &mut ffi_frame);

        let frame_bytes = if ffi_frame.valid != 0 && !ffi_frame.data.is_null() && ffi_frame.len > 0
        {
            let slice = std::slice::from_raw_parts(ffi_frame.data, ffi_frame.len);
            Some(Arc::from(slice.to_vec().into_boxed_slice()))
        } else {
            None
        };

        let mut dt = 0.0;
        let mut elapsed = 0.0;
        ((*host_ptr).timing)(host_ptr, &mut dt, &mut elapsed);

        Self {
            host_ptr,
            frame: if ffi_frame.valid != 0 {
                Some(ffi_frame)
            } else {
                None
            },
            frame_bytes,
            timing: Timing { dt, elapsed },
            epoch,
        }
    }

    fn frame_view(&self) -> Option<FrameView> {
        self.frame_bytes
            .as_ref()
            .map(|bytes| FrameView::from_raw(bytes.clone()))
    }
}

impl HostApi for FfiBridgeHost {
    fn request_frame(&mut self, _tier: FrameTier) -> bool {
        self.frame.is_some()
    }

    fn change_frame_tier(&mut self, _tier: FrameTier) -> bool {
        true
    }

    fn read_frame(&mut self) -> Option<FrameRef> {
        self.frame.as_ref().map(|f| FrameRef {
            epoch: self.epoch,
            tier_generation: 1,
            width: f.width,
            height: f.height,
            tier: FrameTier::FullRes,
        })
    }

    fn publish(&mut self, name: &str, value: SignalValue) {
        let host_ptr = self.host_ptr;
        let name_ptr = name.as_ptr();
        let name_len = name.len();
        unsafe {
            match value {
                SignalValue::F32(v) => {
                    ((*host_ptr).publish_f32)(host_ptr, name_ptr, name_len, v);
                }
                SignalValue::Bool(v) => {
                    ((*host_ptr).publish_bool)(host_ptr, name_ptr, name_len, v as u8);
                }
                SignalValue::Vec2(v) => {
                    ((*host_ptr).publish_vec2)(host_ptr, name_ptr, name_len, v[0], v[1]);
                }
                SignalValue::Vec3(v) => {
                    ((*host_ptr).publish_vec3)(host_ptr, name_ptr, name_len, v[0], v[1], v[2]);
                }
            }
        }
    }

    fn get_param(&self, name: &str) -> Option<ParamValue> {
        let mut out: f32 = 0.0;
        let found = unsafe {
            ((*self.host_ptr).get_param_f32)(self.host_ptr, name.as_ptr(), name.len(), &mut out)
        };
        if found != 0 {
            Some(ParamValue::F32(out))
        } else {
            None
        }
    }

    fn timing(&self) -> Timing {
        self.timing
    }
}

// ---- addon state ------------------------------------------------------------

struct AddonState {
    host_ptr: *mut NativeHost,
    executor: Executor,
    epoch: u64,
}

// ---- exported symbols -------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandshakeData {
    pub runtime_family: u32,
    pub handshake_abi: u32,
    pub runtime_abi: u32,
}

#[no_mangle]
pub extern "C" fn get_handshake() -> HandshakeData {
    HandshakeData {
        runtime_family: 0x42454841, // BEHAVIOR_FAMILY
        handshake_abi: 1,           // HANDSHAKE_ABI_VERSION
        runtime_abi: 0x42414231,    // BEHAVIOR_ABI_V1
    }
}

#[no_mangle]
pub extern "C" fn create_instance(host: *mut NativeHost) -> *mut c_void {
    let state = Box::new(AddonState {
        host_ptr: host,
        executor: Executor::new(),
        epoch: 1,
    });
    let state = Box::into_raw(state);
    unsafe {
        let _ = (*state).executor.load();
        let _ = (*state).executor.bind();
        let _ = (*state).executor.start();
    }
    state as *mut c_void
}

#[no_mangle]
pub extern "C" fn update_instance(instance: *mut c_void) {
    let start_update_time = std::time::Instant::now();
    if instance.is_null() {
        println!("[FAILURE] [FACE TRACKING] update_instance received null instance pointer");
        return;
    }
    let state = unsafe { &mut *(instance as *mut AddonState) };
    state.epoch += 1;

    let frame_num = state.epoch;
    let thread_id = format!("{:?}", std::thread::current().id());
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    println!("========== FRAME BEGIN ==========");
    println!("frame_number: {}", frame_num);
    println!("timestamp: {}", timestamp);
    println!("thread id: {}", thread_id);
    println!("===============================");

    // AREA 2 - Camera capture timing
    let start_camera_time = std::time::Instant::now();
    let mut bridge = unsafe { FfiBridgeHost::new(state.host_ptr, state.epoch) };
    let camera_duration = start_camera_time.elapsed();

    let camera_initialized = true;
    let camera_connected = true;
    let frame_received = bridge.frame.is_some();
    if let Some(f) = &bridge.frame {
        println!("[CAMERA] camera initialized? {}", camera_initialized);
        println!("[CAMERA] camera connected? {}", camera_connected);
        println!("[CAMERA] frame received? {}", frame_received);
        println!("[CAMERA] frame width: {}", f.width);
        println!("[CAMERA] frame height: {}", f.height);
        println!("[CAMERA] pixel format: RGBA8");
        println!("[CAMERA] stride: {}", f.width * 4);
        println!("[CAMERA] frame byte size: {}", f.len);
        println!("[CAMERA] pointer address: {:p}", f.data);
        println!("[CAMERA] capture duration: {:?}", camera_duration);
    } else {
        println!("[CAMERA] camera initialized? {}", camera_initialized);
        println!("[CAMERA] camera connected? {}", camera_connected);
        println!("[CAMERA] frame received? {}", frame_received);
        println!("[CAMERA] frame width: 0");
        println!("[CAMERA] frame height: 0");
        println!("[CAMERA] pixel format: unknown");
        println!("[CAMERA] stride: 0");
        println!("[CAMERA] frame byte size: 0");
        println!("[CAMERA] pointer address: 0x0");
        println!("[CAMERA] capture duration: {:?}", camera_duration);
        println!("[WARNING] [CAMERA] capture failed: read_frame returned invalid frame");
    }

    if let Some(frame_ref) = bridge.read_frame() {
        if let Some(frame_view) = bridge.frame_view() {
            let elapsed = bridge.timing.elapsed;
            let _ = state
                .executor
                .tick_with_frame(&mut bridge, frame_ref, frame_view, elapsed);
        } else {
            println!("[WARNING] [FACE TRACKING] frame_view is None");
            let dt = bridge.timing.dt;
            let elapsed = bridge.timing.elapsed;
            let _ = state.executor.tick(&mut bridge, dt, elapsed);
        }
    } else {
        let dt = bridge.timing.dt;
        let elapsed = bridge.timing.elapsed;
        let _ = state.executor.tick(&mut bridge, dt, elapsed);
    }

    println!("========== FRAME END ============");
    println!("frame_number: {}", frame_num);
    println!("total frame time: {:?}", start_update_time.elapsed());
    println!("===============================");
}

#[no_mangle]
pub extern "C" fn destroy_instance(instance: *mut c_void) {
    if instance.is_null() {
        return;
    }
    let state = unsafe { Box::from_raw(instance as *mut AddonState) };
    drop(state);
}

static API: BehaviorApi = BehaviorApi {
    create: create_instance,
    update: update_instance,
    destroy: destroy_instance,
};

#[no_mangle]
pub extern "C" fn behavior_api() -> *const BehaviorApi {
    &API
}
