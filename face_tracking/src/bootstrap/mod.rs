//! src/bootstrap/mod.rs
//! Lifecycle + Protocol handling for Native Runner.

use crate::host::HostApi;
use crate::runtime::Executor;
use std::io::{self, Read, Write};

pub struct Bootstrap {
    executor: Executor,
    host: crate::host::MockHost,
}

impl Default for Bootstrap {
    fn default() -> Self {
        Self::new()
    }
}

impl Bootstrap {
    pub fn new() -> Self {
        Self {
            executor: Executor::new(),
            host: crate::host::MockHost::new(),
        }
    }

    pub fn run(&mut self) -> io::Result<()> {
        let mut stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut tag = [0u8; 1];
        let mut timing_buf = [0u8; 8];

        loop {
            if stdin.read_exact(&mut tag).is_err() {
                break; // Host closed channel
            }

            match tag[0] {
                0 => {
                    // Load
                    self.executor.load()?;
                    stdout.write_all(&[0])?;
                }
                1 => {
                    // Bind
                    self.executor.bind()?;
                    stdout.write_all(&[1])?;
                }
                2 => {
                    // Start
                    self.executor.start()?;
                    stdout.write_all(&[2])?;
                }
                3 => {
                    // Tick
                    stdin.read_exact(&mut timing_buf)?;
                    let dt = f32::from_le_bytes(timing_buf[0..4].try_into().unwrap());
                    let elapsed = f32::from_le_bytes(timing_buf[4..8].try_into().unwrap());

                    self.host.begin_tick();

                    let outcome = if let Some(frame_ref) = self.host.read_frame() {
                        let frame_view = self.host.read_frame_view().unwrap();
                        self.executor.tick_with_frame(
                            &mut self.host,
                            frame_ref,
                            frame_view,
                            elapsed,
                        )
                    } else {
                        self.executor.tick(&mut self.host, dt, elapsed)
                    };

                    match outcome {
                        Ok(_) => stdout.write_all(&[3, 0])?, // Ticked(Ok)
                        Err(e) => {
                            let msg = e.to_string();
                            let bytes = msg.as_bytes();
                            stdout.write_all(&[3, 2])?; // Ticked(Faulted)
                            stdout.write_all(&(bytes.len() as u32).to_le_bytes())?;
                            stdout.write_all(bytes)?;
                        }
                    }
                }
                4 => {
                    // Stop
                    self.executor.stop()?;
                    stdout.write_all(&[4])?;
                }
                5 => {
                    // Unload
                    self.executor.unload()?;
                    stdout.write_all(&[5])?;
                    break;
                }
                _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid tag")),
            }
            stdout.flush()?;
        }
        Ok(())
    }
}
