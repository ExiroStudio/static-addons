use face_tracking_addon::bootstrap::Bootstrap;
use std::io;

fn main() -> io::Result<()> {
    let mut app = Bootstrap::new();
    app.run()
}
