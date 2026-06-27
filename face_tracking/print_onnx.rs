use ort::{Environment, SessionBuilder, GraphOptimizationLevel};
fn main() {
    let env = Environment::builder().build().unwrap();
    let session = SessionBuilder::new(&env).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .with_model_from_file("../ascii-realtime/addons/face_tracking/models/blazeface.onnx").unwrap();
    for input in session.inputs.iter() {
        println!("Input: {} {:?}", input.name, input.dimensions);
    }
    for output in session.outputs.iter() {
        println!("Output: {} {:?}", output.name, output.dimensions);
    }
}
