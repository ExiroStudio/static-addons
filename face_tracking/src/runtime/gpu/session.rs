use ndarray::Array4;
use ort::session::Session;
use std::path::Path;

pub struct GpuSession {
    session: Session,
}

impl GpuSession {
    pub fn new(model_path: &Path) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("{:?}", e))?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])
            .map_err(|e| format!("{:?}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("{:?}", e))?;
        for input in session.inputs() {
            eprintln!("[runtime] model loaded: input {}", input.name());
        }
        for output in session.outputs() {
            eprintln!("[runtime] model loaded: output {}", output.name());
        }

        Ok(Self { session })
    }

    pub fn run(&mut self, input: Array4<f32>) -> Result<ort::session::SessionOutputs<'_>, String> {
        let input_shape = input
            .shape()
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>();
        let input_data = input.into_raw_vec();

        let input_value = ort::value::Value::from_array((input_shape, input_data))
            .map_err(|e| format!("{:?}", e))?;

        self.session
            .run(ort::inputs![
                "input.1" => input_value,
            ])
            .map_err(|e| format!("{:?}", e))
    }
}
