use ort::session::SessionOutputs;

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedFace {
    pub score: f32,
    pub bbox: [f32; 4], // [x1, y1, x2, y2]
    pub kps: [[f32; 2]; 5],
    pub capture_ts: u64,
    pub infer_ts: u64,
    pub frame_id: u64,
    pub is_predicted: bool,
}

#[derive(Clone, Copy)]
struct Candidate {
    score: f32,
    stride: usize,
    idx: usize,
    cx: f32,
    cy: f32,
    bbox_raw: [f32; 4],
}

pub fn postprocess(
    outputs: SessionOutputs,
    threshold: f32,
    scale: f32,
    offset_x: f32,
    offset_y: f32,
    orig_width: f32,
    orig_height: f32,
) -> Result<Option<DecodedFace>, String> {
    // 1. Candidate Filter
    let mut candidates = [Candidate {
        score: 0.0,
        stride: 0,
        idx: 0,
        cx: 0.0,
        cy: 0.0,
        bbox_raw: [0.0; 4],
    }; 1050];
    let mut num_candidates = 0;

    let strides = [8, 16, 32];
    for &stride in &strides {
        let score_name = format!("score_{}", stride);
        let bbox_name = format!("bbox_{}", stride);

        let score_val = outputs.get(&score_name).ok_or("missing score")?;
        let bbox_val = outputs.get(&bbox_name).ok_or("missing bbox")?;

        let (score_shape, score_data) = score_val
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let (_, bbox_data) = bbox_val
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;

        let num_anchors = score_shape[1] as usize;
        let grid_w = 224 / stride;

        for (idx, &score) in score_data.iter().enumerate().take(num_anchors) {
            if score >= threshold {
                let cell_idx = idx / 2;
                let grid_y = cell_idx / grid_w;
                let grid_x = cell_idx % grid_w;

                let cx = grid_x as f32 * stride as f32;
                let cy = grid_y as f32 * stride as f32;

                let b_idx = idx * 4;
                let bbox_raw = [
                    cx - bbox_data[b_idx] * stride as f32,
                    cy - bbox_data[b_idx + 1] * stride as f32,
                    cx + bbox_data[b_idx + 2] * stride as f32,
                    cy + bbox_data[b_idx + 3] * stride as f32,
                ];

                if num_candidates < 1050 {
                    candidates[num_candidates] = Candidate {
                        score,
                        stride,
                        idx,
                        cx,
                        cy,
                        bbox_raw,
                    };
                    num_candidates += 1;
                }
            }
        }
    }

    if num_candidates == 0 {
        return Ok(None);
    }

    // 2. Top-K Limit
    let valid_cands = &mut candidates[..num_candidates];
    valid_cands.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_k = valid_cands.len().min(16);
    let top_cands = &valid_cands[..top_k];

    // 3. Minimal NMS
    let mut suppressed = [false; 16];
    let mut best_cand: Option<&Candidate> = None;

    for i in 0..top_k {
        if suppressed[i] {
            continue;
        }

        // Single-face pipeline: take the first unsuppressed face
        if best_cand.is_none() {
            best_cand = Some(&top_cands[i]);
        }

        let a = &top_cands[i];
        let area_a =
            (a.bbox_raw[2] - a.bbox_raw[0]).max(0.0) * (a.bbox_raw[3] - a.bbox_raw[1]).max(0.0);

        for j in (i + 1)..top_k {
            if suppressed[j] {
                continue;
            }
            let b = &top_cands[j];
            let xx1 = a.bbox_raw[0].max(b.bbox_raw[0]);
            let yy1 = a.bbox_raw[1].max(b.bbox_raw[1]);
            let xx2 = a.bbox_raw[2].min(b.bbox_raw[2]);
            let yy2 = a.bbox_raw[3].min(b.bbox_raw[3]);
            let w = (xx2 - xx1).max(0.0);
            let h = (yy2 - yy1).max(0.0);
            let inter = w * h;
            let area_b =
                (b.bbox_raw[2] - b.bbox_raw[0]).max(0.0) * (b.bbox_raw[3] - b.bbox_raw[1]).max(0.0);
            let iou = inter / (area_a + area_b - inter);

            if iou > 0.35 {
                suppressed[j] = true;
            }
        }
    }

    let best = match best_cand {
        Some(b) => b,
        None => return Ok(None),
    };

    // 4. Decode ONLY selected candidates
    let kps_name = format!("kps_{}", best.stride);
    let kps_val = outputs.get(&kps_name).ok_or("missing kps")?;
    let (_, kps_data) = kps_val
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;

    let mut kps = [[0.0; 2]; 5];
    let k_idx = best.idx * 10;
    for k in 0..5 {
        let kx = best.cx + kps_data[k_idx + k * 2] * best.stride as f32;
        let ky = best.cy + kps_data[k_idx + k * 2 + 1] * best.stride as f32;
        kps[k] = [
            ((kx - offset_x) / scale) / orig_width,
            ((ky - offset_y) / scale) / orig_height,
        ];
    }

    let x1 = ((best.bbox_raw[0] - offset_x) / scale) / orig_width;
    let y1 = ((best.bbox_raw[1] - offset_y) / scale) / orig_height;
    let x2 = ((best.bbox_raw[2] - offset_x) / scale) / orig_width;
    let y2 = ((best.bbox_raw[3] - offset_y) / scale) / orig_height;

    // Metrics logs
    eprintln!("raw_candidates={} filtered_candidates={} nms_out=1", num_candidates, top_k);

    Ok(Some(DecodedFace {
        score: best.score,
        bbox: [x1, y1, x2, y2],
        kps,
        capture_ts: 0,
        infer_ts: 0,
        frame_id: 0,
        is_predicted: false,
    }))
}
