use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Kinds of predictions the model can produce.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PredictionType {
    ValuePrediction,
    Classification,
    AnomalyScore,
    Action,
    Trend,
    MultiTarget(Vec<f64>),
}

/// A single prediction produced by a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionOutput {
    pub prediction_type: PredictionType,
    pub value: f64,
    pub confidence: f64,
    pub model_name: String,
    pub latency_ms: f64,
}

/// Encoding method for turning prediction outputs into vectors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PredEncodingMethod {
    Raw,
    Confidence,
    Hierarchical,
    MultiHead { heads: usize },
}

/// Encoder that transforms `PredictionOutput`s into `PredictionVector`s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionEncoder {
    pub input_dim: usize,
    pub output_dim: usize,
    pub method: PredEncodingMethod,
}

/// A prediction encoded as a vector, ready for storage in the prediction DB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionVector {
    pub id: Uuid,
    pub vector: Vec<f64>,
    pub output: PredictionOutput,
    pub room_id: String,
    pub timestamp: u64,
}

/// A batch of encoded prediction vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionBatch {
    pub vectors: Vec<PredictionVector>,
    pub encoding_method: PredEncodingMethod,
}

/// Aggregate statistics for a prediction batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredBatchStats {
    pub count: usize,
    pub mean_confidence: f64,
    pub min_confidence: f64,
    pub max_confidence: f64,
    pub mean_value: f64,
    pub mean_latency_ms: f64,
}

// ---------------------------------------------------------------------------
// PredictionOutput
// ---------------------------------------------------------------------------

impl PredictionOutput {
    pub fn new(pred_type: PredictionType, value: f64, confidence: f64) -> Self {
        Self {
            prediction_type: pred_type,
            value,
            confidence,
            model_name: String::new(),
            latency_ms: 0.0,
        }
    }

    /// Encode this prediction as an f64 vector.
    pub fn to_vector(&self) -> Vec<f64> {
        let mut v = vec![self.value, self.confidence];
        match &self.prediction_type {
            PredictionType::ValuePrediction => v.push(1.0),
            PredictionType::Classification => v.push(2.0),
            PredictionType::AnomalyScore => v.push(3.0),
            PredictionType::Action => v.push(4.0),
            PredictionType::Trend => v.push(5.0),
            PredictionType::MultiTarget(targets) => {
                v.push(6.0);
                v.extend(targets);
            }
        }
        v
    }
}

// ---------------------------------------------------------------------------
// PredictionEncoder
// ---------------------------------------------------------------------------

impl PredictionEncoder {
    pub fn new(input_dim: usize, output_dim: usize, method: PredEncodingMethod) -> Self {
        Self {
            input_dim,
            output_dim,
            method,
        }
    }

    /// Encode a single prediction output into a prediction vector.
    pub fn encode(&self, output: &PredictionOutput) -> PredictionVector {
        let raw = output.to_vector();
        let vector = match &self.method {
            PredEncodingMethod::Raw => raw,
            PredEncodingMethod::Confidence => {
                let mut v = raw;
                // Weight every element by confidence.
                v.iter_mut().for_each(|x| *x *= output.confidence);
                v
            }
            PredEncodingMethod::Hierarchical => {
                let mut v = raw;
                // Append hierarchical summary features.
                let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
                let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
                v.push(mean);
                v.push(max);
                v.push(min);
                v
            }
            PredEncodingMethod::MultiHead { heads } => {
                let mut v = Vec::with_capacity(raw.len() * heads);
                for _ in 0..*heads {
                    v.extend(&raw);
                }
                v
            }
        };
        // Pad or truncate to output_dim.
        let mut final_vec = vector;
        final_vec.resize(self.output_dim, 0.0);

        PredictionVector {
            id: Uuid::new_v4(),
            vector: final_vec,
            output: output.clone(),
            room_id: String::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Encode a slice of prediction outputs into a batch.
    pub fn encode_batch(&self, outputs: &[PredictionOutput]) -> PredictionBatch {
        let vectors = outputs.iter().map(|o| self.encode(o)).collect();
        PredictionBatch {
            vectors,
            encoding_method: self.method.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PredictionVector
// ---------------------------------------------------------------------------

impl PredictionVector {
    pub fn cosine_similarity(a: &Self, b: &Self) -> f64 {
        let min_len = a.vector.len().min(b.vector.len());
        if min_len == 0 {
            return 0.0;
        }
        let dot: f64 = a.vector[..min_len]
            .iter()
            .zip(&b.vector[..min_len])
            .map(|(x, y)| x * y)
            .sum();
        let mag_a: f64 = a.vector[..min_len].iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = b.vector[..min_len].iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    pub fn euclidean(a: &Self, b: &Self) -> f64 {
        let min_len = a.vector.len().min(b.vector.len());
        let sum: f64 = a.vector[..min_len]
            .iter()
            .zip(&b.vector[..min_len])
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        sum.sqrt()
    }
}

// ---------------------------------------------------------------------------
// PredictionBatch
// ---------------------------------------------------------------------------

impl PredictionBatch {
    /// Return the k nearest vectors to `query` by Euclidean distance.
    pub fn nearest(&self, query: &PredictionVector, k: usize) -> Vec<&PredictionVector> {
        let mut scored: Vec<(f64, &PredictionVector)> = self
            .vectors
            .iter()
            .map(|v| (PredictionVector::euclidean(v, query), v))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, v)| v).collect()
    }

    /// Filter vectors by minimum confidence.
    pub fn filter_by_confidence(&self, min_confidence: f64) -> Vec<&PredictionVector> {
        self.vectors
            .iter()
            .filter(|v| v.output.confidence >= min_confidence)
            .collect()
    }

    /// Compute aggregate statistics for the batch.
    pub fn stats(&self) -> PredBatchStats {
        let count = self.vectors.len();
        if count == 0 {
            return PredBatchStats {
                count: 0,
                mean_confidence: 0.0,
                min_confidence: 0.0,
                max_confidence: 0.0,
                mean_value: 0.0,
                mean_latency_ms: 0.0,
            };
        }
        let confidences: Vec<f64> = self.vectors.iter().map(|v| v.output.confidence).collect();
        let values: Vec<f64> = self.vectors.iter().map(|v| v.output.value).collect();
        let latencies: Vec<f64> = self.vectors.iter().map(|v| v.output.latency_ms).collect();

        PredBatchStats {
            count,
            mean_confidence: confidences.iter().sum::<f64>() / count as f64,
            min_confidence: confidences.iter().cloned().fold(f64::INFINITY, f64::min),
            max_confidence: confidences.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            mean_value: values.iter().sum::<f64>() / count as f64,
            mean_latency_ms: latencies.iter().sum::<f64>() / count as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PredictionOutput ---

    #[test]
    fn test_prediction_output_new() {
        let po = PredictionOutput::new(PredictionType::ValuePrediction, 42.0, 0.9);
        assert_eq!(po.prediction_type, PredictionType::ValuePrediction);
        assert!((po.value - 42.0).abs() < f64::EPSILON);
        assert!((po.confidence - 0.9).abs() < f64::EPSILON);
        assert!(po.model_name.is_empty());
        assert!((po.latency_ms).abs() < f64::EPSILON);
    }

    #[test]
    fn test_to_vector_value_prediction() {
        let po = PredictionOutput::new(PredictionType::ValuePrediction, 1.0, 0.5);
        let v = po.to_vector();
        assert_eq!(v, vec![1.0, 0.5, 1.0]);
    }

    #[test]
    fn test_to_vector_classification() {
        let po = PredictionOutput::new(PredictionType::Classification, 3.0, 0.8);
        assert_eq!(po.to_vector(), vec![3.0, 0.8, 2.0]);
    }

    #[test]
    fn test_to_vector_anomaly() {
        let po = PredictionOutput::new(PredictionType::AnomalyScore, 0.1, 0.95);
        assert_eq!(po.to_vector(), vec![0.1, 0.95, 3.0]);
    }

    #[test]
    fn test_to_vector_action() {
        let po = PredictionOutput::new(PredictionType::Action, 5.0, 0.7);
        assert_eq!(po.to_vector(), vec![5.0, 0.7, 4.0]);
    }

    #[test]
    fn test_to_vector_trend() {
        let po = PredictionOutput::new(PredictionType::Trend, -1.0, 0.6);
        assert_eq!(po.to_vector(), vec![-1.0, 0.6, 5.0]);
    }

    #[test]
    fn test_to_vector_multi_target() {
        let targets = vec![1.0, 2.0, 3.0];
        let po = PredictionOutput::new(PredictionType::MultiTarget(targets.clone()), 0.0, 1.0);
        let v = po.to_vector();
        assert_eq!(v, vec![0.0, 1.0, 6.0, 1.0, 2.0, 3.0]);
    }

    // --- Encoding methods ---

    #[test]
    fn test_encode_raw() {
        let enc = PredictionEncoder::new(4, 4, PredEncodingMethod::Raw);
        let po = PredictionOutput::new(PredictionType::ValuePrediction, 1.0, 0.5);
        let pv = enc.encode(&po);
        assert_eq!(pv.vector.len(), 4);
        assert_eq!(pv.vector[..3], [1.0, 0.5, 1.0]);
        assert!((pv.vector[3]).abs() < f64::EPSILON); // padded
    }

    #[test]
    fn test_encode_confidence() {
        let enc = PredictionEncoder::new(4, 4, PredEncodingMethod::Confidence);
        let po = PredictionOutput::new(PredictionType::ValuePrediction, 1.0, 0.5);
        let pv = enc.encode(&po);
        assert!((pv.vector[0] - 0.5).abs() < 1e-10);
        assert!((pv.vector[1] - 0.25).abs() < 1e-10);
        assert!((pv.vector[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_encode_hierarchical() {
        let enc = PredictionEncoder::new(8, 8, PredEncodingMethod::Hierarchical);
        let po = PredictionOutput::new(PredictionType::ValuePrediction, 1.0, 0.5);
        let pv = enc.encode(&po);
        // Raw: [1.0, 0.5, 1.0] + 3 summary = 6 elements, padded to 8
        assert_eq!(pv.vector.len(), 8);
        // summary features should be non-zero
        assert!(pv.vector[3].abs() > 0.0 || pv.vector[4].abs() > 0.0);
    }

    #[test]
    fn test_encode_multihead() {
        let enc = PredictionEncoder::new(6, 6, PredEncodingMethod::MultiHead { heads: 2 });
        let po = PredictionOutput::new(PredictionType::ValuePrediction, 1.0, 0.5);
        let pv = enc.encode(&po);
        // Raw: 3 elements * 2 heads = 6
        assert_eq!(pv.vector.len(), 6);
        assert!((pv.vector[0] - pv.vector[3]).abs() < f64::EPSILON);
    }

    // --- Batch ---

    #[test]
    fn test_encode_batch() {
        let enc = PredictionEncoder::new(4, 4, PredEncodingMethod::Raw);
        let outputs = vec![
            PredictionOutput::new(PredictionType::ValuePrediction, 1.0, 0.9),
            PredictionOutput::new(PredictionType::Classification, 0.0, 0.5),
        ];
        let batch = enc.encode_batch(&outputs);
        assert_eq!(batch.vectors.len(), 2);
        assert_eq!(batch.encoding_method, PredEncodingMethod::Raw);
    }

    // --- Similarity / Distance ---

    #[test]
    fn test_cosine_similarity_identical() {
        let enc = PredictionEncoder::new(4, 4, PredEncodingMethod::Raw);
        let po = PredictionOutput::new(PredictionType::ValuePrediction, 1.0, 0.5);
        let a = enc.encode(&po);
        let b = enc.encode(&po);
        let sim = PredictionVector::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = PredictionVector {
            id: Uuid::new_v4(),
            vector: vec![1.0, 0.0],
            output: PredictionOutput::new(PredictionType::ValuePrediction, 1.0, 0.5),
            room_id: String::new(),
            timestamp: 0,
        };
        let b = PredictionVector {
            id: Uuid::new_v4(),
            vector: vec![0.0, 1.0],
            output: PredictionOutput::new(PredictionType::ValuePrediction, 1.0, 0.5),
            room_id: String::new(),
            timestamp: 0,
        };
        let sim = PredictionVector::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = PredictionVector {
            id: Uuid::new_v4(),
            vector: vec![0.0, 0.0],
            output: PredictionOutput::new(PredictionType::ValuePrediction, 0.0, 0.0),
            room_id: String::new(),
            timestamp: 0,
        };
        let b = PredictionVector {
            id: Uuid::new_v4(),
            vector: vec![3.0, 4.0],
            output: PredictionOutput::new(PredictionType::ValuePrediction, 3.0, 4.0),
            room_id: String::new(),
            timestamp: 0,
        };
        let dist = PredictionVector::euclidean(&a, &b);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    // --- Nearest / Filter ---

    #[test]
    fn test_nearest() {
        let enc = PredictionEncoder::new(4, 4, PredEncodingMethod::Raw);
        let outputs: Vec<PredictionOutput> = (0..5)
            .map(|i| PredictionOutput::new(PredictionType::ValuePrediction, i as f64, 0.5))
            .collect();
        let batch = enc.encode_batch(&outputs);
        let query = enc.encode(&PredictionOutput::new(PredictionType::ValuePrediction, 1.0, 0.5));
        let nearest = batch.nearest(&query, 2);
        assert_eq!(nearest.len(), 2);
    }

    #[test]
    fn test_filter_by_confidence() {
        let enc = PredictionEncoder::new(4, 4, PredEncodingMethod::Raw);
        let outputs = vec![
            PredictionOutput::new(PredictionType::ValuePrediction, 1.0, 0.9),
            PredictionOutput::new(PredictionType::ValuePrediction, 2.0, 0.3),
            PredictionOutput::new(PredictionType::ValuePrediction, 3.0, 0.7),
        ];
        let batch = enc.encode_batch(&outputs);
        let filtered = batch.filter_by_confidence(0.7);
        assert_eq!(filtered.len(), 2);
    }

    // --- Stats ---

    #[test]
    fn test_batch_stats() {
        let enc = PredictionEncoder::new(4, 4, PredEncodingMethod::Raw);
        let mut outputs = vec![
            PredictionOutput::new(PredictionType::ValuePrediction, 1.0, 0.8),
            PredictionOutput::new(PredictionType::ValuePrediction, 2.0, 0.6),
        ];
        outputs[0].latency_ms = 10.0;
        outputs[1].latency_ms = 20.0;
        let batch = enc.encode_batch(&outputs);
        let stats = batch.stats();
        assert_eq!(stats.count, 2);
        assert!((stats.mean_confidence - 0.7).abs() < 1e-10);
        assert!((stats.min_confidence - 0.6).abs() < 1e-10);
        assert!((stats.max_confidence - 0.8).abs() < 1e-10);
        assert!((stats.mean_value - 1.5).abs() < 1e-10);
        assert!((stats.mean_latency_ms - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_batch_stats() {
        let batch = PredictionBatch {
            vectors: vec![],
            encoding_method: PredEncodingMethod::Raw,
        };
        let stats = batch.stats();
        assert_eq!(stats.count, 0);
        assert!((stats.mean_confidence).abs() < f64::EPSILON);
    }

    // --- Edge cases ---

    #[test]
    fn test_zero_confidence() {
        let po = PredictionOutput::new(PredictionType::AnomalyScore, 0.5, 0.0);
        let enc = PredictionEncoder::new(4, 4, PredEncodingMethod::Confidence);
        let pv = enc.encode(&po);
        // All raw values multiplied by 0 → zero
        assert!(pv.vector.iter().all(|x| x.abs() < 1e-10));
    }

    #[test]
    fn test_perfect_confidence() {
        let po = PredictionOutput::new(PredictionType::Action, 10.0, 1.0);
        let enc = PredictionEncoder::new(4, 4, PredEncodingMethod::Confidence);
        let pv = enc.encode(&po);
        assert!((pv.vector[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_encoding_determinism() {
        let enc = PredictionEncoder::new(4, 4, PredEncodingMethod::Raw);
        let po = PredictionOutput::new(PredictionType::ValuePrediction, 42.0, 0.77);
        let v1 = enc.encode(&po).vector;
        let v2 = enc.encode(&po).vector;
        assert_eq!(v1, v2);
    }
}
