use rand::Rng;

#[derive(Debug, Clone)]
struct TrainingData {
    data: Vec<Vec<f32>>,
}

impl TrainingData {
    fn new() -> TrainingData {
        TrainingData {
            data: Vec::from([
                Vec::from([0.0, 0.0]),
                Vec::from([1.0, 2.0]),
                Vec::from([2.0, 4.0]),
                Vec::from([3.0, 6.0]),
            ]),
        }
    }
}

fn cost(w: f32, training_data: &TrainingData) -> f64 {
    let mut result: f64 = 0.0;
    training_data.to_owned().data.into_iter().for_each(|point| {
        let x = &point[0];
        let y = point[0] * w;
        let d = y - point[1];
        result += d.powi(2) as f64;
    });

    return result;
}

fn main() {
    let mut rand = rand::thread_rng();

    let mut w: f64 = rand.gen_range(0..10) as f64;
    let training_data = TrainingData::new();

    let eps = 1e-5;

    // coding the derivative
    // with finite difference
    // (only to learn)
    let dcost =
        cost(w as f32 - eps, &training_data) - cost(w as f32, &training_data) / (eps as f64);
    w -= dcost;
}
