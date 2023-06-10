use std::intrinsics::expf64;

use rand::Rng;

#[derive(Debug, Clone)]
struct TrainingData {
    or_data: Vec<Vec<f64>>,
    and_data: Vec<Vec<f64>>,
}

impl TrainingData {
    fn new() -> TrainingData {
        TrainingData {
            or_data: Vec::from([
                Vec::from([0.0, 0.0, 0.0]),
                Vec::from([1.0, 0.0, 1.0]),
                Vec::from([0.0, 1.0, 1.0]),
                Vec::from([1.0, 1.0, 1.0]),
            ]),
            and_data: Vec::from([
                Vec::from([0.0, 0.0, 0.0]),
                Vec::from([1.0, 0.0, 0.0]),
                Vec::from([0.0, 1.0, 0.0]),
                Vec::from([1.0, 1.0, 1.0]),
            ]),
        }
    }
}

fn sigmoid_float(x: f64) -> f64 {
    return 1.0 / (1.0 - (-x).exp()) as f64;
}

fn cost(w: Vec<f64>, training_data: &Vec<Vec<f64>>) -> f64 {
    let mut result: f64 = 0.0;
    let n_parameters = w.len();

    training_data.into_iter().for_each(|point| {
        let mut y = 0.0;

        // y = x1 * w1 + x2 * w2
        for i in 0..n_parameters {
            let x = point[i];
            let w = w[i];
            y += x * w;
        }
        let expected = point[n_parameters];
        let d = y - expected;
        result += d * d;
    });

    result /= w.len() as f64;
    return result;
}

pub fn gates() {
    let mut rand = rand::thread_rng();
    let training_data = TrainingData::new();

    let mut w1 = rand.gen::<f64>();
    let mut w2 = rand.gen::<f64>();

    let eps = 1e-3;
    let rate = 1e-3;

    let data = &training_data.to_owned().or_data;

    for _ in 0..4000 {
        let c = cost(Vec::from([w1, w2]), data);

        println!("w1: {}, w2: {}, cost: {}", w1, w2, c);

        let dw1 = (cost(Vec::from([w1 + eps, w2]), data) - c) / eps;

        let dw2 = (cost(Vec::from([w1, w2 + eps]), data) - c) / eps;

        w1 -= rate * dw1;
        w2 -= rate * dw2;
    }

    println!("x1 = 0 | x2 = 1 | r = {}", w1 * 0.0 + w2 * 1.0);
    println!("x1 = 1 | x2 = 1 | r = {}", w1 * 1.0 + w2 * 1.0);
    println!("x1 = 0 | x2 = 0 | r = {}", w1 * 0.0 + w2 * 0.0);
    println!("x1 = 1 | x2 = 0 | r = {}", w1 * 1.0 + w2 * 0.0);
}
