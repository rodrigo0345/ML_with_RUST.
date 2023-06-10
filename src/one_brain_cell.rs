use rand::Rng;

#[derive(Debug, Clone)]
struct TrainingData {
    data: Vec<Vec<f64>>,
}

impl TrainingData {
    fn new() -> TrainingData {
        TrainingData {
            data: Vec::from([
                Vec::from([0.0, 0.0]),
                Vec::from([1.0, 2.0]),
                Vec::from([2.0, 4.0]),
                Vec::from([3.0, 6.0]),
                Vec::from([4.0, 8.0]),
            ]),
        }
    }
}

// basicaly this is just
// a perceptron
// with only one weight
// and bias = 0
fn cost(w: f64, b: f64, training_data: &TrainingData) -> f64 {
    let mut result: f64 = 0.0;
    training_data.to_owned().data.into_iter().for_each(|point| {
        let y = point.get(0).unwrap() * w + b;
        let d = y - point.get(1).unwrap();
        result += d.powi(2) as f64;
    });

    return result;
}

pub fn one_brain_cell() {
    let mut rand = rand::thread_rng();

    let mut w: f64 = rand.gen_range(1..5) as f64;
    let mut b: f64 = rand.gen_range(0..5) as f64;

    let training_data = TrainingData::new();

    println!(
        "before: {}, w value = {}, b value = {}",
        cost(w, b, &training_data),
        w,
        b
    );

    let eps = 1e-3;
    let rate = 1e-2;

    let mut error: f64 = 10.0;
    while error > 0.00005 {
        //for _ in 0..100 {
        let c = cost(w, b, &training_data);
        let dweight = (cost(w + eps, b, &training_data) - c) / eps;
        let dbias = (cost(w, b + eps, &training_data) - c) / eps;

        w -= dweight * rate;
        b -= dbias * rate;
        error = cost(w, b, &training_data);
    }

    println!(
        "after: {}, w value = {}, b value = {}",
        cost(w, b, &training_data),
        w,
        b
    );
}
