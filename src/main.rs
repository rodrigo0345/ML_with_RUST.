use rand::Rng;

#[derive(Debug, Clone)]
struct TrainingData {
    data: Vec<Vec<f64>>,
}

impl TrainingData {
    fn new() -> TrainingData {
        TrainingData {
            data: Vec::from([
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
fn cost(w: f64, training_data: &TrainingData) -> f64 {
    let mut result: f64 = 0.0;
    training_data.to_owned().data.into_iter().for_each(|point| {
        let y = point.get(0).unwrap() * w;
        let d = y - point.get(1).unwrap();
        result += d.powi(2) as f64;
    });

    return result;
}

// coding the derivative
// with finite difference
// (only to learn)
// a is the parameter of the algorithm
// ex: y = x * a, a is the unknown parameter
// h is the value to add, in order to find the stationary point
fn derivate_u_finite_dif(a: f64, h: f64, training_data: &TrainingData) -> f64 {
    let dcost = (cost(a + h, &training_data) - cost(a, &training_data)) / h;

    return dcost;
}

fn main() {
    let mut rand = rand::thread_rng();

    let mut w: f64 = rand.gen_range(0..40) as f64;

    let training_data = TrainingData::new();

    println!("before: {}, w value = {}", cost(w, &training_data), w);

    let eps = 1e-3;
    let rate = 1e-2;

    for _ in 0..100 {
        let dcost = derivate_u_finite_dif(w, eps, &training_data);

        w -= dcost * rate;
    }

    println!("w: {}", w);
}
