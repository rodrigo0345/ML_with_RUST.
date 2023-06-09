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

// coding the derivative
// with finite difference
// (only to learn)
// a is the parameter of the algorithm
// ex: y = x * a, a is the unknown parameter
// h is the value to add, in order to find the stationary point
fn derivate_u_finite_dif(a: f64, h: f64, training_data: &TrainingData) -> f64 {
    let dcost = cost((a + h) as f32, &training_data) - cost(a as f32, &training_data) / (h as f64);
    return a - dcost;
}

fn main() {
    let mut rand = rand::thread_rng();

    let mut w: f64 = rand.gen_range(0..10) as f64;
    let training_data = TrainingData::new();

    println!(
        "before: {}, w value = {}",
        cost(w as f32, &training_data),
        w
    );

    let eps = 1e-5;
    let res = derivate_u_finite_dif(w, eps as f64, &training_data);

    println!("after: {}", res);
}
