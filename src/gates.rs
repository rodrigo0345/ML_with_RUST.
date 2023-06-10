use rand::Rng;

#[derive(Debug, Clone)]
struct TrainingData {
    or_data: Vec<Vec<f64>>,
    and_data: Vec<Vec<f64>>,
    nand_data: Vec<Vec<f64>>,
    xor_data: Vec<Vec<f64>>,
}

impl TrainingData {
    fn new() -> TrainingData {
        TrainingData {
            or_data: Vec::from([
                Vec::from([0.0, 0.0, 0.0]),
                Vec::from([0.3, 0.0, 1.0]),
                Vec::from([1.0, 0.0, 1.0]),
                Vec::from([0.0, 1.0, 1.0]),
                Vec::from([1.0, 1.0, 1.0]),
            ]),
            and_data: Vec::from([
                Vec::from([0.0, 0.0, 0.0]),
                Vec::from([0.3, 0.0, 0.0]),
                Vec::from([5.0, 0.0, 0.0]),
                Vec::from([1.0, 0.0, 0.0]),
                Vec::from([0.0, 1.0, 0.0]),
                Vec::from([1.0, 1.0, 1.0]),
            ]),
            nand_data: Vec::from([
                Vec::from([0.0, 0.0, 1.0]),
                Vec::from([1.0, 0.0, 1.0]),
                Vec::from([0.0, 1.0, 1.0]),
                Vec::from([1.0, 1.0, 0.0]),
            ]),
            xor_data: Vec::from([
                Vec::from([0.0, 0.0, 0.0]),
                Vec::from([1.0, 0.0, 1.0]),
                Vec::from([0.0, 1.0, 1.0]),
                Vec::from([1.0, 1.0, 0.0]),
            ]),
        }
    }
}

struct Neuron {
    pub weights: Vec<f64>,
    pub b: f64,
}

impl Neuron {
    fn new(weights: Vec<f64>, b: f64) -> Neuron {
        Neuron {
            weights: weights,
            b: b,
        }
    }

    fn new_default() -> Neuron {
        let mut rand = rand::thread_rng();
        let weights = Vec::from([rand.gen::<f64>(), rand.gen::<f64>()]);

        Neuron {
            weights: weights,
            b: rand.gen::<f64>(),
        }
    }

    fn one_iter(&mut self, data: &Vec<Vec<f64>>, eps: f64, rate: f64) {
        let c = cost(self.weights.to_owned(), self.b, data);

        let dw1 = (cost(
            Vec::from([self.weights[0] + eps, self.weights[1]]),
            self.b,
            data,
        ) - c)
            / eps;

        let dw2 = (cost(
            Vec::from([self.weights[0], self.weights[1] + eps]),
            self.b,
            data,
        ) - c)
            / eps;

        let db = (cost(self.weights.to_owned(), self.b + eps, data) - c) / eps;

        let w1 = self.weights[0] - rate * dw1;
        let w2 = self.weights[1] - rate * dw2;

        self.weights.clear();
        self.weights.push(w1);
        self.weights.push(w2);
        self.b -= rate * db;
    }

    fn result(&self, x: f64, y: f64) -> f64 {
        return sigmoid_float(x * self.weights[0] + y * self.weights[1] + self.b);
    }
}

fn sigmoid_float(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp()) as f64;
}

fn cost(w: Vec<f64>, b: f64, training_data: &Vec<Vec<f64>>) -> f64 {
    let mut result: f64 = 0.0;
    let n_parameters = w.len();

    training_data.into_iter().for_each(|point| {
        let mut y = 0.0;

        // y = x1 * w1 + x2 * w2 + b
        for i in 0..n_parameters {
            let x = point[i];
            let w = w[i];
            y += x * w;
        }
        let expected = point[n_parameters];
        let sig_y = sigmoid_float(y + b);
        let d = sig_y - expected;
        result += d * d;
    });

    result /= (training_data.len()) as f64;
    return result;
}

pub fn gates() {
    let training_data = TrainingData::new();

    let mut neuron_or = Neuron::new_default();
    let mut neuron_nand = Neuron::new_default();
    let mut neuron_and = Neuron::new_default();

    let eps = 1e-1;
    let rate = 1e-1;

    let data = &training_data.to_owned().xor_data;

    for _ in 0..10000 {
        let x = neuron_and.weights[0];
        let y = neuron_and.weights[1];

        neuron_or.one_iter(data, eps, rate);
        let a = neuron_or.result(x, y);

        neuron_nand.one_iter(data, eps, rate);
        let b = neuron_nand.result(x, y);

        neuron_and.weights = Vec::from([a, b]);
        neuron_and.one_iter(data, eps, rate);

        println!(
            "w1: {}, w2: {}, b: {}, cost: {}",
            neuron_and.weights[0],
            neuron_and.weights[1],
            neuron_and.b,
            cost(neuron_and.weights.to_owned(), neuron_and.b, data)
        );
    }

    println!(
        "w1: {}, w2: {}, b: {}, cost: {}",
        neuron_and.weights[0],
        neuron_and.weights[1],
        neuron_and.b,
        cost(neuron_and.weights.to_owned(), neuron_and.b, data)
    );

    data.into_iter().for_each(|point| {
        println!(
            "{} | {} = {}, expected: {}",
            point[0],
            point[1],
            sigmoid_float(
                point[0] * neuron_and.weights[0] + point[1] * neuron_and.weights[1] + neuron_and.b
            ),
            point[2]
        );
    });
}
