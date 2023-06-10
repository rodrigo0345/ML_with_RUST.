use std::{alloc::System, sync::mpsc::TryRecvError};

use rand::Rng;

struct TrainingData {
    xor_data: Vec<Vec<f64>>,
    or_data: Vec<Vec<f64>>,
    and_data: Vec<Vec<f64>>,
    nand_data: Vec<Vec<f64>>,
}

impl TrainingData {
    fn new() -> TrainingData {
        TrainingData {
            xor_data: Vec::from([
                Vec::from([0.0, 0.0, 0.0]),
                Vec::from([1.0, 0.0, 1.0]),
                Vec::from([0.0, 1.0, 1.0]),
                Vec::from([1.0, 1.0, 0.0]),
            ]),
            or_data: Vec::from([
                Vec::from([0.0, 0.0, 0.0]),
                Vec::from([1.0, 0.0, 1.0]),
                Vec::from([0.0, 1.0, 1.0]),
                Vec::from([1.0, 1.0, 1.0]),
            ]),
            and_data: Vec::from([
                Vec::from([0.0, 0.0, 0.0]),
                Vec::from([0.3, 0.0, 0.0]),
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
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Xor {
    or_w1: f64,
    or_w2: f64,
    or_b: f64,

    nand_w1: f64,
    nand_w2: f64,
    nand_b: f64,

    and_w1: f64,
    and_w2: f64,
    and_b: f64,
}

impl Xor {
    fn new_random() -> Xor {
        let mut rand = rand::thread_rng();

        Xor {
            and_b: rand.gen(),
            and_w1: rand.gen(),
            and_w2: rand.gen(),
            nand_b: rand.gen(),
            nand_w1: rand.gen(),
            nand_w2: rand.gen(),
            or_b: rand.gen(),
            or_w1: rand.gen(),
            or_w2: rand.gen(),
        }
    }

    fn forward(&self, x1: f64, x2: f64) -> f64 {
        let a = sigmoid_float((self.or_w1 * x1) + (self.or_w2 * x2) + self.or_b);
        let b = sigmoid_float((self.nand_w1 * x1) + (self.nand_w1 * x2) + self.nand_b);

        return sigmoid_float((self.and_w1 * a + self.and_w2 * b) + self.and_b);
    }

    fn finite_diff(&mut self, c: f64, eps: f64, data: &Vec<Vec<f64>>) -> Xor {
        let mut result = Xor::new_random();
        let mut saved: f64;

        saved = self.or_w1;
        self.or_w1 += eps;
        result.or_w1 = (cost(*self, data) - c) / eps;
        self.or_w1 = saved;

        saved = self.or_w2;
        self.or_w2 += eps;
        result.or_w2 = (cost(*self, data) - c) / eps;
        self.or_w2 = saved;

        saved = self.or_b;
        self.or_b += eps;
        result.or_b = (cost(*self, data) - c) / eps;
        self.or_b = saved;

        saved = self.nand_w1;
        self.nand_w1 += eps;
        result.nand_w1 = (cost(*self, data) - c) / eps;
        self.nand_w1 = saved;

        saved = self.nand_w2;
        self.nand_w2 += eps;
        result.nand_w2 = (cost(*self, data) - c) / eps;
        self.nand_w2 = saved;

        saved = self.nand_b;
        self.nand_b += eps;
        result.nand_b = (cost(*self, data) - c) / eps;
        self.nand_b = saved;

        saved = self.and_w1;
        self.and_w1 += eps;
        result.and_w1 = (cost(*self, data) - c) / eps;
        self.and_w1 = saved;

        saved = self.and_w2;
        self.and_w2 += eps;
        result.and_w2 = (cost(*self, data) - c) / eps;
        self.and_w2 = saved;

        saved = self.and_b;
        self.and_b += eps;
        result.and_b = (cost(*self, data) - c) / eps;
        self.and_b = saved;

        return result;
    }

    fn learn(&mut self, g: Xor, rate: f64) -> Xor {
        self.or_w1 -= g.or_w1 * rate;
        self.or_w2 -= g.or_w2 * rate;
        self.or_b -= g.or_b * rate;

        self.nand_w1 -= g.nand_w1 * rate;
        self.nand_w2 -= g.nand_w2 * rate;
        self.nand_b -= g.nand_b * rate;

        self.and_w1 -= g.and_w1 * rate;
        self.and_w2 -= g.and_w2 * rate;
        self.and_b -= g.and_b * rate;

        return *self;
    }
}

fn sigmoid_float(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp()) as f64;
}

fn cost(m: Xor, data: &Vec<Vec<f64>>) -> f64 {
    let mut result = 0.0;

    data.into_iter().for_each(|point| {
        let x1 = point[0];
        let x2 = point[1];
        let y = m.forward(x1, x2);
        let d = y - point[2];
        result += d * d;
    });

    result /= (data.len()) as f64;
    return result;
}

pub fn xor_model() {
    let td = TrainingData::new();
    let data = td.nand_data;

    let mut xor = Xor::new_random();

    let eps = 1e-3;
    let rate = 1e-1;

    for _ in 0..1000 * 100 {
        let c = cost(xor, &data);
        let dif = xor.finite_diff(c, eps, &data);
        let fin = xor.learn(dif, rate);
        xor = fin;
    }

    println!("cost: {:?}", cost(xor, &data));

    for i in 0..2 {
        for j in 0..2 {
            println!("{} xor {} = {}", i, j, xor.forward(i as f64, j as f64));
        }
    }
}
