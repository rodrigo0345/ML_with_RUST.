#[derive(Debug, Clone, PartialEq, PartialOrd, Default)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn fill(&mut self, value: f64) {
        for i in 0..self.data.len() {
            self.data[i] = value;
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        return Some(
            self.get_as_ref(row, col)
                .unwrap_or_else(|| {
                    panic!("Error in Matrix::get(): self.get_as_ref(row, col) returned None");
                })
                .clone(),
        );
    }

    pub fn get_as_ref(&self, row: usize, col: usize) -> Option<&f64> {
        if (row >= self.rows) || (col >= self.cols) {
            return None;
        }

        if col >= self.cols {
            return None;
        }

        return Some(&self.data[row * self.cols + col]);
    }

    pub fn get_as_mut_ref(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        if (row >= self.rows) || (col >= self.cols) {
            return None;
        }

        if col >= self.cols {
            return None;
        }

        return Some(&mut self.data[row * self.cols + col]);
    }

    pub fn mult(&self, mat2: Matrix) -> Option<Matrix> {
        if self.cols != mat2.rows {
            return None;
        }

        let mut result = Matrix::new(self.rows, mat2.cols);

        for i in 0..self.rows {
            for j in 0..mat2.cols {
                let mut sum = 0.0;

                for k in 0..self.cols {
                    sum += self.get(i, k).unwrap_or_else(|| {
                        panic!("Error in Matrix::mult(): self.get(i, k) returned None")
                    }) * mat2.get(k, j).unwrap_or_else(|| {
                        panic!("Error in Matrix::mult(): mat2.get(i, k) returned None")
                    });
                }

                *(result.get_as_mut_ref(i, j).unwrap()) = sum;
            }
        }

        return Some(result);
    }

    pub fn sum(&self, mat2: Matrix) -> Option<Matrix> {
        if (mat2.rows != self.rows) || (mat2.cols != self.cols) {
            return None;
        }

        if mat2.cols != self.cols {
            return None;
        }

        let mut result = Matrix::new(self.rows, self.cols);

        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + mat2.data[i];
        }

        return Some(result);
    }
}
