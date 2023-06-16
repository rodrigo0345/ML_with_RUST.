mod gates;
mod matrix;
mod one_brain_cell;
mod xor;

use matrix::Matrix;
use xor::xor_model;

fn main() {
    //one_brain_cell();
    //gates();
    //xor_model();

    let mut mat1 = Matrix::new(2, 2);
    mat1.fill(2.0);
    let mut mat2 = Matrix::new(2, 2);
    mat2.fill(2.0);

    println!("{:?}", mat1.mult(mat2));
}
