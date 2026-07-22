use super::Tensor;
use std::ops::{Add, Mul};

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(self, rhs).expect("tensor addition failed")
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(self, rhs).expect("tensor multiplication failed")
    }
}
