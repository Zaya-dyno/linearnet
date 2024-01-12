use super::tensor::{Tensor,Tens};
use super::shape::{Shape};
use thiserror::Error;

#[derive(Debug)]
pub struct LinearNet {
    l1: Tensor,
    l2: Tensor,
}
#[derive(Error,Debug)]
#[error("{message:} ({line:}, {column})")]
pub struct LNError {
    message: String,
    line: usize,
    column: usize,
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

impl LinearNet {
    pub fn new_with_vec<A>(shapes: (A,A),
                           vecs: (Vec<Tens>,Vec<Tens>)) -> LinearNet 
        where A: Into<Shape>
    {
        let (l1_s, l2_s) = (shapes.0.into(), shapes.1.into());
        let (l1_v, l2_v) = vecs;
        let l1 = Tensor::new(l1_s, l1_v);
        let l2 = Tensor::new(l2_s, l2_v);
        LinearNet {
            l1,
            l2,
        }
    }
    pub fn new<A>(shapes: (A,A)) -> LinearNet 
        where A: Into<Shape>
    {
        let l1 = Tensor::random_uniform(shapes.0);
        let l2 = Tensor::random_uniform(shapes.1);
        LinearNet {
            l1,
            l2,
        }
    }

    pub fn call(&self, x: &Tensor) -> Result<Tensor> {
        if x.scalar() {
            eprintln!("x tensor has {} shape",
                      x.shape);
            return Err( Box::new(LNError {
                message: "cannot call LinearNet on scalar".to_string(),
                line: line!() as usize,
                column: column!() as usize,
            }));
        }
        println!("{:#?}",x.dot(&self.l1)?.relu()?.dot(&self.l2).unwrap());
        Ok(x.dot(&self.l1)?.relu()?.dot(&self.l2)?.softmax()?)
    }
}

