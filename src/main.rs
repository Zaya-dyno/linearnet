mod regres;

use regres::tensor::{Tensor,Shape,Tens};
use regres::nn::Hello;
use thiserror::Error;

#[derive(Debug)]
struct LinearNet {
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
        if !x.flat_tensor() {
            eprintln!("x tensor has {} shape",
                      x.shape);
            return Err( Box::new(LNError {
                message: "cannot call LinearNet on non-flat tensor".to_string(),
                line: 0,
                column: 0,
            }));
        }
        Ok(x.multi(&self.l1)?.relu()?.multi(&self.l2)?)
    }
}

fn main() {
    let ln = LinearNet::new(((5,10),(10,3)));
    let x = Tensor::new_flat(5,vec![1.0;5]);
    let y = match ln.call(&x) {
        Ok(x) => x,
        Err(error) => panic!("problem Relu: {:?}", error),
    };
    println!("{:?}",y);
    Hello();
}
