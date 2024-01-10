mod regres;

use regres::tensor::{Tensor,Shape,Tens};
use regres::linearnet::{LinearNet};
use thiserror::Error;
use std::iter::repeat;

fn main() {
    let ln = LinearNet::new(((5,10),(10,3)));
    let x = Tensor::new_flat(5,vec![1.0;5]);
    let label: Vec<i32> = repeat(0).take(5).collect();
    println!("{:?}",label);
    let y = match ln.call(&x) {
        Ok(x) => x,
        Err(error) => panic!("problem inference: {:?}", error),
    };
    println!("{:?}",y);
}
