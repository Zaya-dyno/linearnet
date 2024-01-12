mod regres;

use regres::tensor::{Tensor,Tens};
use regres::linearnet::{LinearNet};
use regres::shape::{Shape,ituple,matsh};
use thiserror::Error;
use std::iter::repeat;

fn main() {
    let ln = LinearNet::new(((3,5),(5,4)));
    let x = Tensor::random_uniform((6,3));
    println!("{:#?}",ln.call(&x).unwrap());
}
