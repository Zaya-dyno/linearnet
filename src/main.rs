mod regres;

use regres::tensor::{Tensor,Tens};
use regres::linearnet::{LinearNet};
use regres::shape::{Shape,ituple,matsh};
use thiserror::Error;
use std::iter::repeat;

fn main() {
    let ln = LinearNet::new(((5,10),(10,3)));
    let x = Tensor::new((2,5),vec![1.0;10]);
    println!("{:#?}",ln.call(&x));
}
