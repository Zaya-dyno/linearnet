mod regres;

use regres::tensor::{Tensor,Shape,Tens,ituple};
use regres::linearnet::{LinearNet};
use thiserror::Error;
use std::iter::repeat;

fn main() {
    let ln = LinearNet::new(((5,10),(10,3)));
    let mut x = Tensor::new::<ituple>((5,3),vec![1.0;15]);
    x.shape.T = true;
    let ans = ln.call(&x);
    println!("{:#?}",ans);
}
