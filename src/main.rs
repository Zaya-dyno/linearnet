mod regres;

use regres::tensor::{Tensor,Tens};
use regres::linearnet::{LinearNet};
use regres::shape::{Shape,ituple,matsh};
use thiserror::Error;
use std::iter::repeat;

fn main() {
    let mut x = Tensor::new::<matsh>((2,2,false),vec![1.0,3.0,
                                                      7.0,11.0]);
    let mut y = Tensor::new::<matsh>((2,2,false),vec![2.0,5.0,
                                                      13.0,17.0]);
    println!("{:#?}",x.dot(&y));
}
