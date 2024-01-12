mod regres;

use regres::tensor::{Tensor};
use regres::linearnet::{LinearNet};




fn main() {
    let ln = LinearNet::new(((3,5),(5,4)));
    let x = Tensor::random_uniform((6,3));
    println!("{:#?}",ln.call(&x).unwrap());
}
