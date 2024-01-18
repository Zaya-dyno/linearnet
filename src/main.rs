mod regres;

use regres::tensor::{Tensor,Tens};
use regres::linearnet::{LinearNet};


fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() {
    let ln = LinearNet::new(((3,5),(5,4)));
    let x = Tensor::random_uniform((6,3));
    let y = vec![1,2,3,4,1,2];
    let y_hat = ln.call(&x).unwrap(); 
    let loss = y_hat.cross_entropy_loss(&y).unwrap();
    println!("Total loss: {:?}",loss.iter().sum::<Tens>());
}
