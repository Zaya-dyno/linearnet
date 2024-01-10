use rand::distributions::Uniform;
use rand::{thread_rng,Rng};
use rand::rngs::ThreadRng;
use std::fmt;
use thiserror::Error;
use std::iter::{repeat,zip};

pub type Tens = f64;

#[derive(Debug, Copy, Clone)]
pub struct Shape {
    first: i32,
    second: i32,
}


impl Shape {
    pub fn new(first: i32, second: i32) -> Shape {
        Shape {
            first,
            second,
        }
    }
}

impl Into<Shape> for (i32,i32) {
    fn into(self) -> Shape {
        Shape {
            first: self.0,
            second: self.1,
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        write!(f,"({},{})",self.first,self.second)
    }
}

#[derive(Error,Debug)]
#[error("{message:} ({line:}, {column})")]
pub struct TensorError {
    message: String,
    line: usize,
    column: usize,
}

#[derive(Debug)]
pub struct Tensor{
    pub shape: Shape,
    // size shape.second
    bias: Vec<Tens>, 
    // size shape.first * shape.second
    // w01, w02 .. w(shape.second)(shape.first)
    weights: Vec<Tens>, 
}

impl Tensor {
    // Not for production
    pub fn empty() -> Tensor {
        Self::new_flat(1,vec![1.0])
    }


    pub fn random_uniform<A>(shape:A) -> Tensor 
        where A: Into<Shape>
    {
        let shape: Shape = shape.into();
        let mut rng = thread_rng();
        let side = Uniform::new(-1.0,1.0);
        let bias = (0..shape.second).map(|_| rng.sample(side)).collect();
        let weights = (0..(shape.second*shape.first)).map(|_| rng.sample(side)).collect();
        Tensor {
            shape,
            bias,
            weights,
        }
    }

    pub fn new_flat(size: i32, val:Vec<Tens>)->Tensor{
        Tensor {
            shape:Shape::new(0,size),
            bias:val,
            weights:vec![],
        }
    }
    
    #[inline(always)]
    fn can_multi_shape(lhs: Shape, rhs: Shape) -> bool {
        lhs.second == rhs.first
    }

    fn accum_sum(lhs:&[f64],rhs:&[f64],size: usize) -> Tens
    {
        let mut sum = 0.0;
        for i in 0..size {
            sum += lhs[i] * rhs[i];
        }
        sum
    }

    #[inline(always)]
    pub fn flat_tensor(&self) -> bool {
        self.shape.first == 0
    }

    pub fn softmax(mut self) -> Result<Tensor, TensorError> {
        if !self.flat_tensor() {
            eprintln!("Current tensor has {} shape",
                      self.shape);
            return Err( TensorError {
                message: "cannot softmax non-flat tensor".to_string(),
                line: 0,
                column: 0,
            });
        }
        let mut exps: Vec<f64>= (0..self.shape.second).map(|i| self.bias[i as usize].exp()).collect();
        let sum: f64 = exps.iter().sum();
        Ok(Tensor::new_flat(self.shape.second,
                            exps.iter().map(|i| i/sum).collect()))
    }

    pub fn relu(mut self) -> Result<Tensor,TensorError> {
        if !self.flat_tensor() {
            eprintln!("Current tensor has {} shape",
                      self.shape);
            return Err( TensorError {
                message: "cannot relu non-flat tensor".to_string(),
                line: 0,
                column: 0,
            });
        }
        for i in 0..self.shape.second {
            if self.bias[i as usize] < 0.0 {
                self.bias[i as usize] = 0.0;
            }
        }
        Ok(self)
    }

    pub fn multi(&self, rhs: &Tensor) -> Result<Tensor,TensorError> {
        if !Self::can_multi_shape(self.shape,rhs.shape) {
            eprintln!("Tried to multiply {} shape with {} shape",
                      self.shape, rhs.shape);
            return Err( TensorError {
                message: "wrong size tensors".to_string(),
                line: 0,
                column: 0,
            });
        }
        let mut ret = vec![0.0;rhs.shape.second as usize];
        let mut binding = rhs.weights.chunks(rhs.shape.first as usize);
        let weights = binding.by_ref();
        let bias = rhs.bias.iter().by_ref();
        for (i,(x,y)) in zip(weights,
                             repeat(&self.bias)).enumerate() {
            ret[i] = Self::accum_sum(x,y,rhs.shape.first as usize);
        }
        Ok(Tensor::new_flat(rhs.shape.second,ret))
    }
}
