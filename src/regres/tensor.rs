use rand::distributions::Uniform;
use rand::{thread_rng,Rng};
use rand::rngs::ThreadRng;
use std::fmt;
use thiserror::Error;
use std::iter::{repeat,zip};
use std::ops::Range;
use super::shape::{ituple,Shape,Degree};
use super::shape::Degree::{SCA,VEC,MAT};
pub type Tens = f64;

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
    val: Vec<Tens>, 
}

impl Tensor {
    // Not for production
    pub fn empty(degree: Degree) -> Tensor {
        match degree {
            SCA => Self::new_empty(Shape::scalar()),
            VEC => Self::new_empty::<i32>(0.into()),
            MAT => Self::new_empty::<ituple>((0,0).into()),
        }
    }

    pub fn new_empty<A>(shape: A) -> Tensor
        where A: Into<Shape>
    {
        let shape = shape.into();
        Tensor {
            shape,
            val: vec![],
        }
    }

    pub fn new<A>(shape: A, val: Vec<Tens>) -> Tensor
        where A: Into<Shape>
    {
        let shape = shape.into();
        Tensor {
            shape,
            val,
        }
    }

    pub fn random_uniform<A>(shape:A) -> Tensor 
        where A: Into<Shape>
    {
        let shape: Shape = shape.into();
        let size = shape.size();

        let mut rng = thread_rng();
        let side = Uniform::new(-1.0,1.0);
        let val = (0..(shape.size())).map(|_| rng.sample(side)).collect();
        Tensor {
            shape,
            val,
        }
    }

    fn horizontal(&self, i: i32) ->  Box<dyn Iterator<Item = &Tens> + '_>
    {
        Box::new(self.val.iter().skip((self.shape.dim.1*i) as usize))
    }

    fn vertical(&self, i: i32) -> Box<dyn Iterator<Item = &Tens> + '_>
    {
        Box::new(self.val.iter().skip(i as usize).step_by(self.shape.dim.1 as usize))
    }

    fn row(&self, i: i32) ->  Box<dyn Iterator<Item = &Tens> + '_> {
        if (self.shape.T) {
            self.vertical(i)
        } else {
            self.horizontal(i)
        }
    }

    fn column(&self, i: i32) ->  Box<dyn Iterator<Item = &Tens> + '_> {
        if (self.shape.T) {
            self.horizontal(i)
        } else {
            self.vertical(i)
        }
    }

    fn can_dot_shape(lhs: Shape,mut rhs: Shape) -> bool {
        match (lhs.degree, rhs.degree) {
            (SCA, _ ) => true,
            ( _ ,SCA) => true,
            (VEC,VEC) => lhs.dim.0 == rhs.dim.0,
            (VEC,MAT) => (lhs.dim.0 == rhs.dim.0) |
                         (lhs.dim.0 == rhs.dim.1),
            (MAT,VEC) => (rhs.dim.0 == lhs.dim.0) |
                         (rhs.dim.0 == lhs.dim.1),
            (MAT,MAT) => (rhs.dim.1 == lhs.dim.0),
        }
    }

    fn accum_sum<'a, A>(lhs:A,rhs:A) -> Tens
        where A: Iterator<Item = &'a Tens>
    {
        let mut sum = 0.0;
        for (l,r) in zip(lhs,rhs) {
            sum += *l * *r;
        }
        sum
    }

    #[inline(always)]
    pub fn vector(&self) -> bool {
        self.shape.degree == VEC
    }
    #[inline(always)]
    pub fn scalar(&self) -> bool {
        self.shape.degree == SCA 
    }
    #[inline(always)]
    pub fn matrix(&self) -> bool {
        self.shape.degree == MAT 
    }

    pub fn softmax(&self) -> Result<Tensor, TensorError> {
        if self.scalar() {
            eprintln!("Current tensor has {} shape",
                      self.shape);
            return Err( TensorError {
                message: "cannot softmax on scalar".to_string(),
                line: line!() as usize,
                column: column!() as usize,
            });
        }
        let mut exps: Vec<f64>= self.val.iter().map(|a| a.exp()).collect();
        let chunks = match self.vector() {
            true => self.shape.dim.0,
            false => self.shape.dim.1,
        };
        let sums:Vec<f64> = exps.chunks(chunks as usize).map(|a| a.iter().sum()).collect();
        for i in 0..exps.len() {
            exps[i] = exps[i] / sums[(i as i32/chunks) as usize]
        }
        Ok(Tensor::new(self.shape,
                       exps))
    }

    pub fn relu(&self) -> Result<Tensor,TensorError> {
        if self.scalar() {
            eprintln!("Current tensor has {} shape",
                      self.shape);
            return Err( TensorError {
                message: "cannot relu on scalar".to_string(),
                line: line!() as usize,
                column: column!() as usize,
            });
        }
        let mut ret = Vec::with_capacity(self.shape.size());
        for i in &self.val {
            if *i < 0.0 {
                ret.push( 0.0 );
            } else {
                ret.push(  *i  );
            }
        }
        Ok(Tensor::new(self.shape,
                       ret))
    }

    pub fn dot(&self, rhs: &Tensor) -> Result<Tensor,TensorError> {
        if !Self::can_dot_shape(self.shape,rhs.shape) {
            eprintln!("Tried to multiply {} shape with {} shape",
                      self.shape, rhs.shape);
            return Err( TensorError {
                message: "wrong size tensors".to_string(),
                line: line!() as usize,
                column: column!() as usize,
            });
        }
        match (self.shape.degree, rhs.shape.degree) {
           (SCA, _ ) => Ok(Tensor::new_empty::<i32>(0.into())),//dot_scalar_matrix(self.val[0],rhs),
           ( _ ,SCA) => Ok(Tensor::new_empty::<i32>(0.into())),//dot_scalar_matrix(rhs.val[0],&self),
           (VEC,VEC) => Ok(Tensor::new_empty::<i32>(0.into())),//dot_vector_vector(&self,rhs),
           (VEC,MAT) => Self::dot_vector_matrix(&self,rhs), 
           (MAT,VEC) => Ok(Tensor::new_empty::<i32>(0.into())),//dot_vector_matrix(rhs,&self),
           (MAT,MAT) => Self::dot_matrix_matrix(&self,rhs),
        }           
    }      


    pub fn dot_matrix_matrix(lhs : &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError>
    {
        let ret_shape: Shape = (lhs.shape.dim.0,rhs.shape.dim.1).into();
        let mut ret = Vec::with_capacity(ret_shape.size());
        for i in 0..lhs.shape.dim.0 {
            for j in 0..rhs.shape.dim.1 {
                println!("{:#?}",lhs.row(i).map(|a| *a).collect::<Vec<f64>>());
                println!("{:#?}",rhs.column(j).map(|a| *a).collect::<Vec<f64>>());
                ret.push(Self::accum_sum(lhs.row(i),
                                         rhs.column(j)));
            }
        }
        Ok( Tensor::new(ret_shape,
                        ret))
    }

        
    pub fn dot_vector_matrix(lhs : &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError>
    {
        //let mut ret = Vec::with_capacity(rhs.shape.dim.1 as usize );
        //for j in 0..rhs.shape.dim.1 {
        //    let r_start = (j*lhs.shape.dim.0) as usize;
        //    let r_end = r_start + rhs.shape.dim.0 as usize;
        //    let r_range:Range<usize> = r_start..r_end;
        //    ret.push(Self::accum_sum(&lhs.val,
        //                             &rhs.val[r_range],
        //                             rhs.shape.dim.0 as usize));
        //}
        //Ok( Tensor::new::<i32>( rhs.shape.dim.1.into(),
        //                        ret))
        Ok( Tensor::empty(MAT) )
    }

    // return single element vector when self is vector.
    //pub fn cross_entropy_loss(&self, labels: &[i32]) -> Vec<f64> {
    //irintln!("{:#?}",loss);
    //    if self.scalar() {
    //        eprintln!("Current tensor has {} shape",
    //                  self.shape);
    //        return Err( TensorError {
    //            message: "cannot entropy loss non-flat tensor".to_string(),
    //            line: line!() as usize,
    //            column: column!() as usize,
    //        });
    //    }
    //    if self.shape.dim.0 != labels.len() {
    //        eprintln!("Current tensor has {} shape",
    //                  self.shape);
    //        return Err( TensorError {
    //            message: "cannot entropy loss wrong length label".to_string(),
    //            line: line!() as usize,
    //            column: column!() as usize,
    //        });
    //    }
    //    // check non of the labels are higher than number of class 
    //    if self.shape.dim.0 < labels.iter().max() {
    //        eprintln!("Labels has {} value, when max is",
    //                  labels.iter().max(),self.shape.dim.0);
    //        return Err( TensorError {
    //            message: "wrong label".to_string(),
    //            line: line!() as usize;
    //            column: column!() as usize;
    //        })
    //    }
    //}
}
