use rand::distributions::Uniform;
use rand::{thread_rng,Rng};
use rand::rngs::ThreadRng;
use std::fmt;
use thiserror::Error;
use std::iter::{repeat,zip};
use Degree::{SCA,VEC,MAT};
use std::ops::Range;

pub type Tens = f64;
pub type ituple = (i32,i32);
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Degree {
    SCA,
    VEC,
    MAT,
}

#[derive(Debug, Copy, Clone)]
pub struct Shape {
    degree: Degree,
    dim: (i32,i32),
    pub T: bool,
}

impl Shape {
    pub fn new(degree: Degree, dim: (i32,i32)) -> Shape {
        Shape {
            degree,
            dim,
            T:false,
        }
    }
    pub fn scalar() -> Shape {
        Shape {
            degree:SCA,
            dim:(0,0),
            T:false,
        }
    }
    pub fn size(&self) -> usize {
        match self.degree {
            SCA => 1 as usize,
            VEC => self.dim.0 as usize,
            MAT => (self.dim.0 * self.dim.1) as usize,
        }
    }

}

impl Into<Shape> for (i32,i32) {
    fn into(self) -> Shape {
        Shape::new(MAT,self)
    }
}

impl Into<Shape> for i32 {
    fn into(self) -> Shape {
        Shape::new(VEC,(self,0))
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        match self.degree {
            SCA => write!(f,"()"),
            VEC => write!(f,"({})",self.dim.0),
            MAT => write!(f,"{:?}",self.dim),
        }
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
    val: Vec<Tens>, 
}

impl Tensor {
    // Not for production
    pub fn empty(degree: Degree) -> Tensor {
        match degree {
            SCA => Self::new_empty(Shape::scalar()),
            VEC => Self::new_empty(0.into()),
            MAT => Self::new_empty((0,0).into()),
        }
    }

    pub fn new_empty(shape: Shape) -> Tensor {
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

    #[inline(always)]
    fn can_dot_shape(lhs: Shape,mut rhs: Shape) -> bool {
        if lhs.T != rhs.T  {
            let t = rhs.dim.1;
            rhs.dim.1 = rhs.dim.0;
            rhs.dim.0 = t;
        }
        
        match (lhs.degree, rhs.degree) {
            (SCA, _ ) => true,
            ( _ ,SCA) => true,
            (VEC,VEC) => true,
            (VEC,MAT) => (lhs.dim.0 == rhs.dim.0) |
                     (lhs.dim.0 == rhs.dim.1),
            (MAT,VEC) => (rhs.dim.0 == lhs.dim.0) |
                     (rhs.dim.0 == lhs.dim.1),
            (MAT,MAT) => (rhs.dim.1 == lhs.dim.0),
        }
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
           (SCA, _ ) => Ok(Tensor::new_empty(0.into())),//dot_scalar_matrix(self.val[0],rhs),
           ( _ ,SCA) => Ok(Tensor::new_empty(0.into())),//dot_scalar_matrix(rhs.val[0],&self),
           (VEC,VEC) => Ok(Tensor::new_empty(0.into())),//dot_vector_vector(&self,rhs),
           (VEC,MAT) => Self::dot_vector_matrix(&self,rhs), 
           (MAT,VEC) => Ok(Tensor::new_empty(0.into())),//dot_vector_matrix(rhs,&self),
           (MAT,MAT) => Self::dot_matrix_matrix(&self,rhs),
        }           
    }      


    pub fn dot_matrix_matrix(lhs : &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError>
    {
        let mut ret = Vec::with_capacity((rhs.shape.dim.1 *
                                          lhs.shape.dim.1).try_into().unwrap() );
        for i in 0..lhs.shape.dim.1 {
            for j in 0..rhs.shape.dim.1 {
                let l_start = (i*lhs.shape.dim.0) as usize;
                let l_end = l_start + lhs.shape.dim.0 as usize;
                let r_start = (j*lhs.shape.dim.0) as usize;
                let r_end = r_start + rhs.shape.dim.0 as usize;
                let l_range:Range<usize> = l_start..l_end;
                let r_range:Range<usize> = r_start..r_end;
                ret.push(Self::accum_sum(&lhs.val[l_range],
                                         &rhs.val[r_range],
                                         rhs.shape.dim.0 as usize));
            }
        }
        let mut shape:Shape = (rhs.shape.dim.1,lhs.shape.dim.1).into();
        shape.T = true;
        Ok( Tensor::new(shape,
                        ret))
    }

        
    pub fn dot_vector_matrix(lhs : &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError>
    {
        let mut ret = Vec::with_capacity(rhs.shape.dim.1 as usize );
        for j in 0..rhs.shape.dim.1 {
            let r_start = (j*lhs.shape.dim.0) as usize;
            let r_end = r_start + rhs.shape.dim.0 as usize;
            let r_range:Range<usize> = r_start..r_end;
            ret.push(Self::accum_sum(&lhs.val,
                                     &rhs.val[r_range],
                                     rhs.shape.dim.0 as usize));
        }
        Ok( Tensor::new::<i32>( rhs.shape.dim.1.into(),
                                ret))
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
