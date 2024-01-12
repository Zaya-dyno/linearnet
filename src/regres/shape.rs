use std::fmt;
use Degree::{SCA,VEC,MAT};

pub type matsh = (i32,i32,bool);
pub type ituple = (i32,i32);
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Degree {
    SCA,
    VEC,
    MAT,
}

#[derive(Debug, Copy, Clone)]
pub struct Shape {
    pub degree: Degree,
    pub dim: ituple,
    pub T: bool,
}

impl Shape {
    pub fn new(degree: Degree, dim: ituple, T: bool) -> Shape {
        Shape {
            degree,
            dim,
            T,
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

impl Into<Shape> for (i32,i32,bool) {
    fn into(self) -> Shape {
        Shape::new(MAT,(self.0,self.1),self.2)
    }
}

impl Into<Shape> for (i32,i32) {
    fn into(self) -> Shape {
        Shape::new(MAT,self,false)
    }
}

impl Into<Shape> for i32 {
    fn into(self) -> Shape {
        Shape::new(VEC,(self,0),false)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        match self.degree {
            SCA => write!(f,"()"),
            VEC => write!(f,"({})",self.dim.0),
            MAT => {
                if self.T {
                    return write!(f,"{:?}.T",self.dim);
                } else {
                    return write!(f,"{:?}",self.dim);
                };
            }
        }
    }
}

