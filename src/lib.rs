//#![no_std]
//extern crate no_std_compat as std;

pub(crate) mod internal_prelude {
    pub use core::ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub,
        SubAssign,
    };
}
//use std::prelude::v1::*;

pub mod float;
pub mod int;

//#[cfg(test)]
//mod tests;
