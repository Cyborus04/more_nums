#![allow(clippy::many_single_char_names)]

//! [LEB128](https://en.wikipedia.org/wiki/LEB128) Encoded Integers

mod leb_i32;
mod leb_i64;
mod leb_u32;
mod leb_u64;

pub use leb_i32::LebI32;
pub use leb_i64::LebI64;
pub use leb_u32::LebU32;
pub use leb_u64::LebU64;
