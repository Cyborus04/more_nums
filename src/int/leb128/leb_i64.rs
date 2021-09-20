#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
/// A LEB128 encoded 64-bit signed integer.
pub struct LebI64 {
    inner: [u8; 10],
}

const MASK: u8 = 0b1000_0000;
const SIGN_MASK: u8 = 0b0100_0000;

impl LebI64 {
    pub fn new(mut value: i64) -> LebI64 {
        // Adapted from Wikipedia's example pseudo-code
        let mut arr = [0; 10];

        for i in &mut arr {
            let mut byte = value as u8 & MASK;
            value = value.wrapping_shr(7);

            if (value == 0 && byte & MASK == 0) || (value == -1 && byte & SIGN_MASK == 0) {
                *i = byte;
                break;
            } else {
                byte |= MASK;
            }
            *i = byte;
        }

        LebI64 { inner: arr }
    }

    pub const fn zero() -> LebI64 {
        LebI64 { inner: [0; 10] }
    }

    pub fn from_slice(data: &[u8]) -> LebI64 {
        let mut output = [0; 10];
        for (i, val) in data.iter().take(10).enumerate() {
            output[i] = *val;
            if val & MASK == 0 {
                break;
            }
        }
        LebI64 { inner: output }
    }

    pub fn from_array(data: [u8; 10]) -> LebI64 {
        LebI64::from_slice(&data)
    }
    pub fn from_array_unchecked(data: [u8; 10]) -> LebI64 {
        LebI64 { inner: data }
    }

    pub fn as_slice(&self) -> &[u8] {
        //let mut output = Vec::with_capacity(4);
        for (i, val) in self.inner.iter().enumerate() {
            if val & MASK == 0 {
                return &self.inner[0..=i];
            }
        }
        &self.inner // No need to panic!("U32::as_slice reached end of `inner` without exiting!");
    }

    pub fn repair(&mut self) {
        *self = Self::from_slice(self.as_slice());
    }

    pub fn as_i64(self) -> i64 {
        // Adapted from Wikipedia's example Javascript code
        let mut result = 0;
        let mut shift = 0;
        for &byte in &self.inner {
            result |= ((byte & 0x7f) as i64) << shift;
            shift += 7;
            if byte & MASK == 0 {
                if shift < 64 && byte & SIGN_MASK != 0 {
                    return result | (!0 << shift);
                }
                return result;
            }
        }
        result
    }

    pub fn into_inner(self) -> [u8; 10] {
        self.inner
    }
}

macro_rules! leb128_i64_between_int {
    ($($kind:ty),*) => {
        $(
            impl From<$kind> for LebI64 {
                fn from(other: $kind) -> LebI64 {
                    LebI64::new(other as i64)
                }
            }

            impl From<LebI64> for $kind {
                fn from(other: LebI64) -> $kind {
                    other.as_i64() as $kind
                }
            }
        )*
    };
}

leb128_i64_between_int! {
    u8,    i8,
    u16,   i16,
    u32,   i32,
    u64,   i64,
    u128,  i128,
    usize, isize
}

/* TODO:

use std::ops::{
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign,
}

impl Add


*/
