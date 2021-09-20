#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
/// A LEB128 encoded 32-bit unsigned integer.
pub struct LebU32 {
    inner: [u8; 5],
}

const MASK: u8 = 0b1000_0000;
const NOT_MASK: u8 = !MASK;

impl LebU32 {
    pub fn new(mut value: u32) -> LebU32 {
        // Adapted from Wikipedia's example pseudo-code
        let mut arr = [0; 5];

        for i in &mut arr {
            *i = value as u8;
            value = value.wrapping_shr(7);
            if value != 0 {
                *i |= MASK;
            } else {
                break;
            }
        }

        LebU32 { inner: arr }
    }

    pub const fn zero() -> LebU32 {
        LebU32 { inner: [0; 5] }
    }

    pub fn from_slice(data: &[u8]) -> LebU32 {
        let mut output = [0; 5];
        for (i, val) in data.iter().take(5).enumerate() {
            output[i] = *val;
            if val & MASK == 0 {
                break;
            }
        }
        LebU32 { inner: output }
    }

    pub fn from_array(data: [u8; 5]) -> LebU32 {
        LebU32::from_slice(&data)
    }
    pub const fn from_array_unchecked(data: [u8; 5]) -> LebU32 {
        LebU32 { inner: data }
    }

    pub fn as_slice(&self) -> &[u8] {
        for (i, val) in self.inner.iter().enumerate() {
            if val & MASK == 0 {
                return &self.inner[0..=i];
            }
        }
        &self.inner
    }

    pub const fn as_u32(self) -> u32 {
        u32::from_le(
            ((self.inner[0] & NOT_MASK) as u32)
                | ((self.inner[1] & NOT_MASK) as u32) << 7
                | ((self.inner[2] & NOT_MASK) as u32) << 14
                | ((self.inner[3] & NOT_MASK) as u32) << 21
                | ((self.inner[4] & NOT_MASK) as u32).wrapping_shl(28), // this can go more than 32 bits
        )
    }

    pub const fn into_inner(self) -> [u8; 5] {
        self.inner
    }
}

macro_rules! leb128_u32_between_int {
    ($($kind:ty),*) => {
        $(
            impl From<$kind> for LebU32 {
                fn from(other: $kind) -> LebU32 {
                    LebU32::new(other as u32)
                }
            }

            impl From<LebU32> for $kind {
                fn from(other: LebU32) -> $kind {
                    other.as_u32() as $kind
                }
            }
        )*
    };
}

leb128_u32_between_int! {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        assert_eq!(LebU32::from_array([128, 1, 0, 0, 0]), LebU32::new(128));
    }
}
