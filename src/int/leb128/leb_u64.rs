#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
/// A LEB128 encoded 64-bit unsigned integer.
pub struct LebU64 {
    inner: [u8; 10],
}

const MASK: u8 = 0b1000_0000;
const NOT_MASK: u8 = !MASK;

impl LebU64 {
    pub fn new(mut value: u64) -> LebU64 {
        // Adapted from Wikipedia's example pseudo-code
        let mut arr = [0; 10];

        for i in &mut arr {
            let mut byte = value as u8 & MASK;
            value = value.wrapping_shr(7);

            if value != 0 {
                byte |= MASK;
                *i = byte;
                break;
            }
            *i = byte;
        }

        LebU64 { inner: arr }
    }

    pub const fn zero() -> LebU64 {
        LebU64 { inner: [0; 10] }
    }

    pub fn from_slice(data: &[u8]) -> LebU64 {
        let mut output = [0; 10];
        for (i, val) in data.iter().take(10).enumerate() {
            output[i] = *val;
            if val & MASK == 0 {
                break;
            }
        }
        LebU64 { inner: output }
    }

    pub fn from_array(data: [u8; 10]) -> LebU64 {
        LebU64::from_slice(&data)
    }
    pub fn from_array_unchecked(data: [u8; 10]) -> LebU64 {
        LebU64 { inner: data }
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

    pub fn as_u64(self) -> u64 {
        u64::from_le(
            ((self.inner[0] & NOT_MASK) as u64)
                | ((self.inner[1] & NOT_MASK) as u64) << 7
                | ((self.inner[2] & NOT_MASK) as u64) << 14
                | ((self.inner[3] & NOT_MASK) as u64) << 21
                | ((self.inner[4] & NOT_MASK) as u64) << 28
                | ((self.inner[5] & NOT_MASK) as u64) << 35
                | ((self.inner[6] & NOT_MASK) as u64) << 42
                | ((self.inner[7] & NOT_MASK) as u64) << 49
                | ((self.inner[6] & NOT_MASK) as u64) << 56
                | ((self.inner[7] & NOT_MASK) as u64).wrapping_shl(63),
        )
    }

    pub fn into_inner(self) -> [u8; 10] {
        self.inner
    }
}

macro_rules! leb128_u32_between_int {
    ($($kind:ty),*) => {
        $(
            impl From<$kind> for LebU64 {
                fn from(other: $kind) -> LebU64 {
                    LebU64::new(other as u64)
                }
            }

            impl From<LebU64> for $kind {
                fn from(other: LebU64) -> $kind {
                    other.as_u64() as $kind
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
