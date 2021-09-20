#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
/// A LEB128 encoded 32-bit signed integer.
pub struct LebI32 {
    inner: [u8; 5],
}

const MASK: u8 = 0b1000_0000;
const SIGN_MASK: u8 = 0b0100_0000;

impl LebI32 {
    pub fn new(mut value: i32) -> LebI32 {
        // Adapted from Wikipedia's example pseudo-code
        let mut arr = [0; 5];

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

        LebI32 { inner: arr }
    }

    pub const fn zero() -> LebI32 {
        LebI32 { inner: [0; 5] }
    }

    pub fn from_slice(data: &[u8]) -> LebI32 {
        let mut output = [0; 5];
        for (i, val) in data.iter().take(5).enumerate() {
            output[i] = *val;
            if val & MASK == 0 {
                break;
            }
        }
        LebI32 { inner: output }
    }

    pub fn from_array(data: [u8; 5]) -> LebI32 {
        LebI32::from_slice(&data)
    }
    pub fn from_array_unchecked(data: [u8; 5]) -> LebI32 {
        LebI32 { inner: data }
    }

    pub fn as_slice(&self) -> &[u8] {
        for (i, val) in self.inner.iter().enumerate() {
            if val & MASK == 0 {
                return &self.inner[0..=i];
            }
        }
        &self.inner
    }

    pub fn repair(&mut self) {
        *self = Self::from_slice(self.as_slice());
    }

    pub fn as_i32(self) -> i32 {
        // Adapted from Wikipedia's example Javascript code
        let mut result = 0;
        let mut shift = 0;
        for &byte in &self.inner {
            result |= ((byte & 0x7f) as i32) << shift;
            shift += 7;
            if byte & MASK == 0 {
                if shift < 32 && byte & SIGN_MASK != 0 {
                    return result | (!0 << shift);
                }
                return result;
            }
        }
        result
    }

    pub fn into_inner(self) -> [u8; 5] {
        self.inner
    }
}

macro_rules! leb128_i32_between_int {
    ($($kind:ty),*) => {
        $(
            impl From<$kind> for LebI32 {
                fn from(other: $kind) -> LebI32 {
                    LebI32::new(other as i32)
                }
            }

            impl From<LebI32> for $kind {
                fn from(other: LebI32) -> $kind {
                    other.as_i32() as $kind
                }
            }
        )*
    };
}

leb128_i32_between_int! {
    u8,    i8,
    u16,   i16,
    u32,   i32,
    u64,   i64,
    u128,  i128,
    usize, isize
}
