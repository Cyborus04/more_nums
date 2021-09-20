#![allow(clippy::unusual_byte_groupings)]

use crate::internal_prelude::*;
/// Under construction!
///
/// A 16-bit floating point number
///
/// Since this is implemented in software,
/// it is likely slower than if it were natively supported by the language
#[derive(Copy, Clone, Debug)]
pub struct F16(u16);

impl F16 {
    /// Creates a new `F16` with a value of 0.0
    pub const fn new() -> Self {
        Self(0)
    }

    /// Creates a new `F16` with a value of 0.0
    pub const fn zero() -> Self {
        Self::new()
    }

    pub const fn one() -> Self {
        Self(0b0_01111_0000000000)
    }

    pub const fn infinity() -> Self {
        Self(0b0_11111_0000000000)
    }

    pub const fn negative_infinity() -> Self {
        Self(0b1_11111_0000000000)
    }

    pub const fn nan() -> Self {
        Self(0b0_11111_0000000001)
    }

    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    pub const fn to_bits(self) -> u16 {
        self.0
    }

    pub fn to_be_bytes(self) -> [u8; 2] {
        self.to_bits().to_be_bytes()
    }

    pub fn to_le_bytes(self) -> [u8; 2] {
        self.to_bits().to_le_bytes()
    }

    pub fn to_ne_bytes(self) -> [u8; 2] {
        self.to_bits().to_ne_bytes()
    }

    pub fn from_be_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_be_bytes(bytes))
    }

    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    pub fn from_ne_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_ne_bytes(bytes))
    }

    pub const fn abs(self) -> Self {
        Self(self.0 & 0b0_11111_1111111111)
    }

    pub fn signum(self) -> Self {
        if self.is_nan() {
            self
        } else if self.sign() {
            -Self::one()
        } else {
            Self::one()
        }
    }

    pub fn is_infinite(self) -> bool {
        self.exponent_raw() == 0b11111 && self.mantissa_raw() == 0
    }

    pub fn is_nan(self) -> bool {
        self.exponent_raw() == 0b11111 && self.mantissa_raw() != 0
    }

    pub fn is_finite(self) -> bool {
        self.exponent_raw() != 0b11111
    }

    pub fn is_positive(self) -> bool {
        !self.sign()
    }

    pub fn is_negative(self) -> bool {
        self.sign()
    }

    // basically using `bool` as u1
    // true means negative,
    // false means positive
    fn sign(self) -> bool {
        self.0 & 0b1_00000_0000000000 != 0
    }

    fn exponent_raw(self) -> u8 {
        ((self.0 & 0b0_11111_0000000000) >> 10) as u8
    }

    fn exponent(self) -> i8 {
        self.exponent_raw() as i8 - 15
    }

    fn mantissa_raw(self) -> u16 {
        self.0 & 0b0_00000_1111111111
    }

    fn mantissa(self) -> u16 {
        if dbg!(dbg!(self.0 == 0) || dbg!(self.0 == 0b1_00000_0000000000)) {
            0
        } else {
            self.mantissa_raw() | 0b0000010000000000
        }
    }
}

impl PartialEq for F16 {
    fn eq(&self, other: &F16) -> bool {
        if self.is_nan() || other.is_nan() {
            false
        } else {
            self.0 == other.0
        }
    }
}

// impl Hash for F16 {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         state.write_u16(self.0)
//     }
// }

impl Neg for F16 {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0 ^ 0b1_00000_0000000000)
    }
}

impl Add for F16 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let offset = self.exponent() - other.exponent();
        let (self_mant, other_mant) = if offset > 0 {
            (
                self.mantissa(),
                other.mantissa().wrapping_shr(offset.abs() as u32),
            )
        } else {
            (
                self.mantissa().wrapping_shr(offset.abs() as u32),
                other.mantissa(),
            )
        };
        let mut new_mant = if self.is_negative() ^ other.is_negative() {
            self_mant - other_mant
        } else {
            self_mant + other_mant
        };
        let mut new_exp = self.exponent_raw().max(other.exponent_raw());
        let mut infinite = false;
        if new_mant.leading_zeros() < 5 {
            let shift = 5 - new_mant.leading_zeros();
            new_mant = new_mant.wrapping_shr(shift);
            let exp_result = new_exp.overflowing_add(shift as u8);
            new_exp = exp_result.0;
            infinite = exp_result.1;
        }
        let (new_mant, new_exp, infinite) = (new_mant, new_exp, infinite);

        if infinite {
            Self::infinity()
        } else {
            let bits = (((new_exp as u16) & 0b11111) << 10) | ((new_mant as u16) & 0b1111111111);
            Self(bits)
        }
    }
}

impl Sub for F16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}

impl core::fmt::Display for F16 {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(fmt, "1.{} * 2^{}", self.mantissa_raw(), self.exponent())
    }
}
