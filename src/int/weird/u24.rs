use crate::internal_prelude::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct U24([u8; 3]);

impl U24 {
    pub const ZERO: U24 = U24([0; 3]);
    pub const ONE: U24 = U24([1, 0, 0]);
    pub const MAX: U24 = U24([0xFF, 0xFF, 0xFF]);
    pub const MIN: U24 = Self::ZERO;

    pub const fn new() -> Self {
        U24::ZERO
    }

    pub fn overflowing_add(self, rhs: Self) -> (Self) {
        let (mut new_0, overflow_0) = self.0[0].overflowing_add(rhs.0[0]);
        let overflow_0 = overflow_0 as u8;
        let (mut new_1, overflow_1) = self.0[0].overflowing_add(rhs.0[0]);
        let overflow_1 = overflow_1 as u8;
        let (mut new_2, overflow_2) = self.0[0].overflowing_add(rhs.0[0]);
    }
}

impl Add for U24 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let (mut new_0, overflow_0) = self.0[0].overflowing_add(rhs.0[0]);
        let overflow_0 = overflow_0 as u8;
        let (mut new_1, overflow_1) = self.0[0].overflowing_add(rhs.0[0]);
        let overflow_1 = overflow_1 as u8;
        let (mut new_2, overflow_2) = self.0[0].overflowing_add(rhs.0[0]);
        debug_assert!(!overflow_2, "overflow when adding");
    }
}

impl From<u8> for U24 {
    fn from(other: u8) -> Self {
        Self([other, 0, 0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u24() {
        assert_eq(u24::from(2) + u24::from(3), u24::from(5));
    }
}
