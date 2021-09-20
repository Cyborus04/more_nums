use super::*;

macro_rules! with_uint_ops_impl {
	($self:ident; $($other:ty),*) => {
		$(
			impl From<$other> for $self {
				fn from(other : $other) -> Self {
					let mut out = Self::zero();
					out.inner[0] = other as u128;
					out
				}
			}

			impl TryFrom<$self> for $other {
				type Error = $self;
				fn try_from(other : $self) -> Result<Self, Self::Error> {
					if other > <$other>::max_value().into() {
						return Err(other);
					}
					Ok(other.inner[0] as $other)
				}
			}

			impl Add<$other> for $self {
				type Output = Self;
				fn add(self, rhs : $other) -> Self {
					let (value, overflowed) = self.overflowing_add(rhs.into());
					debug_assert!(!overflowed, "Overflow when adding!");
					value
				}
			}
		)*

	};
}

macro_rules! into_int {
	($($num:ty, $func:ident);*) => {
		$(
			pub fn $func(self) -> $num {
				self.inner[0] as $num
			}
		)*
	};
}

macro_rules! with_int_ops_impl {
	($self:ident; $($other:ty),*) => {
		$(
			impl From<$other> for $self {
				fn from(other : $other) -> Self {
					if other >= 0 {
						(other as u128).into()
					} else {
						let filled = Self::max_value() ^ <$other>::max_value().into();
						Self::from(other as u32) | filled
					}
				}
			}

			impl Add<$other> for $self {
				type Output = Self;
				fn add(self, rhs : $other) -> Self {
					let (value, overflowed) = self.overflowing_add(rhs.into());
					debug_assert!(!overflowed, "Overflow when adding!");
					value
				}
			}
		)*

	};
}

macro_rules! structs {
	($($name:ident, $size:expr);*) => {
		$(
			#[derive(Copy, Clone, Debug, PartialEq, Eq)]
			pub struct $name {
				pub(crate) inner : [u128; $size]
			}

			impl Default for $name {
				fn default() -> Self {
					Self::MIN
				}
			}

			impl $name {
				pub const MIN: Self = Self::zero();
				pub const MAX: Self = Self::max_value();

				pub const fn min_value() -> Self {
					Self { inner : [0; $size] }
				}

				pub const fn zero() -> Self {
					Self { inner : [0; $size] }
				}

				pub const fn one() -> Self {
					let mut inner = [0; $size];

					inner[0] = 1;

					Self { inner }
				}

				pub const fn max_value() -> Self {
					Self { inner : [u128::max_value(); $size] }
				}

				pub const fn from_array(inner : [u128; $size]) -> Self {
					Self { inner }
				}

				pub const fn into_inner(self) -> [u128; $size] { self.inner }

				pub fn to_le_bytes(self) -> [u8; $size * 16] {
					let mut bytes = [0; $size * 16];
					for (val, chunk) in self.inner.iter().zip(bytes.chunks_mut(16)) {
						let val = val.to_le_bytes();
						chunk.copy_from_slice(&val);
					}
					bytes
				}

				pub fn to_be_bytes(self) -> [u8; $size * 16] {
					let mut bytes = [0; $size * 16];
					for (val, chunk) in self.inner.iter().rev().zip(bytes.chunks_mut(16)) {
						let val = val.to_be_bytes();
						chunk.copy_from_slice(&val);
					}
					bytes
				}

				pub fn from_le_bytes(bytes : [u8; $size * 16]) -> Self {
					use core::convert::TryInto;

					let mut u128_array = [0; $size];

					for (bytes, num) in bytes.chunks(16).zip(u128_array.iter_mut()) {
						*num = u128::from_le_bytes(bytes.try_into().unwrap())
					}

					Self { inner : u128_array }
				}

				pub fn count_ones(self) -> u32 {
					self.inner.iter().copied().map(|x| x.count_ones()).sum()
				}

				pub fn count_zeros(self) -> u32 {
					($size * 128) - self.count_ones()
				}

				pub fn overflowing_add(self, rhs: Self) -> (Self, bool) {
					let mut carry = false;
					let mut output = [0; $size];

					for (i, (&left, &right)) in self.inner.iter().zip(rhs.inner.iter()).enumerate() {
						let (value, new_carry_1) = left.overflowing_add(right);
						let (value, new_carry_2) = value.overflowing_add(carry as u128);
						// Converting bools to integers maps `true` to `1` and `false` to `0`

						carry = new_carry_1 || new_carry_2;
						output[i] = value;
					}

					( Self { inner : output }, carry )
				}

				pub fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
					(self.wrapping_add((!rhs).wrapping_add(Self::one())), rhs > self)
				}

				pub fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
					let big = self.big_mul(rhs);
					(big.1, big.0 != Self::zero())
				}

				pub fn overflowing_pow(self, rhs: u32) -> (Self, bool) {
					let (upper, lower) = self.big_pow(rhs);
					(lower, upper != Self::zero())
				}

				pub fn overflowing_div(self, rhs: Self) -> (Self, bool) {
					(self.div_and_rem(rhs).0, false)
				}

				pub fn overflowing_rem(self, rhs: Self) -> (Self, bool) {
					// Thank you wikipedia. https://www.wikipedia.org/wiki/Division_algorithm

					let mut r = Self::zero();

					for i in (0..(($size * 128))).rev() {
						r <<= 1;
						r |= self.bit_at(i).into();
						if r >= rhs {
							r -= rhs;
						}
					}

					(r, false)
				}

				pub fn overflowing_shl(self, rhs: u32) -> (Self, bool) {
					let mut output = [0; $size];

					let u128s_shifted = (rhs / 128) as usize;
					let bits_shifted = (rhs % 128);

					if u128s_shifted > 0 {
						for i in (u128s_shifted..self.inner.len()).rev() {
							output[i] = self.inner[i - u128s_shifted];
						}
					} else {
						output = self.inner;
					}

					let mut last_extra_bits = 0;

					if bits_shifted > 0 {
						for val in output.iter_mut() {
							let shifted = (*val << bits_shifted) | last_extra_bits;
							last_extra_bits = val.wrapping_shr(128 - bits_shifted);
							*val = shifted;
						}
					}

					(Self { inner : output }, last_extra_bits != 0)
				}

				pub fn overflowing_shr(self, rhs: u32) -> (Self, bool) {
					let mut output = [0; $size];

					let u128s_shifted = (rhs / 128) as usize;
					let bits_shifted = (rhs % 128);

					for i in 0..self.inner.len() - u128s_shifted {
						output[i] = self.inner[i + u128s_shifted];
					}

					let mut last_extra_bits = 0;

					for val in output.iter_mut().rev() {
						let shifted = (*val >> bits_shifted) | last_extra_bits;
						last_extra_bits = val.wrapping_shl(128 - bits_shifted);
						*val = shifted;
					}

					(Self { inner : output }, last_extra_bits != 0)
				}

				pub fn checked_add(self, rhs: Self) -> Option<Self> {
					match self.overflowing_add(rhs) {
						(x, true) => Some(x),
						(_, false) => None,
					}
				}

				pub fn checked_sub(self, rhs: Self) -> Option<Self> {
					match self.overflowing_sub(rhs) {
						(x, true) => Some(x),
						(_, false) => None,
					}
				}

				pub fn checked_mul(self, rhs: Self) -> Option<Self> {
					match self.overflowing_mul(rhs) {
						(x, true) => Some(x),
						(_, false) => None,
					}
				}

				pub fn checked_pow(self, rhs: u32) -> Option<Self> {
					match self.overflowing_pow(rhs) {
						(x, true) => Some(x),
						(_, false) => None,
					}
				}

				pub fn checked_div(self, rhs: Self) -> Option<Self> {
					match self.overflowing_div(rhs) {
						(x, true) => Some(x),
						(_, false) => None,
					}
				}

				pub fn checked_rem(self, rhs: Self) -> Option<Self> {
					match self.overflowing_div(rhs) {
						(x, true) => Some(x),
						(_, false) => None,
					}
				}

				pub fn checked_shl(self, rhs: u32) -> Option<Self> {
					match self.overflowing_shl(rhs) {
						(x, true) => Some(x),
						(_, false) => None,
					}
				}

				pub fn checked_shr(self, rhs: u32) -> Option<Self> {
					match self.overflowing_shr(rhs) {
						(x, true) => Some(x),
						(_, false) => None,
					}
				}

				pub fn wrapping_add(self, rhs: Self) -> Self {
					self.overflowing_add(rhs).0
				}

				pub fn wrapping_sub(self, rhs: Self) -> Self {
					self.overflowing_sub(rhs).0
				}

				pub fn wrapping_mul(self, rhs: Self) -> Self {
					self.overflowing_mul(rhs).0
				}

				pub fn wrapping_pow(self, rhs: u32) -> Self {
					self.overflowing_pow(rhs).0
				}

				pub fn wrapping_div(self, rhs: Self) -> Self {
					self.overflowing_div(rhs).0
				}

				pub fn wrapping_rem(self, rhs: Self) -> Self {
					self.overflowing_rem(rhs).0
				}

				pub fn wrapping_shr(self, rhs : u32) -> Self {
					self.overflowing_shr(rhs).0
				}

				pub fn wrapping_shl(self, rhs : u32) -> Self {
					self.overflowing_shl(rhs).0
				}

				pub fn pow(self, rhs: u32) -> Self {
					let (upper, lower) = self.big_pow(rhs);
					assert_eq!(upper, Self::zero(), "Overflow during exponentiation!");
					lower
				}





				pub fn big_mul(self, mut rhs : Self) -> (Self, Self) {
					let mut result = (Self::zero(), Self::zero());

					let mut a = (Self::zero(), self);

					// Adapted from Zengr's C code at https://stackoverflow.com/questions/4456442
					while rhs != Self::zero() {
						if rhs & 1u8.into() == 1u8.into()  {
							result = Self::add_big(result,a);
						}
						a = Self::shift_left_one_big(a.0, a.1);
						rhs >>= 1;
					}

					result
				}

				fn big_mul_big(this : (Self, Self), mut that: (Self, Self)) -> (Self, Self) {
					let mut result = (Self::zero(), Self::zero());

					let mut a = this;

					// Adapted from Zengr's C code at https://stackoverflow.com/questions/4456442
					while that != (Self::zero(), Self::zero()) {
						if that.1 & Self::one() == Self::one() && that.0 == Self::zero() {
							result = Self::add_big(result,a);
						}
						a = Self::shift_left_one_big(a.0, a.1);
						that = Self::shift_right_one_big(that.0, that.1);
					}

					result
				}

				fn shift_left_one_big(big : Self, small : Self) -> (Self, Self) {
					let (small_shifted, small_overflow) = small.overflowing_shl(1);
					let (big_shifted, _) = big.overflowing_shl(1);
					(big_shifted | (small_overflow as u8).into(), small_shifted)
				}

				fn shift_right_one_big(big : Self, small : Self) -> (Self, Self) {
					let (small_shifted, _) = small.overflowing_shr(1);
					let (big_shifted, big_overflow) = big.overflowing_shr(1);
					(big_shifted, small_shifted | ((big_overflow as u128) << 127).into())
				}

				fn add_big(this : (Self, Self), that : (Self, Self)) -> (Self, Self) {
					let (small_added, small_overflow) = this.1.overflowing_add(that.1);
					let (big_shifted, _) = this.0.overflowing_add(that.0);

					(big_shifted.overflowing_add((small_overflow as u8).into()).0, small_added)
				}

				pub fn div_and_rem(self, rhs : Self) -> (Self, Self) {
					// Thank you wikipedia. https://www.wikipedia.org/wiki/Division_algorithm
					let mut q = Self::zero();
					let mut r = Self::zero();

					for i in (0..(($size * 128))).rev() {
						r <<= 1;
						r |= self.bit_at(i).into();
						if r >= rhs {
							r -= rhs;
							q |= Self::from(1) << i;
						}
					}

					(q, r)
				}

				pub fn leading_zeros(self) -> u32 {
					for (i, &val) in self.inner.iter().rev().enumerate() {
						if val == 0 {
							continue;
						} else {
							return (i as u32 * 128) + val.leading_zeros();
						}
					}
					128 * $size
				}

				pub fn set_bit(&mut self, index : u32, value : bool) {
					if index < $size * 128 {
						let self_at = u8::try_from(self.wrapping_shr(index) & Self::from(1)).unwrap();
						if self_at == value as u8 {
							return; // If it's already what we want, don't do anything!
						}
						if self_at == 1 {
							*self &= !(Self::from(1) << index)
						} else {
							*self |= Self::from(1) << index
						}
					}
				}

				pub fn big_pow(self, rhs: u32) -> (Self, Self) {
					Self::big_pow_big((Self::zero(), self), rhs)
				}

				fn big_pow_big(this : (Self, Self), mut rhs: u32) -> (Self, Self) {
					// Adapted from the std library's implementation of pow for `u128`
					let mut base = this;
					let mut acc = (Self::zero(), Self::one());

					while rhs > 1 {
						if (rhs & 1) == 1 {
							acc = Self::big_mul_big(acc, base);
						}
						rhs >>= 1;
						base = Self::big_mul_big(base, base);
					}

					if rhs == 1 {
						acc = Self::big_mul_big(acc, base);
					}

					acc
				}

				pub fn mod_pow(self, mut rhs : Self, modulus : Self) -> Self {
					if modulus == Self::one() {
						return Self::zero();
					}
					let mut output = Self::one();
					let mut base = self % modulus;
					while rhs > Self::zero() {
						if (rhs & Self::one() == Self::one()) {
							output = (output * base) % modulus
						}
						rhs >>= 1;
						base = (base * base) % modulus;
					}
					output
				}

				pub fn bit_at(self, index : u32) -> u8 {
					debug_assert!(index < ($size * 128), "bit index out of range");

					(self.inner[(index.wrapping_shr(7)) as usize].wrapping_shr(index & 127) & 1) as u8
				}

				into_int!{
					u8, into_u8;
					u16, into_u16;
					u32, into_u32;
					u64, into_u64;
					u128, into_u128;
					usize, into_usize
				}
			}

			impl From<[u128; $size]> for $name {
				fn from(other : [u128; $size]) -> $name {
					Self { inner : other }
				}
			}

			impl PartialOrd for $name {
				fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
					Some(self.cmp(other))
				}
			}

			impl Ord for $name {
				fn cmp(&self, other : &Self) -> Ordering {
					for (left, right) in self.inner.iter().zip(other.inner.iter()).rev() {
						match left.cmp(right) {
							Ordering::Equal => continue,
							x => return x,
						}
					}

					Ordering::Equal
				}
			}

			impl BitAnd for $name {
				type Output = Self;
				fn bitand(self, rhs : Self) -> Self {
					let mut output = [0; $size];
					for (i, (&a, &b)) in self.inner.iter().zip(rhs.inner.iter()).enumerate() {
						output[i] = a & b;
					}
					Self { inner : output }
				}
			}

			impl BitOr for $name {
				type Output = Self;
				fn bitor(self, rhs : Self) -> Self {
					let mut output = [0; $size];
					for (i, (&a, &b)) in self.inner.iter().zip(rhs.inner.iter()).enumerate() {
						output[i] = a | b;
					}
					Self { inner : output }
				}
			}

			impl BitXor for $name {
				type Output = Self;
				fn bitxor(self, rhs : Self) -> Self {
					let mut output = [0; $size];
					for (i, (&a, &b)) in self.inner.iter().zip(rhs.inner.iter()).enumerate() {
						output[i] = a ^ b;
					}
					Self { inner : output }
				}
			}

			impl BitAndAssign for $name {
				fn bitand_assign(&mut self, rhs : Self) {
					for (a, &b) in self.inner.iter_mut().zip(rhs.inner.iter()) {
						*a &= b;
					}
				}
			}

			impl BitOrAssign for $name {
				fn bitor_assign(&mut self, rhs : Self) {
					for (a, &b) in self.inner.iter_mut().zip(rhs.inner.iter()) {
						*a |= b;
					}
				}
			}

			impl BitXorAssign for $name {
				fn bitxor_assign(&mut self, rhs : Self) {
					for (a, &b) in self.inner.iter_mut().zip(rhs.inner.iter()) {
						*a ^= b;
					}
				}
			}

			impl Not for $name {
				type Output = Self;
				fn not(self) -> Self {
					let mut output = [0; $size];
					for (i, a) in self.inner.iter().enumerate() {
						output[i] = !a;
					}
					Self { inner : output }
				}
			}

			impl Add for $name {
				type Output = Self;
				fn add(self, rhs : Self) -> Self {
					let (value, overflowed) = self.overflowing_add(rhs);
					debug_assert!(!overflowed, "Overflow when adding!");
					value
				}
			}

			impl AddAssign for $name {
				fn add_assign(&mut self, rhs : Self) {
					let (value, overflowed) = self.overflowing_add(rhs);
					debug_assert!(!overflowed, "Overflow when adding!");
					*self = value;
				}
			}

			impl Mul for $name {
				type Output = Self;
				fn mul(self, rhs : Self) -> Self {
					let (value, overflowed) = self.overflowing_mul(rhs);
					debug_assert!(!overflowed, "Overflow when multiplying!");
					value
				}
			}

			impl MulAssign for $name {
				fn mul_assign(&mut self, rhs : Self) {
					let (value, overflowed) = self.overflowing_mul(rhs);
					debug_assert!(!overflowed, "Overflow when multiplying!");
					*self = value;
				}
			}

			impl Sub for $name {
				type Output = Self;
				fn sub(self, rhs : Self) -> Self {
					let (value, overflowed) = self.overflowing_sub(rhs);
					debug_assert!(!overflowed, "Overflow when subtracting!");
					value
				}
			}

			impl SubAssign for $name {
				fn sub_assign(&mut self, rhs : Self) {
					let (value, overflowed) = self.overflowing_sub(rhs);
					debug_assert!(!overflowed, "Overflow when subtracting!");
					*self = value;
				}
			}

			impl Div for $name {
				type Output = Self;
				fn div(self, rhs : Self) -> Self {
					self.overflowing_div(rhs).0
				}
			}

			impl DivAssign for $name {
				fn div_assign(&mut self, rhs : Self) {
					*self = self.overflowing_div(rhs).0
				}
			}

			impl Rem for $name {
				type Output = Self;
				fn rem(self, rhs : Self) -> Self::Output {
					self.overflowing_rem(rhs).0
				}
			}

			impl RemAssign for $name {
				fn rem_assign(&mut self, rhs : Self) {
					*self = self.overflowing_rem(rhs).0;
				}
			}

			impl Shl<u32> for $name {
				type Output = Self;
				fn shl(self, rhs : u32) -> Self::Output {
					self.overflowing_shl(rhs).0
				}
			}

			impl ShlAssign<u32> for $name {
				fn shl_assign(&mut self, rhs : u32) {
					*self = self.overflowing_shl(rhs).0
				}
			}

			impl Shr<u32> for $name {
				type Output = Self;
				fn shr(self, rhs : u32) -> Self::Output {
					self.overflowing_shr(rhs).0
				}
			}

			impl ShrAssign<u32> for $name {
				fn shr_assign(&mut self, rhs : u32) {
					*self = self.overflowing_shr(rhs).0
				}
			}

			impl Sum for $name {
				fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
					iter.fold(Self::zero(), Add::add)
				}
			}

			impl Product for $name {
				fn product<I: Iterator<Item=Self>>(iter: I) -> Self {
					iter.fold(Self::one(), Mul::mul)
				}
			}

			impl UpperHex for $name {
				fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
					for i in self.inner.iter() {
						write!(fmt, "{:X}", i)?;
					}
					Ok(())
				}
			}

			impl LowerHex for $name {
				fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
					for i in self.inner.iter() {
						write!(fmt, "{:x}", i)?;
					}
					Ok(())
				}
			}

			impl Octal for $name {
				fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
					for i in self.inner.iter() {
						write!(fmt, "{:x}", i)?;
					}
					Ok(())
				}
			}


			with_uint_ops_impl!($name; u8, u16, u32, u64, u128, usize);

			with_int_ops_impl!($name; i8, i16, i32, i64, i128, isize);
		)*
	};
}

// Can only go to 32 until const generic array defaults are stabilized.
// (which is never :P (jk))
structs! {
    U256, 2;
    U512, 4;
    U1024, 8;
    U2048, 16;
    U4096, 32;
    U8192, 64;
    U16384, 128;
    U32768, 256;
    U65536, 512
}

impl From<U256> for U512 {
    fn from(other: U256) -> Self {
        let mut new_array = [0; 4];
        (&mut new_array[0..2]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<U256> for U1024 {
    fn from(other: U256) -> Self {
        let mut new_array = [0; 8];
        (&mut new_array[0..2]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<U512> for U1024 {
    fn from(other: U512) -> Self {
        let mut new_array = [0; 8];
        (&mut new_array[0..4]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<U256> for U2048 {
    fn from(other: U256) -> Self {
        let mut new_array = [0; 16];
        (&mut new_array[0..2]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<U512> for U2048 {
    fn from(other: U512) -> Self {
        let mut new_array = [0; 16];
        (&mut new_array[0..4]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<U1024> for U2048 {
    fn from(other: U1024) -> Self {
        let mut new_array = [0; 16];
        (&mut new_array[0..8]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<U256> for U4096 {
    fn from(other: U256) -> Self {
        let mut new_array = [0; 32];
        (&mut new_array[0..2]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<U512> for U4096 {
    fn from(other: U512) -> Self {
        let mut new_array = [0; 32];
        (&mut new_array[0..4]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<U1024> for U4096 {
    fn from(other: U1024) -> Self {
        let mut new_array = [0; 32];
        (&mut new_array[0..8]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<U2048> for U4096 {
    fn from(other: U2048) -> Self {
        let mut new_array = [0; 32];
        (&mut new_array[0..16]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<(U256, U256)> for U512 {
    fn from((other_big, other_small): (U256, U256)) -> Self {
        let mut new_array = [0; 4];
        (&mut new_array[0..2]).copy_from_slice(&other_small.inner);
        (&mut new_array[2..4]).copy_from_slice(&other_big.inner);
        Self { inner: new_array }
    }
}

impl From<(U512, U512)> for U1024 {
    fn from((other_big, other_small): (U512, U512)) -> Self {
        let mut new_array = [0; 8];
        (&mut new_array[0..4]).copy_from_slice(&other_small.inner);
        (&mut new_array[4..8]).copy_from_slice(&other_big.inner);
        Self { inner: new_array }
    }
}

impl From<(U1024, U1024)> for U2048 {
    fn from((other_big, other_small): (U1024, U1024)) -> Self {
        let mut new_array = [0; 16];
        (&mut new_array[0..8]).copy_from_slice(&other_small.inner);
        (&mut new_array[8..16]).copy_from_slice(&other_big.inner);
        Self { inner: new_array }
    }
}

impl From<(U2048, U2048)> for U4096 {
    fn from((other_big, other_small): (U2048, U2048)) -> Self {
        let mut new_array = [0; 32];
        (&mut new_array[0..16]).copy_from_slice(&other_small.inner);
        (&mut new_array[16..32]).copy_from_slice(&other_big.inner);
        Self { inner: new_array }
    }
}

// macro_rules! casting_from {
// 	($($big:ident, $small:ident);*) => {
// 		$(
// 		impl From<$big> for $small {
// 			fn from(other: $big) -> Self {
// 				let mut output = Self::zero();
// 				let value = (other & $small::max_value().into());
// 				let len = output.inner.len();
// 				(&mut output.inner[..]).copy_from_slice(&value.inner[..len]);
// 				output
// 			}
// 		}

// 		impl From<$small> for $big {
// 			fn from(other: $small) -> Self {
// 				let mut output = Self::zero();
// 				let len = other.inner.len();
// 				(&mut output.inner[..len]).copy_from_slice(&other.inner[..]);
// 				output
// 			}
// 		}
// 		)*
// 	}
// }
// Self::from_array(other.into_inner())
// casting_from! {
//     U512, U256;
//     U1024, U256;
//     U1024, U512;
//     U2048, U256;
//     U2048, U512;
//     U2048, U1024;
//     U4096, U256;
//     U4096, U512;
//     U4096, U1024;
//     U4096, U2048
// }

impl From<I256> for U256 {
    fn from(other: I256) -> Self {
        Self::from_array(other.into_inner())
    }
}

impl From<I512> for U512 {
    fn from(other: I512) -> Self {
        Self::from_array(other.into_inner())
    }
}

impl From<I1024> for U1024 {
    fn from(other: I1024) -> Self {
        Self::from_array(other.into_inner())
    }
}

impl From<I2048> for U2048 {
    fn from(other: I2048) -> Self {
        Self::from_array(other.into_inner())
    }
}

impl From<I4096> for U4096 {
    fn from(other: I4096) -> Self {
        Self::from_array(other.into_inner())
    }
}

impl U256 {
    pub fn to_signed(self) -> I256 {
        I256 { inner: self.inner }
    }
}

impl U512 {
    pub fn to_signed(self) -> I512 {
        I512 { inner: self.inner }
    }
}

impl U1024 {
    pub fn to_signed(self) -> I1024 {
        I1024 { inner: self.inner }
    }
}

impl U2048 {
    pub fn to_signed(self) -> I2048 {
        I2048 { inner: self.inner }
    }
}

impl U4096 {
    pub fn to_signed(self) -> I4096 {
        I4096 { inner: self.inner }
    }
}

impl From<U512> for U256 {
    fn from(other: U512) -> U256 {
        U256::from_array(<[u128; 2]>::try_from(&other.into_inner()[0..2]).unwrap())
    }
}

impl From<U1024> for U256 {
    fn from(other: U1024) -> U256 {
        U256::from_array(<[u128; 2]>::try_from(&other.into_inner()[0..2]).unwrap())
    }
}

impl From<U2048> for U256 {
    fn from(other: U2048) -> U256 {
        U256::from_array(<[u128; 2]>::try_from(&other.into_inner()[0..2]).unwrap())
    }
}

impl From<U4096> for U256 {
    fn from(other: U4096) -> U256 {
        U256::from_array(<[u128; 2]>::try_from(&other.into_inner()[0..2]).unwrap())
    }
}

impl From<U1024> for U512 {
    fn from(other: U1024) -> U512 {
        U512::from_array(<[u128; 4]>::try_from(&other.into_inner()[0..4]).unwrap())
    }
}

impl From<U2048> for U512 {
    fn from(other: U2048) -> U512 {
        U512::from_array(<[u128; 4]>::try_from(&other.into_inner()[0..4]).unwrap())
    }
}

impl From<U4096> for U512 {
    fn from(other: U4096) -> U512 {
        U512::from_array(<[u128; 4]>::try_from(&other.into_inner()[0..4]).unwrap())
    }
}

impl From<U2048> for U1024 {
    fn from(other: U2048) -> U1024 {
        U1024::from_array(<[u128; 8]>::try_from(&other.into_inner()[0..8]).unwrap())
    }
}

impl From<U4096> for U1024 {
    fn from(other: U4096) -> U1024 {
        U1024::from_array(<[u128; 8]>::try_from(&other.into_inner()[0..8]).unwrap())
    }
}

impl From<U4096> for U2048 {
    fn from(other: U4096) -> U2048 {
        U2048::from_array(<[u128; 16]>::try_from(&other.into_inner()[0..16]).unwrap())
    }
}
