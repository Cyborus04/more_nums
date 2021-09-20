use super::*;

use core::ops::Neg;

macro_rules! into_int {
	($($num:ty, $func:ident);*) => {
		$(
			pub fn $func(self) -> $num {
				self.inner[0] as $num
			}
		)*
	};
}

macro_rules! with_uint_ops_impl {
	($self:ident; $($other:ty),*) => {
		$(
			impl From<$other> for $self {
				fn from(other : $other) -> Self {
					let mut out = Self::ZERO;
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

macro_rules! with_int_ops_impl {
	($self:ident; $($other:ty),*) => {
		$(
			impl From<$other> for $self {
				fn from(other : $other) -> Self {
					if other >= 0 {
						Self::from_u128(other as u128)
					} else {
						Self::from_i128(other as i128)
						// let mut arr =
						// let other = other as i128 as u128;
						// let filled = !(<$other>::max_value().into());
						// Self::from(other as u32) | filled
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
					Self::ZERO
				}
			}

			impl $name {
				pub const MIN: Self = Self::min_value();
				pub const ZERO: Self = Self::zero();
				pub const ONE: Self = Self::one();
				pub const MAX: Self = Self::min_value();

				pub const fn min_value() -> Self {
					let mut inner = [u128::MAX; $size];
					inner[$size - 1] = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
					Self { inner }
				}

				pub const fn zero() -> Self {
					Self { inner : [0; $size] }
				}

				pub const fn one() -> Self {
					Self::from_u128(1)
				}

				pub const fn max_value() -> Self {
					let mut inner = [0; $size];
					inner[$size - 1] = 0x80000000000000000000000000000000;
					Self { inner }
				}

				const fn from_u128(v: u128) -> Self {
					let mut inner = [0; $size];

					inner[0] = v;

					Self { inner }
				}

				const fn from_i128(v: i128) -> Self {
					let mut inner = if v < 0 { [u128::MAX; $size] } else { [0; $size] };

					inner[0] = v as u128;

					Self { inner }
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

				pub fn overflowing_add(self, rhs : Self) -> (Self, bool) {
					let mut carry = false;
					let mut output = [0; $size];

					for (i, (&left, &right)) in self.inner.iter().zip(rhs.inner.iter()).enumerate() {
						let (value, new_carry_1) = left.overflowing_add(right);
						let (value, new_carry_2) = value.overflowing_add(carry as u128);
						// Converting bools to integers maps `true` to `1` and `false` to `0`

						carry = new_carry_1 || new_carry_2;
						output[i] = value;
					}

					let output = Self { inner : output };
					let overflowed = if self.is_same_sign(rhs) { // If they're different signs, it can't overflow.
						if self.is_neg() && rhs.is_neg() { // If they're both negative, it'll have overflow if the output is positive.
							!output.is_neg()
						} else { // If they're both positive, it'll have overflowed if the outputer is negative.
							output.is_neg()
						}
					} else {
						false
					};
					( output, overflowed )
				}

				pub fn overflowing_sub(self, rhs : Self) -> (Self, bool) {
					let value = self.wrapping_add(-rhs);
					(
						value,
						if !self.is_same_sign(rhs) {
							if self.is_neg() && !rhs.is_neg() {
								!value.is_neg()
							} else {
								value.is_neg()
							}
						} else {
							false
						}
					)
				}

				pub fn overflowing_mul(self, rhs : Self) -> (Self, bool) {
					let big = self.big_mul(rhs);
					(big.1, false)
				}

				pub fn overflowing_pow(self, rhs: u32) -> (Self, bool) {
					let (upper, lower) = self.big_pow(rhs);
					(lower, upper != Self::zero())
				}

				pub fn overflowing_div(self, rhs : Self) -> (Self, bool) {
					(self.div_and_rem(rhs).0, false)
				}

				pub fn overflowing_rem(self, rhs : Self) -> (Self, bool) {
					let a = self.wrapping_abs();
					let rhs = rhs.wrapping_abs();

					let mut r = Self::zero();

					for i in (0..(($size * 128))).rev() {
						r <<= 1;
						r |= a.bit_at(i).into();
						if r >= rhs {
							r -= rhs;
						}
					}
					if self.is_neg() { r.overflowing_neg() } else { (r, false) }
				}

				pub fn overflowing_shl(self, rhs : u32) -> (Self, bool) {
					let mut output = [0; $size];

					let u128s_shifted = rhs.wrapping_shr(7) as usize;
					let bits_shifted = (rhs & 127) as usize;

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
							last_extra_bits = val.wrapping_shr(128 - bits_shifted as u32);
							*val = shifted;
						}
					}

					(Self { inner : output }, last_extra_bits != 0)
				}

				pub fn overflowing_shr(self, rhs : u32) -> (Self, bool) {
					let (value, overflow) = self.raw_overflowing_shr(rhs);
					if self.is_neg() {
						let bit_fill_mask = Self::max_value() << (($size * 128) - rhs);
						(value | bit_fill_mask, overflow)
					} else {
						(value, overflow)
					}
				}

				pub fn raw_overflowing_shr(self, rhs : u32) -> (Self, bool) {
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

				pub fn raw_wrapping_shr(self, rhs : u32) -> Self {
					self.raw_overflowing_shr(rhs).0
				}

				pub fn overflowing_neg(self) -> (Self, bool) {
					(self.wrapping_neg(), self == Self::MIN)
				}

				pub fn overflowing_abs(self) -> (Self, bool) {
					if self.is_neg() {
						self.overflowing_neg()
					} else {
						(self, false)
					}
				}

				pub fn wrapping_add(self, rhs : Self) -> Self {
					self.overflowing_add(rhs).0
				}

				pub fn wrapping_sub(self, rhs : Self) -> Self {
					self.overflowing_sub(rhs).0
				}

				pub fn wrapping_shr(self, rhs : u32) -> Self {
					self.overflowing_shr(rhs).0
				}

				pub fn wrapping_shl(self, rhs : u32) -> Self {
					self.overflowing_shl(rhs).0
				}

				pub fn wrapping_neg(self) -> Self {
					(!self).wrapping_add(Self::one())
				}

				pub fn wrapping_abs(self) -> Self {
					if self.is_neg() {
						self.wrapping_neg()
					} else {
						self
					}
				}

				pub fn rem_euclid(self, rhs: Self) -> Self {
					// Adapted from the standard library.
					let r = self % rhs;
					if r < Self::zero() {
						if rhs < Self::zero() {
							r - rhs
						} else {
							r + rhs
						}
					} else {
						r
					}
				}

				pub fn big_mul(self, rhs : Self) -> (Self, Self) {
					// assert_ne!(self, Self::ZERO, "self == 0");
					// assert_ne!(rhs, Self::ZERO, "rhs == 0");
					let mut result = (Self::zero(), Self::zero());
					let mut this = (Self::zero(), self);
					if self == Self::zero() || rhs == Self::zero() {
						return result;
					}
					// let mut rhs = (Self::zero(), rhs);

					// for i in 0..($size * 128) {
					// 	// dbg!(rhs);
					// 	if self.bit_at(i) == 1 {
					// 		result = Self::add_big(result, rhs);
					// 	}
					// 	rhs = Self::shift_left_one_big(rhs.0, rhs.1);
					// }

					// Adapted from Zengr's C code at https://stackoverflow.com/questions/4456442
					for i in 0..($size * 128) {
						if rhs.bit_at(i) == 1 {
							result = Self::add_big(result, this);
						}
						this = Self::shift_left_one_big(this.0, this.1);
					}

					result
				}

				pub fn mod_mul(self, rhs : Self, modulus : Self) -> Self {
					let output_neg = self.is_neg() ^ rhs.is_neg();
					if self == Self::zero() || rhs == Self::zero() || modulus == Self::one() {
						return Self::zero();
					}

					let mut result = Self::zero();

					let this = self.abs();
					let rhs = rhs.abs();

					for i in (0..($size * 128)).rev() {
						//assert!(result < modulus);
						result <<= 1;
						if result >= modulus {
							result -= modulus;
						}
						//assert!(result < modulus);
						if rhs.bit_at(i) == 1 {
							result += this;
							while result >= modulus {
								result -= modulus;
							}
						}
					}

					if output_neg {
						-result
					} else {
						result
					}
				}

				pub fn big_neg(this : (Self, Self)) -> (Self, Self) {
					Self::add_big((!this.0, !this.1), (Self::zero(), Self::one()))
				}

				fn big_mul_big(this : (Self, Self), mut that: (Self, Self)) -> (Self, Self) {
					let mut result = (Self::zero(), Self::zero());

					if this == (Self::zero(), Self::zero()) {
						return (Self::zero(), Self::zero())
					}

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
					(big_shifted | (small_overflow as u128).into(), small_shifted)
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
					self.divide(rhs)

					// Thank you wikipedia. https://www.wikipedia.org/wiki/Division_algorithm

					// let a = self.wrapping_abs();
					// let rhs = rhs.wrapping_abs();

					// let mut q = Self::zero();
					// let mut r = Self::zero();

					// for i in (0..(($size * 128))).rev() {
					// 	r <<= 1;
					// 	r |= (a >> i) & Self::from(1);
					// 	if r >= rhs {
					// 		r -= rhs;
					// 		q |= Self::from(1) << i;
					// 	}
					// }

					// (
					// 	if !self.is_same_sign(rhs) {
					// 		q.wrapping_neg()
					// 	} else {
					// 		q
					// 	},
					// 	if self.is_neg() {
					// 		r.wrapping_neg()
					// 	} else {
					// 		r
					// 	}
					// )
				}

				fn divide(self, rhs: Self) -> (Self, Self) {
					// Thank you wikipedia. https://www.wikipedia.org/wiki/Division_algorithm
					// println!("huh");
					if rhs == Self::ZERO { panic!("div by zero") }
					if rhs < Self::ZERO { let (q, r) = self.divide(-rhs); return (-q, r); }
					if self < Self::ZERO {
						let (q, r) = (-self).divide(rhs);
						if r == Self::ZERO { return (-q, Self::ZERO) }
						else { return ((-q) - Self::ONE, rhs - r) }
					}
					// At this point, N ≥ 0 and D > 0
					self.divide_unsigned(rhs)
			}
				fn divide_unsigned(self, rhs: Self) -> (Self, Self) {
					// Thank you wikipedia. https://www.wikipedia.org/wiki/Division_algorithm
					let mut q = Self::ZERO;
					let mut r = Self::ZERO;
					for i in (0..($size * 128) - 1).rev() {
						r <<= 1;
						r.inner[0] |= self.bit_at(i) as u128;
						if r >= rhs {
							r -= rhs;
							q |= Self::ONE << i;
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

				pub fn pow(self, rhs: u32) -> Self {
					let (upper, lower) = self.big_pow(rhs);
					assert_eq!(upper, Self::zero(), "Overflow during exponentiation!");
					lower
				}

				pub fn mod_pow(self, mut rhs : Self, modulus : Self) -> Self {
					let rhs_odd = rhs.bit_at(0) == 1;
					if modulus == Self::one() {
						return Self::zero();
					}
					let mut output = Self::one();
					let mut base = (self.abs()) % modulus;
					while rhs > Self::zero() {
						if rhs & Self::one() == Self::one() {
							//output = (output * base) % modulus;
							output = output.mod_mul(base, modulus);
						}
						rhs = rhs.wrapping_shr(1);
						base = base.mod_mul(base, modulus);
					}

					if self.is_neg() && rhs_odd {
						-output
					} else {
						output
					}
				}

				pub fn raw_cmp(self, rhs : Self) -> Ordering {
					for (left, right) in self.inner.iter().zip(rhs.inner.iter()).rev() {
						match left.cmp(right) {
							Ordering::Equal => continue,
							x => return x,
						}
					}

					Ordering::Equal
				}

				pub fn is_neg(self) -> bool {
					self.bit_at(($size * 128) - 1) == 1
				}

				pub fn abs(self) -> Self {
					if self.is_neg() {
						-self
					} else {
						self
					}
				}

				pub fn is_same_sign(self, other : Self) -> bool {
					self.is_neg() == other.is_neg()
				}

				pub fn bit_at(self, index : u32) -> u8 {
					assert!(index < ($size * 128), "bit index out of range");

					(self.inner[(index.wrapping_shr(7)) as usize].wrapping_shr(index & 127) & 1) as u8
				}

				pub fn extended_gcd(self, rhs: Self) -> (Self, Self, Self, Self, Self) {
					let mut r = rhs;
					let mut r_prev = self;
					let mut s = Self::zero();
					let mut s_prev = Self::one();
					let mut t = Self::one();
					let mut t_prev = Self::zero();

					while r != Self::zero() {
						let quotient = (r_prev).div_euclid(r);

						let r_new = (r, r_prev.rem_euclid(r));
						r_prev = r_new.0;
						r = r_new.1;

						let s_new = (s, s_prev - (quotient * s));
						s_prev = s_new.0;
						s = s_new.1;

						let t_new = (t, t_prev - (quotient * t));
						t_prev = t_new.0;
						t = t_new.1;
					}

					(s_prev, t_prev, r_prev, t, s)
				}

				pub fn bezout_coefficients(self, rhs: Self) -> (Self, Self) {
					let ext_gcd = self.extended_gcd(rhs);
					(ext_gcd.0, ext_gcd.1)
				}

				pub fn gcd(self, rhs: Self) -> Self {
					self.extended_gcd(rhs).2
				}

				/// Calculates the modular multiplicative inverse of `self`. Returns `None` if no such value exists.
				pub fn mod_mul_inverse(self, modulus: Self) -> Option<Self> {
					// use std::time::Instant;
					// let start = Instant::now();

					// dbg!(&self, &modulus);

					let (mut s_prev, _, r_prev, _, _) = self.extended_gcd(modulus);

					// let mut iterations = 0;
					// let mut t = Self::zero();
					// let mut t_new = Self::one();
					// let mut r = modulus;
					// let mut r_new = self;


					// while r_new != Self::zero() {
					// 	let quotient = r.div_euclid(r_new);

					// 	println!("t: {:#034X?}, r: {:#034X?}", t, r);

					// 	let r_pair_next = (r_new, r - (quotient * r_new));
					// 	r = r_pair_next.0;
					// 	r_new = r_pair_next.1;

					// 	let t_pair_next = (t_new, t - (quotient * t_new));
					// 	t = t_pair_next.0;
					// 	t_new = t_pair_next.1;

					// 	iterations += 1;
					// }
					// let end = Instant::now();
					// let time_taken = end.duration_since(start).as_secs_f64();
					// println!("mod_mul_inverse took {} iterations and {} seconds (avg. {} secs/iter)", iterations, time_taken, time_taken / (iterations as f64) );

					if r_prev > Self::ONE {
						// println!("thingie");
						return None;
					}
					if s_prev < Self::ZERO {
						s_prev += modulus;
					}

					Some(s_prev)
				}

				pub fn div_euclid(self, rhs: Self) -> Self {
					// Adapted from the standard library
					let (quot, rem) = self.div_and_rem(rhs);
					if rem < Self::zero() {
						if rhs > Self::zero() {
							quot - Self::one()
						} else {
							quot + 1
						}
					} else {
						quot
					}
				}

				into_int! {
					u8, into_u8;
					u16, into_u16;
					u32, into_u32;
					u64, into_u64;
					u128, into_u128;
					usize, into_usize;
					i8, into_i8;
					i16, into_i16;
					i32, into_i32;
					i64, into_i64;
					i128, into_i128;
					isize, into_isize
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
					match (self.is_neg() as u8, other.is_neg() as u8) {
						(0, 0) => (),
						(1, 0) => return Ordering::Less,
						(0, 1) => return Ordering::Greater,
						(1, 1) => (),
						_=> unreachable!(),
					};

					self.raw_cmp(*other)
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
					debug_assert!(!overflowed, "attempt to add with overflow");
					value
				}
			}

			impl AddAssign for $name {
				fn add_assign(&mut self, rhs : Self) {
					let (value, overflowed) = self.overflowing_add(rhs);
					debug_assert!(!overflowed, "attempt to add with overflow");
					*self = value;
				}
			}

			impl Mul for $name {
				type Output = Self;
				fn mul(self, rhs : Self) -> Self {
					let (value, overflowed) = self.overflowing_mul(rhs);
					assert!(!overflowed, "attempt to multiply with overflow");
					value
				}
			}

			impl MulAssign for $name {
				fn mul_assign(&mut self, rhs : Self) {
					let (value, overflowed) = self.overflowing_mul(rhs);
					debug_assert!(!overflowed, "attempt to multiply with overflow");
					*self = value;
				}
			}

			impl Sub for $name {
				type Output = Self;
				fn sub(self, rhs : Self) -> Self {
					let (value, overflowed) = self.overflowing_sub(rhs);
					debug_assert!(!overflowed, "attempt to subtract with overflow");
					value
				}
			}

			impl SubAssign for $name {
				fn sub_assign(&mut self, rhs : Self) {
					let (value, overflowed) = self.overflowing_sub(rhs);
					debug_assert!(!overflowed, "attempt to subtract with overflow");
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

			impl Neg for $name {
				type Output = Self;
				fn neg(self) -> Self {
					debug_assert_ne!(self, Self::one() << (($size * 128) - 1), "attempt to negate with overflow");
					(!self) + 1
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

			impl Binary for $name {
				fn fmt(&self, fmtr: &mut fmt::Formatter) -> fmt::Result {
					for i in self.inner.iter().rev() {
						write!(fmtr, "{:0128b}", i)?;
					}
					Ok(())
				}
			}

			impl UpperHex for $name {
				fn fmt(&self, fmtr: &mut fmt::Formatter) -> fmt::Result {
					for i in self.inner.iter().rev() {
						write!(fmtr, "{:032X}", i)?;
					}
					Ok(())
				}
			}

			with_uint_ops_impl!($name; u8, u16, u32, u64, u128, usize);

			with_int_ops_impl!($name; i8, i16, i32, i64, i128, isize);
		)*
	};
}

structs! {
    I256, 2;
    I512, 4;
    I1024, 8;
    I2048, 16;
    I4096, 32;
    I8192, 64;
    I16384, 128;
    I32768, 256;
    I65536, 512
}

impl From<I256> for I512 {
    fn from(other: I256) -> Self {
        let mut new_array = [0; 4];
        (&mut new_array[0..2]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<I256> for I1024 {
    fn from(other: I256) -> Self {
        let mut new_array = [0; 8];
        (&mut new_array[0..2]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<I512> for I1024 {
    fn from(other: I512) -> Self {
        let mut new_array = [0; 8];
        (&mut new_array[0..4]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<I256> for I2048 {
    fn from(other: I256) -> Self {
        let mut new_array = [0; 16];
        (&mut new_array[0..2]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<I512> for I2048 {
    fn from(other: I512) -> Self {
        let mut new_array = [0; 16];
        (&mut new_array[0..4]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<I1024> for I2048 {
    fn from(other: I1024) -> Self {
        let mut new_array = [0; 16];
        (&mut new_array[0..8]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<I256> for I4096 {
    fn from(other: I256) -> Self {
        let mut new_array = [0; 32];
        (&mut new_array[0..2]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<I512> for I4096 {
    fn from(other: I512) -> Self {
        let mut new_array = [0; 32];
        (&mut new_array[0..4]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<I1024> for I4096 {
    fn from(other: I1024) -> Self {
        let mut new_array = [0; 32];
        (&mut new_array[0..8]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<I2048> for I4096 {
    fn from(other: I2048) -> Self {
        let mut new_array = [0; 32];
        (&mut new_array[0..16]).copy_from_slice(&other.inner);
        Self { inner: new_array }
    }
}

impl From<(I256, I256)> for I512 {
    fn from((other_big, other_small): (I256, I256)) -> Self {
        let mut new_array = [0; 4];
        (&mut new_array[0..2]).copy_from_slice(&other_small.inner);
        (&mut new_array[2..4]).copy_from_slice(&other_big.inner);
        Self { inner: new_array }
    }
}

impl From<(I512, I512)> for I1024 {
    fn from((other_big, other_small): (I512, I512)) -> Self {
        let mut new_array = [0; 8];
        (&mut new_array[0..4]).copy_from_slice(&other_small.inner);
        (&mut new_array[4..8]).copy_from_slice(&other_big.inner);
        Self { inner: new_array }
    }
}

impl From<(I1024, I1024)> for I2048 {
    fn from((other_big, other_small): (I1024, I1024)) -> Self {
        let mut new_array = [0; 16];
        (&mut new_array[0..8]).copy_from_slice(&other_small.inner);
        (&mut new_array[8..16]).copy_from_slice(&other_big.inner);
        Self { inner: new_array }
    }
}

impl From<(I2048, I2048)> for I4096 {
    fn from((other_big, other_small): (I2048, I2048)) -> Self {
        let mut new_array = [0; 32];
        (&mut new_array[0..16]).copy_from_slice(&other_small.inner);
        (&mut new_array[16..32]).copy_from_slice(&other_big.inner);
        Self { inner: new_array }
    }
}

impl From<U256> for I256 {
    fn from(other: U256) -> Self {
        Self::from_array(other.into_inner())
    }
}

impl From<U512> for I512 {
    fn from(other: U512) -> Self {
        Self::from_array(other.into_inner())
    }
}

impl From<U1024> for I1024 {
    fn from(other: U1024) -> Self {
        Self::from_array(other.into_inner())
    }
}

impl From<U2048> for I2048 {
    fn from(other: U2048) -> Self {
        Self::from_array(other.into_inner())
    }
}

impl From<U4096> for I4096 {
    fn from(other: U4096) -> Self {
        Self::from_array(other.into_inner())
    }
}

impl I256 {
    pub fn to_unsigned(self) -> U256 {
        U256 { inner: self.inner }
    }
}

impl I512 {
    pub fn to_unsigned(self) -> U512 {
        U512 { inner: self.inner }
    }
}

impl I1024 {
    pub fn to_unsigned(self) -> U1024 {
        U1024 { inner: self.inner }
    }
}

impl I2048 {
    pub fn to_unsigned(self) -> U2048 {
        U2048 { inner: self.inner }
    }
}

impl I4096 {
    pub fn to_unsigned(self) -> U4096 {
        U4096 { inner: self.inner }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rem() {
        assert_eq!(-40 % 7, (I256::from(-40) % I256::from(7)).into_i32());
    }

    #[test]
    fn div_euclid() {
        println!("{:032X?}", I256::from(-6));
        println!("{:032X?}", I256::from(5));
        println!("{:032X?}", I256::from(-6).div_euclid(I256::from(5)));
        println!("{:032X?}", I256::from(-2));
        assert_eq!(I256::from(-6).div_euclid(I256::from(5)), I256::from(-2));
    }

    #[test]
    fn mul() {
        use std::io::{stdout, Write};
        for i in -100..100 {
            for j in -100..100 {
                assert_eq!(I256::from(i) * I256::from(j), I256::from(i * j));
                // print!("H");
            }
            print!("H");
            stdout().flush().unwrap();
        }
    }
}
