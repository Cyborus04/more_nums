use crate::float::*;

use std::{println, print};

#[test]
fn f16_add() { 
	assert_eq!(F16::one() + F16::one(), F16::from_bits(0b0_10000_0000000000));
}

#[test]
fn f16_sub() { 
	assert_eq!(F16::from_bits(0b0_10000_0000000000) - F16::one(), F16::one());
}