/// Convert an `f32` to IEEE-754 binary16 bits.
///
/// Round-to-nearest (ties away from zero) is used; it is accurate enough for
/// animation data (matrices, morph weights) and avoids pulling in a dependency
/// for half floats.
///
/// See https://goals.rust-lang.org/2026/stabilizing-f16.html for more info.
pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 0xFF {
        // Infinity or NaN.
        let payload = if mantissa == 0 { 0 } else { 0x0200 };
        return sign | 0x7C00 | payload;
    }

    // Re-bias from f32 (127) to f16 (15).
    let mut exponent = exponent - 127 + 15;

    if exponent >= 0x1F {
        return sign | 0x7C00;
    }

    if exponent <= 0 {
        // Subnormal half or zero.
        if exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x0080_0000;
        let shift = 14 - exponent;
        let rounded = (mantissa + (1 << (shift - 1))) >> shift;
        return sign | rounded as u16;
    }

    let rounded = mantissa + 0x1000;
    if rounded & 0x0080_0000 != 0 {
        // Rounding carried into the exponent.
        exponent += 1;
        if exponent >= 0x1F {
            return sign | 0x7C00;
        }
        return sign | ((exponent as u16) << 10);
    }
    sign | ((exponent as u16) << 10) | ((rounded >> 13) as u16)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn f16_encoding_matches_known_values() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
        assert_eq!(f32_to_f16_bits(-1.0), 0xBC00);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
        assert_eq!(f32_to_f16_bits(-2.0), 0xC000);
    }
}
