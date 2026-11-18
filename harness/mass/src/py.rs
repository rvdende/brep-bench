//! Python's own arithmetic and printing, as the measurer had them when it was written in Python
//! (harness/mass.py, since removed), so that its answers stay the same digit for digit: its floored
//! `%`, its `round`, `min` and `max`, a float's `repr`, a string's, and `json.dumps`.

/// Why a file is not measured, as the Python measurer's exceptions told them apart.
#[derive(Debug, Clone)]
pub enum Err {
    /// `Unsupported`: what the file holds that the measurer cannot measure.
    Unsupported(String),
    /// `Invalid`: what makes the file no solid at all.
    Invalid(String),
    /// An exception of Python's own that `read` catches (ValueError, ZeroDivisionError,
    /// IndexError, KeyError, TypeError): its type's name and its text.
    Py(&'static str, String),
    /// One that `read` does not catch (OverflowError, StopIteration): Python ends with a
    /// traceback and no answer.
    Uncaught(&'static str, String),
}

pub type R<T> = Result<T, Err>;

pub fn unsupported<T>(s: impl Into<String>) -> R<T> {
    Err(Err::Unsupported(s.into()))
}

pub fn zero_division<T>() -> R<T> {
    Err(Err::Py("ZeroDivisionError", "division by zero".into()))
}

pub fn index_error<T>() -> R<T> {
    Err(Err::Py("IndexError", "list index out of range".into()))
}

/// `a / b`, as Python divides floats: by zero is an error.
#[inline(always)]
pub fn div(a: f64, b: f64) -> R<f64> {
    if b == 0.0 {
        return zero_division();
    }
    Ok(a / b)
}

/// `a % b` of floats: the remainder takes the divisor's sign.
#[inline]
pub fn pymod(a: f64, b: f64) -> R<f64> {
    if b == 0.0 {
        return zero_division();
    }
    let mut m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m += b;
        }
    } else {
        m = 0.0f64.copysign(b);
    }
    Ok(m)
}

/// `max(a, b)`: the first, unless the second is greater.
#[inline(always)]
pub fn pymax(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// `min(a, b)`: the first, unless the second is less.
#[inline(always)]
pub fn pymin(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// `round(x)`, `math.floor(x)` or `math.ceil(x)`'s way with what is no number: an error.
#[inline]
pub fn integer(x: f64, rounded: f64) -> R<f64> {
    if x.is_nan() {
        return Err(Err::Py(
            "ValueError",
            "cannot convert float NaN to integer".into(),
        ));
    }
    if x.is_infinite() {
        return Err(Err::Uncaught(
            "OverflowError",
            "cannot convert float infinity to integer".into(),
        ));
    }
    Ok(rounded)
}

/// `round(x)`: to the nearest integer, a half to the even one.
#[inline]
pub fn round(x: f64) -> R<f64> {
    integer(x, x.round_ties_even())
}

/// `round(x, 9)`.
pub fn round9(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{:.9}", x).parse().unwrap_or(x)
}

/// `sorted(set(xs))`: equal values once (the first of them), ascending.
pub fn sorted_set(xs: impl IntoIterator<Item = f64>) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::new();
    for x in xs {
        out.push(x);
    }
    // a stable sort keeps the first of equal values first
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out.dedup_by(|b, a| a == b);
    out
}

/// `sorted(xs)`.
pub fn sorted(mut xs: Vec<f64>) -> Vec<f64> {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs
}

/// `math.hypot(a, b)`, as CPython takes it (its `vector_norm`: scaled by a power of two, the
/// squares summed without loss, the square root corrected once), which is not always the C
/// library's to the last bit.
pub fn hypot(a: f64, b: f64) -> f64 {
    let (a, b) = (a.abs(), b.abs());
    if a.is_infinite() || b.is_infinite() {
        return f64::INFINITY;
    }
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let max = if b > a { b } else { a };
    if max == 0.0 {
        return max;
    }
    // frexp's exponent: max = f × 2^e, 0.5 <= f < 1
    let bits = max.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let e = if biased == 0 {
        (64 - (bits & ((1 << 52) - 1)).leading_zeros() as i32) - 1074
    } else {
        biased - 1022
    };
    if e < -1023 {
        return f64::MIN_POSITIVE * hypot(a / f64::MIN_POSITIVE, b / f64::MIN_POSITIVE);
    }
    // 2^-e
    let scale = if -e >= -1022 {
        f64::from_bits(((1023 - e) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (1074 - e))
    };
    let (mut csum, mut frac1, mut frac2) = (1.0f64, 0.0f64, 0.0f64);
    for x in [a, b] {
        let x = x * scale;
        let hi = x * x;
        let lo = x.mul_add(x, -hi);
        let sum = csum + hi;
        let low = (csum - sum) + hi;
        csum = sum;
        frac1 += lo;
        frac2 += low;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let hi = -h * h;
    let lo = (-h).mul_add(h, -hi);
    let sum = csum + hi;
    let low = (csum - sum) + hi;
    csum = sum;
    frac1 += lo;
    frac2 += low;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

/// `repr(x)` of a float: the shortest digits that read back the same, in Python's layout.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let s = format!("{:e}", x);
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let (sign, mant) = match mant.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let n = digits.len() as i32;
    let decpt = exp + 1;
    let mut out = String::from(sign);
    if -4 < decpt && decpt <= 16 {
        if decpt <= 0 {
            out.push_str("0.");
            for _ in 0..-decpt {
                out.push('0');
            }
            out.push_str(&digits);
        } else if decpt >= n {
            out.push_str(&digits);
            for _ in 0..decpt - n {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..decpt as usize]);
            out.push('.');
            out.push_str(&digits[decpt as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.abs()));
    }
    out
}

/// A float as `json.dumps` writes it.
pub fn float_json(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        }
    } else {
        float_repr(x)
    }
}

/// `repr(s)` of a string (its characters Latin-1's, as the file is read).
pub fn str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        let n = c as u32;
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            _ if n < 0x20 || (0x7f..=0xa0).contains(&n) || n == 0xad => {
                out.push_str(&format!("\\x{:02x}", n));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// A string as `json.dumps` writes it: ASCII only.
pub fn str_json(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e && (c as u32) != 0x7f => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Bytes as the text Python reads them as (`encoding="latin-1"`).
pub fn latin1(b: &[u8]) -> String {
    b.iter().map(|&c| c as char).collect()
}

/// `c.isspace()`, and the regular expressions' `\s`, of a Latin-1 character.
#[inline]
pub fn is_space(c: u8) -> bool {
    matches!(
        c,
        b'\t' | b'\n' | 0x0b | 0x0c | b'\r' | 0x1c..=0x1f | b' ' | 0x85 | 0xa0
    )
}

pub fn strip(mut b: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = b {
        if is_space(*first) { b = rest } else { break }
    }
    while let [rest @ .., last] = b {
        if is_space(*last) { b = rest } else { break }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_prints_them() {
        for (x, s) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1.5e16, "1.5e+16"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.25e-7, "1.25e-07"),
            (-123.456, "-123.456"),
            (1e100, "1e+100"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (5e-324, "5e-324"),
            (-0.0, "-0.0"),
            (1e22, "1e+22"),
            (0.3, "0.3"),
            (2.5e-5, "2.5e-05"),
            (1234567890123456.0, "1234567890123456.0"),
            (0.001234, "0.001234"),
        ] {
            assert_eq!(float_repr(x), s);
        }
    }

    #[test]
    fn the_remainder_takes_the_divisors_sign() {
        assert_eq!(pymod(-1.0, 3.0).unwrap(), 2.0);
        assert_eq!(pymod(1.0, -3.0).unwrap(), -2.0);
        assert_eq!(pymod(6.0, 3.0).unwrap(), 0.0);
        assert!(pymod(-6.0, -3.0).unwrap().is_sign_negative());
        assert!(pymod(1.0, 0.0).is_err());
    }

    #[test]
    fn hypotenuses() {
        assert_eq!(hypot(3.0, 4.0), 5.0);
        assert_eq!(hypot(-0.0, 0.0), 0.0);
        assert_eq!(hypot(0.0, -3.0), 3.0);
        assert_eq!(hypot(1e300, 1e300), 1.4142135623730952e300);
        assert_eq!(hypot(3e-320, 4e-320), 5e-320);
        assert_eq!(hypot(f64::NAN, f64::INFINITY), f64::INFINITY);
        // (values of Python's math.hypot)
        assert_eq!(hypot(0.1, 0.7), 0.7071067811865475);
        assert_eq!(hypot(1.3, 2.9), 3.1780497164141406);
    }

    #[test]
    fn rounding() {
        assert_eq!(round(2.5).unwrap(), 2.0);
        assert_eq!(round(3.5).unwrap(), 4.0);
        assert_eq!(round(-0.5).unwrap(), 0.0);
        assert_eq!(round9(1.23456789049), 1.23456789);
    }

    #[test]
    fn strings() {
        assert_eq!(str_repr("a'b"), "\"a'b\"");
        assert_eq!(str_repr("a\nb"), "'a\\nb'");
        assert_eq!(str_json("é\"\n"), "\"\\u00e9\\\"\\n\"");
    }
}
