//! A STEP file's entities (as harness/step.py's `parse` reads them) and an entity's arguments.
use crate::py::*;
use std::collections::HashMap;

/// One entity: its names (one, or several for a complex entity), and its record's text after
/// `#n=`, strings blanked to `''` and comments dropped.
pub struct Entity {
    pub id: u64,
    pub names: Vec<String>,
    pub text: Vec<u8>,
}

/// The file's entities by number, in the order the file first gives each.
#[derive(Default)]
pub struct Entities {
    pub list: Vec<Entity>,
    pub index: HashMap<u64, usize>,
}

impl Entities {
    pub fn get(&self, id: u64) -> Option<&Entity> {
        self.index.get(&id).map(|&i| &self.list[i])
    }
}

/// A string from its opening quote at `i` (`'(?:[^']|'')*'`): the index past its closing quote.
fn string_end(text: &[u8], i: usize) -> Option<usize> {
    let n = text.len();
    let mut j = i + 1;
    let mut pair = None;
    loop {
        if j >= n {
            // no closing quote: the last doubled quote's first is one
            return pair.map(|p: usize| p + 1);
        }
        if text[j] != b'\'' {
            j += 1;
        } else if j + 1 < n && text[j + 1] == b'\'' {
            pair = Some(j);
            j += 2;
        } else {
            return Some(j + 1);
        }
    }
}

/// The file's records, strings blanked to '' and comments dropped.
fn records(text: &[u8], mut each: impl FnMut(&[u8])) {
    let n = text.len();
    let mut body: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < n {
        let c = text[i];
        if c == b'\'' {
            match string_end(text, i) {
                Some(j) => {
                    body.extend_from_slice(b"''");
                    i = j;
                }
                None => i += 1,
            }
        } else if c == b'/' {
            let close = if i + 1 < n && text[i + 1] == b'*' {
                text[i + 2..].windows(2).position(|w| w == b"*/")
            } else {
                None
            };
            match close {
                Some(k) => i = i + 2 + k + 2,
                None => {
                    body.push(b'/');
                    i += 1;
                }
            }
        } else if c == b';' {
            each(&body);
            body.clear();
            i += 1;
        } else {
            let mut j = i;
            while j < n && !matches!(text[j], b'\'' | b';' | b'/') {
                j += 1;
            }
            body.extend_from_slice(&text[i..j]);
            i = j;
        }
    }
}

fn name_at(body: &[u8], i: usize) -> Option<usize> {
    if i < body.len() && (body[i].is_ascii_uppercase() || body[i] == b'_') {
        let mut j = i + 1;
        while j < body.len()
            && (body[j].is_ascii_uppercase() || body[j].is_ascii_digit() || body[j] == b'_')
        {
            j += 1;
        }
        Some(j)
    } else {
        None
    }
}

/// An entity's names, from its record's text after `#n=`.
fn names(body: &[u8]) -> Vec<String> {
    let body = strip(body);
    let mut out = Vec::new();
    if body.first() == Some(&b'(') {
        // a complex entity: its parts' names, at the outer level
        let (mut depth, mut i) = (0i64, 1);
        while i < body.len() {
            let c = body[i];
            if c == b'(' {
                depth += 1;
            } else if c == b')' {
                depth -= 1;
            } else if depth == 0 {
                if let Some(j) = name_at(body, i) {
                    out.push(latin1(&body[i..j]));
                    i = j;
                    continue;
                }
            }
            i += 1;
        }
    } else {
        match name_at(body, 0) {
            Some(j) => out.push(latin1(&body[..j])),
            None => out.push("?".into()),
        }
    }
    out
}

/// The file's entities by number (harness/step.py's `parse`, less the schema, which the measurer
/// does not read).
pub fn parse(text: &[u8]) -> Entities {
    let mut entities = Entities::default();
    let mut data = false;
    records(text, |body| {
        let b = strip(body);
        if !data {
            if b.starts_with(b"FILE_SCHEMA") {
                // the header's own; harness/step.py takes the schema's name from it. (It tests
                // for DATA only where this is not a FILE_SCHEMA with a name, which cannot be DATA.)
            } else if b == b"DATA" || b.starts_with(b"DATA(") {
                data = true;
            }
            return;
        }
        if b == b"ENDSEC" {
            data = false;
            return;
        }
        // \s*#(\d+)\s*=\s*(.*)
        let n = body.len();
        let mut i = 0;
        while i < n && is_space(body[i]) {
            i += 1;
        }
        if i >= n || body[i] != b'#' {
            return;
        }
        i += 1;
        let d0 = i;
        while i < n && body[i].is_ascii_digit() {
            i += 1;
        }
        if i == d0 {
            return;
        }
        let id = number(&body[d0..i]);
        while i < n && is_space(body[i]) {
            i += 1;
        }
        if i >= n || body[i] != b'=' {
            return;
        }
        i += 1;
        while i < n && is_space(body[i]) {
            i += 1;
        }
        let e = Entity {
            id,
            names: names(&body[i..]),
            text: body[i..].to_vec(),
        };
        match entities.index.get(&id) {
            Some(&k) => entities.list[k] = e,
            None => {
                entities.index.insert(id, entities.list.len());
                entities.list.push(e);
            }
        }
    });
    entities
}

/// An entity's number from its digits (one past any a file could hold is the largest there is).
fn number(digits: &[u8]) -> u64 {
    let mut n: u64 = 0;
    for &d in digits {
        n = n.saturating_mul(10).saturating_add((d - b'0') as u64);
    }
    n
}

/// An argument of an entity.
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    List(Vec<Val>),
    /// A reference to another entity, `#12`.
    Ref(u64),
    /// A string, `'...'`: always empty, as the records' strings are blanked.
    Str(String),
    /// An enumeration, `.T.`, `.MILLI.`.
    Enum(String),
    /// A bare name.
    Name(String),
    /// `$` or `*`.
    None,
    Num(f64),
}

impl Val {
    /// What Python calls its type.
    fn type_name(&self) -> &'static str {
        match self {
            Val::List(_) => "list",
            Val::Ref(_) => "Ref",
            Val::Str(_) | Val::Name(_) => "str",
            Val::Enum(_) => "Enum",
            Val::None => "NoneType",
            Val::Num(_) => "float",
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) | Val::Enum(s) | Val::Name(s) => Some(s),
            _ => None,
        }
    }

    /// `self == "..."`.
    pub fn is(&self, s: &str) -> bool {
        self.as_str() == Some(s)
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Val::None)
    }

    /// `self[i]`.
    pub fn at(&self, i: usize) -> R<&Val> {
        match self {
            Val::List(v) => match v.get(i) {
                Some(x) => Ok(x),
                None => index_error(),
            },
            Val::Str(s) | Val::Enum(s) | Val::Name(s) if i >= s.len() => {
                Err(Err::Py("IndexError", "string index out of range".into()))
            }
            _ => Err(Err::Py(
                "TypeError",
                format!("'{}' object is not subscriptable", self.type_name()),
            )),
        }
    }

    /// `for x in self`.
    pub fn items(&self) -> R<&[Val]> {
        match self {
            Val::List(v) => Ok(v),
            _ => Err(Err::Py(
                "TypeError",
                format!("'{}' object is not iterable", self.type_name()),
            )),
        }
    }

    /// `len(self)`.
    pub fn len(&self) -> R<usize> {
        match self {
            Val::List(v) => Ok(v.len()),
            Val::Str(s) | Val::Enum(s) | Val::Name(s) => Ok(s.len()),
            _ => Err(Err::Py(
                "TypeError",
                format!("object of type '{}' has no len()", self.type_name()),
            )),
        }
    }

    /// `float(self)`.
    pub fn float(&self) -> R<f64> {
        match self {
            Val::Num(x) => Ok(*x),
            Val::Ref(n) => Ok(*n as f64),
            Val::Str(s) | Val::Enum(s) | Val::Name(s) => Err(Err::Py(
                "ValueError",
                format!("could not convert string to float: {}", str_repr(s)),
            )),
            _ => Err(Err::Py(
                "TypeError",
                format!(
                    "float() argument must be a string or a real number, not '{}'",
                    self.type_name()
                ),
            )),
        }
    }

    /// `int(self)`.
    pub fn int(&self) -> R<i64> {
        match self {
            Val::Num(x) => Ok(integer(*x, x.trunc())? as i64),
            Val::Ref(n) => Ok(*n as i64),
            Val::Str(s) | Val::Enum(s) | Val::Name(s) => Err(Err::Py(
                "ValueError",
                format!("invalid literal for int() with base 10: {}", str_repr(s)),
            )),
            _ => Err(Err::Py(
                "TypeError",
                format!(
                    "int() argument must be a string, a bytes-like object or a real number, not '{}'",
                    self.type_name()
                ),
            )),
        }
    }

    /// The list's items, when there are exactly `n` (`a, b, c = self`).
    pub fn unpack(&self, n: usize) -> R<&[Val]> {
        let v = match self {
            Val::List(v) => v,
            _ => {
                return Err(Err::Py(
                    "TypeError",
                    format!("cannot unpack non-iterable {} object", self.type_name()),
                ));
            }
        };
        if v.len() < n {
            return Err(Err::Py(
                "ValueError",
                format!(
                    "not enough values to unpack (expected {}, got {})",
                    n,
                    v.len()
                ),
            ));
        }
        if v.len() > n {
            return Err(Err::Py(
                "ValueError",
                format!(
                    "too many values to unpack (expected {}, got {})",
                    n,
                    v.len()
                ),
            ));
        }
        Ok(v)
    }

    /// `repr(self)`.
    pub fn repr(&self) -> String {
        match self {
            Val::List(v) => {
                format!(
                    "[{}]",
                    v.iter().map(|x| x.repr()).collect::<Vec<_>>().join(", ")
                )
            }
            Val::Str(s) | Val::Enum(s) | Val::Name(s) => str_repr(s),
            _ => self.str(),
        }
    }

    /// `str(self)`, as an f-string writes it.
    pub fn str(&self) -> String {
        match self {
            Val::List(_) => self.repr(),
            Val::Ref(n) => n.to_string(),
            Val::Str(s) | Val::Enum(s) | Val::Name(s) => s.clone(),
            Val::None => "None".into(),
            Val::Num(x) => float_repr(*x),
        }
    }
}

/// An entity's arguments by the name of each of its parts, in the file's order.
pub type Parts = Vec<(String, Val)>;

pub fn part<'a>(parts: &'a Parts, name: &str) -> Option<&'a Val> {
    parts.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

fn tokens(text: &[u8]) -> R<Vec<&[u8]>> {
    let n = text.len();
    let mut out = Vec::new();
    let mut i = 0;
    let digit = |k: usize| k < n && text[k].is_ascii_digit();
    let word = |k: usize| k < n && (text[k].is_ascii_alphanumeric() || text[k] == b'_');
    while i < n {
        let mut s = i;
        while s < n && is_space(text[s]) {
            s += 1;
        }
        let mut e = None;
        if s < n {
            let c = text[s];
            if c == b'#' && digit(s + 1) {
                let mut j = s + 1;
                while digit(j) {
                    j += 1;
                }
                e = Some(j);
            } else if c == b'\'' {
                e = string_end(text, s);
            }
            if e.is_none() && c == b'.' && word(s + 1) {
                let mut j = s + 1;
                while word(j) {
                    j += 1;
                }
                if j < n && text[j] == b'.' {
                    e = Some(j + 1);
                }
            }
            if e.is_none() {
                // [-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?
                let mut j = s;
                if text[j] == b'-' || text[j] == b'+' {
                    j += 1;
                }
                let mut ok = false;
                if digit(j) {
                    while digit(j) {
                        j += 1;
                    }
                    if j < n && text[j] == b'.' {
                        j += 1;
                    }
                    while digit(j) {
                        j += 1;
                    }
                    ok = true;
                } else if j < n && text[j] == b'.' && digit(j + 1) {
                    j += 1;
                    while digit(j) {
                        j += 1;
                    }
                    ok = true;
                }
                if ok {
                    if j < n && (text[j] == b'e' || text[j] == b'E') {
                        let mut k = j + 1;
                        if k < n && (text[k] == b'-' || text[k] == b'+') {
                            k += 1;
                        }
                        if digit(k) {
                            while digit(k) {
                                k += 1;
                            }
                            j = k;
                        }
                    }
                    e = Some(j);
                }
            }
            if e.is_none() {
                if matches!(c, b'$' | b'*' | b'(' | b')' | b',') {
                    e = Some(s + 1);
                } else if c.is_ascii_alphabetic() || c == b'_' {
                    let mut j = s + 1;
                    while word(j) {
                        j += 1;
                    }
                    e = Some(j);
                }
            }
        }
        match e {
            Some(e) => {
                out.push(&text[s..e]);
                i = e;
            }
            None => {
                if strip(&text[i..]).is_empty() {
                    break;
                }
                let shown = &text[i..n.min(i + 40)];
                return unsupported(format!(
                    "cannot read an entity's arguments: {}",
                    str_repr(&latin1(shown))
                ));
            }
        }
    }
    Ok(out)
}

struct Reader<'a> {
    text: &'a [u8],
    toks: Vec<&'a [u8]>,
    pos: usize,
}

impl<'a> Reader<'a> {
    fn tok(&self, i: usize) -> R<&'a [u8]> {
        match self.toks.get(i) {
            Some(t) => Ok(t),
            None => index_error(),
        }
    }

    fn value(&mut self) -> R<Val> {
        let t = self.tok(self.pos)?;
        self.pos += 1;
        if t == b"(" {
            let mut items = Vec::new();
            if self.tok(self.pos)? == b")" {
                self.pos += 1;
                return Ok(Val::List(items));
            }
            loop {
                items.push(self.value()?);
                let t = self.tok(self.pos)?;
                self.pos += 1;
                if t == b")" {
                    return Ok(Val::List(items));
                }
                if t != b"," {
                    let shown = &self.text[..self.text.len().min(60)];
                    return unsupported(format!(
                        "cannot read an entity's arguments: {}",
                        str_repr(&latin1(shown))
                    ));
                }
            }
        }
        if t[0] == b'#' {
            return Ok(Val::Ref(number(&t[1..])));
        }
        if t[0] == b'\'' {
            return Ok(Val::Str(latin1(&t[1..t.len() - 1])));
        }
        if t[0] == b'.' {
            // (a number written with no digit before its point is read as one of these too)
            let inner = if t.len() >= 2 {
                &t[1..t.len() - 1]
            } else {
                &t[..0]
            };
            return Ok(Val::Enum(latin1(inner).to_ascii_uppercase()));
        }
        if t == b"$" || t == b"*" {
            return Ok(Val::None);
        }
        if t[0].is_ascii_alphabetic() || t[0] == b'_' {
            // a typed value: NAME(value)
            if self.pos < self.toks.len() && self.toks[self.pos] == b"(" {
                self.pos += 1;
                let v = self.value()?;
                self.pos += 1; // ')'
                return Ok(v);
            }
            return Ok(Val::Name(latin1(t)));
        }
        match float(t) {
            Some(x) => Ok(Val::Num(x)),
            None => Err(Err::Py(
                "ValueError",
                format!(
                    "could not convert string to float: {}",
                    str_repr(&latin1(t))
                ),
            )),
        }
    }

    fn part(&mut self) -> R<(String, Val)> {
        let name = self.tok(self.pos)?;
        self.pos += 1;
        if self.tok(self.pos)? != b"(" {
            let shown = &self.text[..self.text.len().min(60)];
            return unsupported(format!(
                "cannot read an entity: {}",
                str_repr(&latin1(shown))
            ));
        }
        Ok((latin1(name).to_ascii_uppercase(), self.value()?))
    }
}

/// `float(t)` of a number's token.
fn float(t: &[u8]) -> Option<f64> {
    let s = std::str::from_utf8(t).ok()?;
    let (mant, exp) = match s.find(['e', 'E']) {
        Some(k) => (&s[..k], &s[k + 1..]),
        None => (s, ""),
    };
    let (sign, mant) = match mant.as_bytes().first()? {
        b'-' => ("-", &mant[1..]),
        b'+' => ("", &mant[1..]),
        _ => ("", mant),
    };
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    if int.is_empty() && frac.is_empty() {
        return None;
    }
    if !int.bytes().all(|c| c.is_ascii_digit()) || !frac.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let exp = if exp.is_empty() { "0" } else { exp };
    format!(
        "{}{}.{}e{}",
        sign,
        if int.is_empty() { "0" } else { int },
        if frac.is_empty() { "0" } else { frac },
        exp
    )
    .parse()
    .ok()
}

/// An entity's arguments, by the name of each of its parts. A typed value (`LENGTH_MEASURE(1.)`)
/// is read as its value.
pub fn arguments(text: &[u8]) -> R<Parts> {
    let toks = tokens(text)?;
    let mut r = Reader { text, toks, pos: 0 };
    let mut parts: Parts = Vec::new();
    if !r.toks.is_empty() && r.toks[0] == b"(" {
        r.pos = 1;
        while r.tok(r.pos)? != b")" {
            let (name, args) = r.part()?;
            match parts.iter_mut().find(|(n, _)| *n == name) {
                Some(p) => p.1 = args,
                None => parts.push((name, args)),
            }
        }
        return Ok(parts);
    }
    parts.push(r.part()?);
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entitys_arguments() {
        let a = arguments(b"ADVANCED_FACE('',(#13,#14),#15,.T.)").unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].0, "ADVANCED_FACE");
        assert_eq!(
            a[0].1,
            Val::List(vec![
                Val::Str("".into()),
                Val::List(vec![Val::Ref(13), Val::Ref(14)]),
                Val::Ref(15),
                Val::Enum("T".into())
            ])
        );
        let a = arguments(b"( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )").unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(
            a[2].1,
            Val::List(vec![Val::Enum("MILLI".into()), Val::Enum("METRE".into())])
        );
        let a = arguments(b"X(LENGTH_MEASURE(1.E-3), -2.5, 3, +.5e1, 1.e2)").unwrap();
        assert_eq!(
            a[0].1,
            Val::List(vec![
                Val::Num(0.001),
                Val::Num(-2.5),
                Val::Num(3.0),
                Val::Num(5.0),
                Val::Num(100.0)
            ])
        );
    }

    #[test]
    fn the_files_entities() {
        let e = parse(b"ISO-10303-21;\nHEADER;\nFILE_NAME('a;b');\nENDSEC;\nDATA;\n#1 = A('x''y;');\n/* c; */#2=(B() C(#1));\nENDSEC;\n");
        assert_eq!(e.list.len(), 2);
        assert_eq!(e.list[0].text, b"A('')");
        assert_eq!(e.list[1].names, vec!["B", "C"]);
    }
}
