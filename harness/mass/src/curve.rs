//! Curves: point(t), deriv(t), invert(p), `period` (or none), `range` (t0, t1), `breaks`.
use crate::geom::*;
use crate::py::*;
use std::sync::{Arc, OnceLock};

unsafe extern "C" {
    // the C library's, as Python's math.asinh is
    safe fn asinh(x: f64) -> f64;
}

pub struct Frame {
    pub o: V3,
    pub x: V3,
    pub y: V3,
    pub z: V3,
}

pub struct BSplineCurve {
    pub p: usize,
    pub pts: Vec<V3>,
    pub w: Option<Vec<f64>>,
    pub u: Vec<f64>,
    pub n: usize,
    pub range: (f64, f64),
    pub knots: Vec<f64>,
    pub breaks: Vec<f64>,
    pub period: Option<f64>,
    samples: OnceLock<Vec<(f64, V3)>>,
}

pub struct Polyline {
    pub pts: Vec<V3>,
    pub range: (f64, f64),
    pub breaks: Vec<f64>,
}

pub enum Curve {
    Line {
        p: V3,
        d: V3,
        dd: f64,
    },
    /// A circle or an ellipse: O + a cos t X + b sin t Y.
    Conic {
        f: Frame,
        a: f64,
        b: f64,
    },
    /// A parabola, O + f (t² X + 2t Y), or a hyperbola, O + a cosh t X + b sinh t Y.
    Open {
        f: Frame,
        parabola: bool,
        a: f64,
        b: f64,
    },
    BSpline(BSplineCurve),
    Polyline(Polyline),
    /// A curve run the other way: a trimmed curve whose sense is false.
    Reversed(Arc<Curve>, Vec<f64>),
}

const INF: f64 = f64::INFINITY;

impl BSplineCurve {
    pub fn new(p: i64, pts: Vec<V3>, w: Option<Vec<f64>>, u: Vec<f64>) -> R<BSplineCurve> {
        let n = pts.len() as i64 - 1;
        if u.len() as i64 != n + p + 2 {
            return unsupported("a B-spline curve whose knots do not match its control points");
        }
        if p < 0 || n < p || w.as_ref().is_some_and(|w| w.len() < pts.len()) {
            // (Python reads its lists from their ends there, or past them)
            return index_error();
        }
        let (p, n) = (p as usize, n as usize);
        let range = (u[p], u[n + 1]);
        let knots = sorted_set(u[p..n + 2].iter().copied());
        let mut c = BSplineCurve {
            p,
            pts,
            w,
            u,
            n,
            range,
            breaks: if knots.len() > 2 {
                knots[1..knots.len() - 1].to_vec()
            } else {
                Vec::new()
            },
            knots,
            period: None,
            samples: OnceLock::new(),
        };
        // a closed curve is followed round as periodic, its join a break (a kink, if it is not
        // smooth there)
        let gap = norm(sub(c.eval(range.0)?.0, c.eval(range.1)?.0));
        let size = c
            .pts
            .iter()
            .map(|a| norm(sub(*a, c.pts[0])))
            .fold(f64::NEG_INFINITY, pymax);
        let closed = gap < 1e-9 * (1.0 + size);
        let period = range.1 - range.0;
        if closed && period != 0.0 {
            c.period = Some(period);
            let t0 = range.0;
            let mut all = c.breaks.clone();
            for k in -2..4 {
                all.push(t0 + k as f64 * period);
            }
            for b in &c.breaks {
                for k in [-1.0, 1.0, 2.0] {
                    all.push(b + k * period);
                }
            }
            c.breaks = sorted_set(all);
        }
        Ok(c)
    }

    fn wrap(&self, t: f64) -> R<f64> {
        match self.period {
            Some(per) => Ok(self.range.0 + pymod(t - self.range.0, per)?),
            None => Ok(t),
        }
    }

    /// The point at `t` and the derivative there.
    pub fn eval(&self, t: f64) -> R<(V3, V3)> {
        let (p, u) = (self.p, &self.u[..]);
        let i = span(self.n, p, t, u);
        let (mut sn, mut sd) = ([0.0; DEG + 1], [0.0; DEG + 1]);
        let (mut hn, mut hd);
        let (nn, dd): (&mut [f64], &mut [f64]) = if p <= DEG {
            (&mut sn[..p + 1], &mut sd[..p + 1])
        } else {
            hn = vec![0.0; p + 1];
            hd = vec![0.0; p + 1];
            (&mut hn, &mut hd)
        };
        basis(i, t, p, u, nn, dd);
        match &self.w {
            None => {
                let (mut c, mut d) = ([0.0; 3], [0.0; 3]);
                for j in 0..=p {
                    let q = self.pts[i - p + j];
                    c[0] += nn[j] * q[0];
                    c[1] += nn[j] * q[1];
                    c[2] += nn[j] * q[2];
                    d[0] += dd[j] * q[0];
                    d[1] += dd[j] * q[1];
                    d[2] += dd[j] * q[2];
                }
                Ok((c, d))
            }
            Some(ws) => {
                let (mut a, mut aw, mut b, mut bw) = ([0.0; 3], 0.0, [0.0; 3], 0.0);
                for j in 0..=p {
                    let (q, w) = (self.pts[i - p + j], ws[i - p + j]);
                    let (nw, dw) = (nn[j] * w, dd[j] * w);
                    a[0] += nw * q[0];
                    a[1] += nw * q[1];
                    a[2] += nw * q[2];
                    aw += nw;
                    b[0] += dw * q[0];
                    b[1] += dw * q[1];
                    b[2] += dw * q[2];
                    bw += dw;
                }
                if aw == 0.0 {
                    return zero_division();
                }
                let c = [a[0] / aw, a[1] / aw, a[2] / aw];
                Ok((
                    c,
                    [
                        (b[0] - bw * c[0]) / aw,
                        (b[1] - bw * c[1]) / aw,
                        (b[2] - bw * c[2]) / aw,
                    ],
                ))
            }
        }
    }

    fn samples(&self) -> R<&Vec<(f64, V3)>> {
        if let Some(s) = self.samples.get() {
            return Ok(s);
        }
        let mut ts = Vec::new();
        let each = if self.knots.len() < 1000 { 4 } else { 1 }; // a long fit: a sample a piece is plenty
        for w in self.knots.windows(2) {
            let (a, b) = (w[0], w[1]);
            for k in 0..each {
                ts.push(a + (b - a) * k as f64 / each as f64);
            }
        }
        match self.knots.last() {
            Some(t) => ts.push(*t),
            None => return index_error(),
        }
        let mut out = Vec::with_capacity(ts.len());
        for t in ts {
            out.push((t, self.eval(self.wrap(t)?)?.0));
        }
        Ok(self.samples.get_or_init(|| out))
    }

    /// Curve inversion by Newton's method, from the nearest of samples or a guess.
    fn invert(&self, q: V3, guess: Option<f64>) -> R<f64> {
        let (t0, t1) = self.range;
        let mut t = match guess {
            Some(g) => g,
            None => {
                let mut best = (INF, 0.0);
                let mut first = true;
                for (t, p) in self.samples()? {
                    let e = sub(*p, q);
                    let d = dot(e, e);
                    if first || d < best.0 {
                        best = (d, *t);
                        first = false;
                    }
                }
                best.1
            }
        };
        for _ in 0..50 {
            let (p, d) = self.eval(self.wrap(t)?)?;
            let e = sub(p, q);
            let g = dot(e, d);
            let h = dot(d, d);
            if h == 0.0 {
                break;
            }
            let dt = g / h;
            let mut t2 = t - dt;
            if self.period.is_none() {
                t2 = pymin(pymax(t2, t0), t1);
            }
            if (t2 - t).abs() < 1e-15 * pymax(1.0, t.abs()) {
                t = t2;
                break;
            }
            t = t2;
        }
        Ok(t)
    }
}

impl Polyline {
    pub fn new(pts: Vec<V3>) -> R<Polyline> {
        if pts.len() < 2 {
            return unsupported("a polyline of fewer than two points");
        }
        let n = pts.len();
        Ok(Polyline {
            range: (0.0, (n - 1) as f64),
            breaks: (1..n - 1).map(|i| i as f64).collect(),
            pts,
        })
    }

    fn seg(&self, t: f64) -> R<(usize, f64)> {
        let f = integer(t, t.floor())?;
        let i = (f.max(0.0) as usize).min(self.pts.len() - 2);
        Ok((i, t - i as f64))
    }
}

impl Curve {
    pub fn line(p: V3, d: V3) -> Curve {
        Curve::Line {
            p,
            d,
            dd: dot(d, d),
        }
    }

    pub fn reversed(c: Arc<Curve>) -> Curve {
        let breaks = c.breaks().iter().map(|b| -b).collect();
        Curve::Reversed(c, breaks)
    }

    /// A polynomial's degree, else none.
    pub fn degree(&self) -> Option<usize> {
        match self {
            Curve::Line { .. } | Curve::Polyline(_) => Some(1),
            Curve::Conic { .. } | Curve::Open { .. } => None,
            Curve::BSpline(c) => {
                if c.w.is_none() {
                    Some(c.p)
                } else {
                    None
                }
            }
            Curve::Reversed(c, _) => c.degree(),
        }
    }

    /// Whether its parameter is an angle.
    pub fn angular(&self) -> bool {
        match self {
            Curve::Conic { .. } => true,
            Curve::Reversed(c, _) => c.angular(),
            _ => false,
        }
    }

    pub fn period(&self) -> Option<f64> {
        match self {
            Curve::Conic { .. } => Some(TAU),
            Curve::BSpline(c) => c.period,
            Curve::Reversed(c, _) => c.period(),
            _ => None,
        }
    }

    pub fn range(&self) -> (f64, f64) {
        match self {
            Curve::Line { .. } | Curve::Open { .. } => (-INF, INF),
            Curve::Conic { .. } => (0.0, TAU),
            Curve::BSpline(c) => c.range,
            Curve::Polyline(c) => c.range,
            Curve::Reversed(c, _) => {
                let r = c.range();
                (-r.1, -r.0)
            }
        }
    }

    pub fn breaks(&self) -> &[f64] {
        match self {
            Curve::BSpline(c) => &c.breaks,
            Curve::Polyline(c) => &c.breaks,
            Curve::Reversed(_, b) => b,
            _ => &[],
        }
    }

    /// The point at `t` and the derivative there.
    pub fn eval(&self, t: f64) -> R<(V3, V3)> {
        Ok(match self {
            Curve::Line { p, d, .. } => (add(*p, mul(*d, t)), *d),
            Curve::Conic { f, a, b } => {
                let (c, s) = (cosine(t)?, sine(t)?);
                (
                    lin3(1.0, f.o, a * c, f.x, b * s, f.y),
                    lin2(-a * s, f.x, b * c, f.y),
                )
            }
            Curve::Open { f, parabola, a, b } => {
                if *parabola {
                    (
                        lin3(1.0, f.o, a * t * t, f.x, 2.0 * a * t, f.y),
                        lin2(2.0 * a * t, f.x, 2.0 * a, f.y),
                    )
                } else {
                    let (ch, sh) = (cosh(t)?, sinh(t)?);
                    (
                        lin3(1.0, f.o, a * ch, f.x, b * sh, f.y),
                        lin2(a * sh, f.x, b * ch, f.y),
                    )
                }
            }
            Curve::BSpline(c) => c.eval(c.wrap(t)?)?,
            Curve::Polyline(c) => {
                let (i, s) = c.seg(t)?;
                let d = sub(c.pts[i + 1], c.pts[i]);
                (add(c.pts[i], mul(d, s)), d)
            }
            Curve::Reversed(c, _) => {
                let (p, d) = c.eval(-t)?;
                (p, mul(d, -1.0))
            }
        })
    }

    pub fn point(&self, t: f64) -> R<V3> {
        Ok(self.eval(t)?.0)
    }

    /// The parameter of the point of the curve nearest `q`.
    pub fn invert(&self, q: V3, guess: Option<f64>) -> R<f64> {
        match self {
            Curve::Line { p, d, dd } => Ok(if *dd != 0.0 {
                dot(sub(q, *p), *d) / dd
            } else {
                0.0
            }),
            Curve::Conic { f, a, b } => {
                let d = sub(q, f.o);
                let mut t = div(dot(d, f.y), *b)?.atan2(div(dot(d, f.x), *a)?);
                if a != b {
                    // the nearest point of an ellipse, by Newton from the circle's angle
                    for _ in 0..20 {
                        let (p, d1) = self.eval(t)?;
                        let e = sub(p, q);
                        let d2 = sub(f.o, p);
                        let g = dot(e, d1);
                        let h = dot(d1, d1) + dot(e, d2);
                        if h == 0.0 {
                            break;
                        }
                        let dt = g / h;
                        t -= dt;
                        if dt.abs() < 1e-15 {
                            break;
                        }
                    }
                }
                pymod(t, TAU)
            }
            Curve::Open { f, parabola, a, b } => {
                let d = sub(q, f.o);
                let y = dot(d, f.y);
                let mut t = if *parabola {
                    div(y, 2.0 * a)?
                } else {
                    asinh(div(y, *b)?)
                };
                for _ in 0..50 {
                    let (p, d1) = self.eval(t)?;
                    let e = sub(p, q);
                    let h = dot(d1, d1);
                    if h == 0.0 {
                        break;
                    }
                    let dt = dot(e, d1) / h;
                    t -= dt;
                    if dt.abs() < 1e-15 * (1.0 + t.abs()) {
                        break;
                    }
                }
                Ok(t)
            }
            Curve::BSpline(c) => c.invert(q, guess),
            Curve::Polyline(c) => {
                let mut best: Option<(f64, f64)> = None;
                for i in 0..c.pts.len() - 1 {
                    let (a, b) = (c.pts[i], c.pts[i + 1]);
                    let d = sub(b, a);
                    let dd = dot(d, d);
                    let s = if dd != 0.0 {
                        pymin(pymax(dot(sub(q, a), d) / dd, 0.0), 1.0)
                    } else {
                        0.0
                    };
                    let e = sub(add(a, mul(d, s)), q);
                    if best.is_none() || dot(e, e) < best.unwrap().0 {
                        best = Some((dot(e, e), i as f64 + s));
                    }
                }
                Ok(best.unwrap().1)
            }
            Curve::Reversed(c, _) => Ok(-c.invert(q, guess.map(|g| -g))?),
        }
    }
}

/// `math.cos`, `math.sin`: of what is not finite, an error.
#[inline(always)]
pub fn cosine(t: f64) -> R<f64> {
    if t.is_infinite() {
        return not_finite(t);
    }
    Ok(t.cos())
}

#[inline(always)]
pub fn sine(t: f64) -> R<f64> {
    if t.is_infinite() {
        return not_finite(t);
    }
    Ok(t.sin())
}

#[cold]
fn not_finite<T>(t: f64) -> R<T> {
    Err(Err::Py(
        "ValueError",
        format!("expected a finite input, got {}", float_repr(t)),
    ))
}

fn cosh(t: f64) -> R<f64> {
    let x = t.cosh();
    if x.is_infinite() && t.is_finite() {
        return Err(Err::Uncaught("OverflowError", "math range error".into()));
    }
    Ok(x)
}

fn sinh(t: f64) -> R<f64> {
    let x = t.sinh();
    if x.is_infinite() && t.is_finite() {
        return Err(Err::Uncaught("OverflowError", "math range error".into()));
    }
    Ok(x)
}
