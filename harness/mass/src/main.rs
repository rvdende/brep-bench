//! The mass properties of the solid in a STEP file (ISO 10303-21), from its exact geometry: its
//! volume, its area, its centre of gravity and its moments of inertia. No crates.
//!
//!     mass FILE.step     # prints them as JSON
//!
//! The harness grades a kernel on the STEP file it writes (SPEC.md, "STEP is the answer"), so the
//! file's own solid is measured here, by the harness, the same way for every kernel, and not by
//! any one kernel's reader. harness/run.py's `mass_of` runs it, and builds it first where it is not
//! built or is older than its sources
//! (`cargo build --release --manifest-path harness/mass/Cargo.toml`).
//!
//! How. By the divergence theorem each measure of the solid is an integral over its boundary of a
//! polynomial in the position times the outward normal: the volume is ∮ (x·n)/3 dA, its first
//! moment ∫ x dV is ∮ (x²/2) n_x dA, its second moment ∫ x² dV is ∮ (x³/3) n_x dA, and so on. Each
//! face is a surface S(u, v) trimmed by loops of edges, and its integral ∬ F(S, S_u × S_v) du dv
//! over the trimmed region of the (u, v) plane is, by Green's theorem, -∮ H du round the loops,
//! where H(u, v) = ∫ F(u, s) ds from a fixed v_ref to v. Both integrals are taken by Gauss
//! quadrature, the one round the loops adaptively (Gauss-Kronrod), to about 1e-12 of the solid's
//! size.
//!
//! The loops are put on the (u, v) plane by inverting the points of each edge's curve in space
//! onto the face's surface (closed forms for the analytic surfaces, Newton's method on a
//! B-spline), never by the file's own curves in (u, v), which many writers leave out and some
//! write loosely. A periodic parameter (an angle) is unwrapped along each loop, so that a loop
//! round a cylinder or a torus is followed as it winds. Where an edge passes through a singular
//! point of its surface (the pole of a sphere, the apex of a cone) the loop runs along the
//! singular line in (u, v) from the direction it arrives in to the one it leaves in. What a
//! periodic face's loops leave open (whether a face round a sphere holds the pole) is settled by
//! the face's area in (u, v), which must be positive and no more than the whole surface's.
//!
//! A face's loops are taken as ISO 10303-42 directs them: anticlockwise seen from where the face's
//! normal points, the face's normal its surface's turned by `same_sense`. Where a face's normal is
//! against its surface's, the answer also has `by_surface`: the solid as it would be with the
//! loops directed by the surfaces' normals, as some writers direct them; the harness reads it only
//! to say why a file fails.
//!
//! Supported: planes, cylinders, cones, spheres, tori (degenerate too), B-spline surfaces
//! (rational too), surfaces of revolution and of linear extrusion; lines, circles, ellipses,
//! B-spline curves (rational too), polylines and trimmed curves; MANIFOLD_SOLID_BREP,
//! BREP_WITH_VOIDS and FACETED_BREP; lengths in any unit, given in millimetres. Anything else is
//! said, as an error.
//!
//! It was first written in Python (harness/mass.py, since removed) and then in Rust step for step,
//! and still gives the Python measurer's answers digit for digit: it keeps Python's floored
//! remainder, rounding, hypot and printing (py.rs), and sums in the same order. A change that
//! alters an answer's last digits is a change to how every kernel is measured.
//!
//! The faces are measured several at once (`MASS_THREADS`, by default as many as the machine has
//! cores, eight at most) and summed in the file's order, so the answer does not depend on how
//! many.
#![allow(clippy::needless_range_loop)]

mod curve;
mod face;
mod geom;
mod loops;
mod model;
mod py;
mod step;
mod surface;

use curve::Frame;
use face::{Face, Settled};
use geom::*;
use loops::{Loop, face_loops, newell};
use model::Model;
use py::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use step::Val;
use surface::Surface;

const SOLIDS: [&str; 3] = ["MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS", "FACETED_BREP"];

/// What `json.dumps` writes.
enum Json {
    Num(f64),
    Int(usize),
    Str(String),
    List(Vec<Json>),
    Obj(Vec<(&'static str, Json)>),
}

impl Json {
    fn dump(&self, out: &mut String) {
        match self {
            Json::Num(x) => out.push_str(&float_json(*x)),
            Json::Int(n) => out.push_str(&n.to_string()),
            Json::Str(s) => out.push_str(&str_json(s)),
            Json::List(v) => {
                out.push('[');
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    x.dump(out);
                }
                out.push(']');
            }
            Json::Obj(v) => {
                out.push('{');
                for (i, (k, x)) in v.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&str_json(k));
                    out.push_str(": ");
                    x.dump(out);
                }
                out.push('}');
            }
        }
    }
}

/// Each face of the file's solids: (face entity, its shell's orientation); and how many solids.
fn faces_of(m: &Model) -> R<(Vec<(Val, bool)>, usize)> {
    let mut out = Vec::new();
    let solids: Vec<u64> = m
        .entities
        .list
        .iter()
        .filter(|e| e.names.iter().any(|n| SOLIDS.contains(&n.as_str())))
        .map(|e| e.id)
        .collect();
    if solids.is_empty() {
        return unsupported("no solid in the file");
    }
    for r in &solids {
        let a = m.args_of(*r)?;
        let Some((name, args)) = a.iter().find(|(n, _)| SOLIDS.contains(&n.as_str())) else {
            return Err(Err::Uncaught("StopIteration", String::new()));
        };
        let mut shells = vec![(args.at(1)?.clone(), true)];
        if name == "BREP_WITH_VOIDS" {
            for v in args.at(2)?.items()? {
                shells.push((v.clone(), true));
            }
        }
        for (sh, mut sense) in shells {
            let mut sa = m.one(&sh)?;
            if sa.is("ORIENTED_CLOSED_SHELL") {
                sense = sense == !sa.at(3)?.is("F");
                sa = m.one(sa.at(2)?)?;
            }
            if !(sa.is("CLOSED_SHELL") || sa.is("OPEN_SHELL")) {
                return unsupported(format!("a shell {}", sa.named()));
            }
            for f in sa.at(1)?.items()? {
                out.push((f.clone(), sense));
            }
        }
    }
    Ok((out, solids.len()))
}

/// The middle and the size of a set of points' box.
fn extent(pts: &[V3]) -> (V3, V3) {
    let (mut lo, mut hi) = (pts[0], pts[0]);
    for p in &pts[1..] {
        for i in 0..3 {
            lo[i] = pymin(lo[i], p[i]);
            hi[i] = pymax(hi[i], p[i]);
        }
    }
    (lo, hi)
}

fn bounding(m: &Model) -> R<(V3, f64)> {
    // every point in space: vertices, centres, control points (not the (u, v) points of curves
    // on surfaces, which have two coordinates)
    let mut pts = Vec::new();
    for e in &m.entities.list {
        // (an entity of no name at all has no first name: Python's IndexError)
        let Some(name) = e.names.first() else {
            return index_error();
        };
        if name == "CARTESIAN_POINT" {
            let r = Val::Ref(e.id);
            if m.one(&r)?.at(1)?.len()? == 3 {
                pts.push(m.point(&r)?);
            }
        }
    }
    if pts.is_empty() {
        return unsupported("no points");
    }
    let (lo, hi) = extent(&pts);
    Ok((mul(add(lo, hi), 0.5), pymax(norm(sub(hi, lo)), 1e-9)))
}

/// The error with the face it is of, as `raise type(e)(f"face #{f}: {e}")` gives it.
fn of_face(e: Err, f: &Val) -> Err {
    match e {
        Err::Unsupported(s) => Err::Unsupported(format!("face #{}: {}", f.str(), s)),
        Err::Invalid(s) => Err::Invalid(format!("face #{}: {}", f.str(), s)),
        e => e,
    }
}

struct Found {
    f: Val,
    s: Arc<Surface>,
    loops: Vec<Loop>,
    same: bool,
    sense: bool,
}

/// The centre of gravity and the inertia tensor about it, of unit density, from a solid's
/// volume and its first and second moments about `origin` (harness/stl.py's `about_centroid`).
fn about_centroid(
    volume: f64,
    m1: [f64; 3],
    m2: [[f64; 3]; 3],
    origin: V3,
) -> ([f64; 3], [[f64; 3]; 3]) {
    let g = [m1[0] / volume, m1[1] / volume, m1[2] / volume];
    let mut s2 = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            s2[i][j] = m2[i][j] - volume * g[i] * g[j];
        }
    }
    let tr = s2[0][0] + s2[1][1] + s2[2][2];
    let mut inertia = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            inertia[i][j] = if i == j { tr - s2[i][i] } else { -s2[i][j] };
        }
    }
    (
        [origin[0] + g[0], origin[1] + g[1], origin[2] + g[2]],
        inertia,
    )
}

/// The mass properties from a solid's summed integrals, or {"error": why}.
fn props(
    total: &[f64; NM],
    failed: Option<&String>,
    centre: V3,
    nsolids: usize,
    nfaces: usize,
) -> Vec<(&'static str, Json)> {
    if let Some(failed) = failed {
        return vec![("error", Json::Str(format!("cannot measure: {}", failed)))];
    }
    let v = total[2];
    if v == 0.0 {
        return vec![(
            "error",
            Json::Str("cannot measure: the solid has no volume".into()),
        )];
    }
    let m2 = [
        [total[6], total[9], total[11]],
        [total[9], total[7], total[10]],
        [total[11], total[10], total[8]],
    ];
    let (g, inertia) = about_centroid(v, [total[3], total[4], total[5]], m2, centre);
    vec![
        ("volume", Json::Num(v)),
        ("area", Json::Num(total[1])),
        (
            "centroid",
            Json::List(g.iter().map(|x| Json::Num(*x)).collect()),
        ),
        (
            "inertia",
            Json::List(
                inertia
                    .iter()
                    .map(|r| Json::List(r.iter().map(|x| Json::Num(*x)).collect()))
                    .collect(),
            ),
        ),
        ("solids", Json::Int(nsolids)),
        ("faces", Json::Int(nfaces)),
    ]
}

type Measured = R<(Settled, Option<Settled>)>;

/// Each face measured, `threads` at once, in the faces' order. A face after one that could not
/// be measured may be left out (`None`): Python stops at the first.
fn measure_faces(
    jobs: Vec<(Arc<Surface>, Vec<Loop>, bool)>,
    centre: V3,
    scale: f64,
    threads: usize,
) -> Vec<Option<Measured>> {
    let n = jobs.len();
    let threads = threads.clamp(1, n.max(1));
    if threads == 1 {
        let failed = AtomicUsize::new(usize::MAX);
        let mut out = Vec::with_capacity(n);
        for (s, loops, same) in jobs {
            let got = Face::measure(s, loops, same, centre, scale, out.len(), &failed);
            let stop = got.is_err();
            out.push(Some(got));
            if stop {
                break;
            }
        }
        out.resize_with(n, || None);
        return out;
    }
    let jobs: Vec<Mutex<Option<(Arc<Surface>, Vec<Loop>, bool)>>> =
        jobs.into_iter().map(|j| Mutex::new(Some(j))).collect();
    let results: Vec<Mutex<Option<Measured>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let failed = AtomicUsize::new(usize::MAX);
    let work = || {
        loop {
            let k = next.fetch_add(1, Ordering::SeqCst);
            if k >= n {
                break;
            }
            if k > failed.load(Ordering::SeqCst) {
                continue;
            }
            let (s, loops, same) = jobs[k].lock().unwrap().take().unwrap();
            let got = Face::measure(s, loops, same, centre, scale, k, &failed);
            if got.is_err() {
                failed.fetch_min(k, Ordering::SeqCst);
            }
            *results[k].lock().unwrap() = Some(got);
        }
    };
    std::thread::scope(|scope| {
        // this thread measures too; the others are given its stack's size (a thread that cannot
        // be started is done without)
        for _ in 1..threads {
            let _ = std::thread::Builder::new()
                .stack_size(8 << 20)
                .spawn_scoped(scope, work);
        }
        work();
    });
    results
        .into_iter()
        .map(|r| r.into_inner().unwrap())
        .collect()
}

/// The mass properties of the solid in a STEP file's text: {volume, area, centroid, inertia
/// (about the centroid), solids, faces}, lengths in millimetres, or {"error": why}; and, where
/// it differs, `by_surface`.
fn measure(text: &[u8], threads: usize) -> R<Json> {
    let m = Model::new(text)?;
    let (faces, nsolids) = faces_of(&m)?;
    let (mut centre, mut scale) = bounding(&m)?;
    // each face's loops on its surface first
    let mut found: Vec<Found> = Vec::with_capacity(faces.len());
    for (f, sense) in &faces {
        let fa = m.one(f)?;
        let (s, same);
        if fa.is("ADVANCED_FACE") || fa.is("FACE_SURFACE") {
            s = m.surface(fa.at(2)?)?;
            same = !fa.at(3)?.is("F");
        } else if fa.is("FACE") {
            // a faceted B-rep's polygon: on the plane its points span
            let bounds = fa.at(1)?;
            let la = m.one(m.one(bounds.at(0)?)?.at(1)?)?;
            let mut pts = Vec::new();
            for p in la.at(1)?.items()? {
                pts.push(m.point(p)?);
            }
            let z = unit(newell(&pts))?;
            if pts.len() < 2 {
                return index_error();
            }
            let x = unit(sub(pts[1], pts[0]))?;
            let x = unit(sub(x, mul(z, dot(x, z))))?;
            s = Arc::new(Surface::plane(Frame {
                o: pts[0],
                x,
                y: cross(z, x),
                z,
            }));
            same = true;
        } else {
            return unsupported(format!("a face {}", fa.named()));
        }
        let loops = face_loops(&m, f, &s, scale).map_err(|e| of_face(e, f))?;
        found.push(Found {
            f: f.clone(),
            s,
            loops,
            same,
            sense: *sense,
        });
    }
    // the measures about the middle of the solid's own edges, their tolerances by its size: the
    // file's other points (a sphere's centre, a placement's origin) may lie far off a small solid
    // whose measures are the small differences of its faces'
    let mut edge = Vec::new();
    for fd in &found {
        for (ps, _) in &fd.loops {
            for p in ps.iter().filter(|p| p.line.is_none()) {
                let c = p.c.as_ref().unwrap();
                for (t, _) in p.rf.iter().step_by(4) {
                    edge.push(c.point(*t)?);
                }
            }
        }
    }
    if !edge.is_empty() && found.iter().all(|fd| !fd.loops.is_empty()) {
        let (lo, hi) = extent(&edge);
        if norm(sub(hi, lo)) > 0.0 {
            (centre, scale) = (mul(add(lo, hi), 0.5), norm(sub(hi, lo)));
        }
    }
    // the solid as the standard reads it, and as a writer who directs a face's loops by its
    // surface's normal would mean it (where any face's normal is against its surface's)
    let against = found.iter().any(|fd| !fd.same);
    let mut names = Vec::with_capacity(found.len());
    let mut jobs = Vec::with_capacity(found.len());
    for fd in found {
        names.push((fd.f, fd.sense));
        jobs.push((fd.s, fd.loops, fd.same));
    }
    drop(m);
    let measured = measure_faces(jobs, centre, scale, threads);
    let mut totals = [[0.0; NM]; 2];
    let mut failed: [Option<String>; 2] = [None, None];
    for ((f, sense), got) in names.iter().zip(measured) {
        let (standard, other) = match got {
            Some(Ok(parts)) => parts,
            Some(Err(e)) => return Err(of_face(e, f)),
            None => unreachable!("a face left unmeasured before any failed"),
        };
        let sign = if *sense { 1.0 } else { -1.0 };
        let other = other.unwrap_or_else(|| standard.clone());
        for (key, part) in [standard, other].into_iter().enumerate() {
            let part = match part {
                Ok(part) => part,
                Err(why) => {
                    if failed[key].is_none() {
                        failed[key] = Some(format!("face #{}: {}", f.str(), why));
                    }
                    continue;
                }
            };
            let total = &mut totals[key];
            total[1] += part[1];
            for i in 2..NM {
                total[i] += sign * part[i];
            }
        }
    }
    let nfaces = faces.len();
    let mut out = props(&totals[0], failed[0].as_ref(), centre, nsolids, nfaces);
    if against {
        out.push((
            "by_surface",
            Json::Obj(props(
                &totals[1],
                failed[1].as_ref(),
                centre,
                nsolids,
                nfaces,
            )),
        ));
    }
    Ok(Json::Obj(out))
}

/// The file as Python reads its text: every line's end a `\n`.
fn newlines(data: Vec<u8>) -> Vec<u8> {
    if !data.contains(&b'\r') {
        return data;
    }
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if data[i] == b'\r' {
            out.push(b'\n');
            if data.get(i + 1) == Some(&b'\n') {
                i += 1;
            }
        } else {
            out.push(data[i]);
        }
        i += 1;
    }
    out
}

/// Python's last word, where it would end with a traceback: no answer, and failure.
fn uncaught(kind: &str, why: &str) -> ! {
    eprintln!("Traceback (most recent call last):");
    if why.is_empty() {
        eprintln!("{}", kind);
    } else {
        eprintln!("{}: {}", kind, why);
    }
    std::process::exit(1);
}

/// `measure` of the STEP file at `path`, or {"invalid": why} where the file holds no solid
/// (whoever reads it), or {"error": why} where the measurer cannot measure it.
fn read(path: &str, threads: usize) -> Json {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            let (kind, errno, what) = match e.kind() {
                std::io::ErrorKind::NotFound => {
                    ("FileNotFoundError", 2, "No such file or directory")
                }
                std::io::ErrorKind::PermissionDenied => {
                    ("PermissionError", 13, "Permission denied")
                }
                std::io::ErrorKind::IsADirectory => ("IsADirectoryError", 21, "Is a directory"),
                _ => uncaught("OSError", &format!("{}: {}", e, str_repr(path))),
            };
            uncaught(
                kind,
                &format!("[Errno {}] {}: {}", errno, what, str_repr(path)),
            );
        }
    };
    match measure(&newlines(data), threads) {
        Ok(j) => j,
        Err(Err::Invalid(why)) => Json::Obj(vec![("invalid", Json::Str(why))]),
        Err(Err::Unsupported(why)) => Json::Obj(vec![(
            "error",
            Json::Str(format!("cannot measure: {}", why)),
        )]),
        Err(Err::Py(kind, why)) => {
            Json::Obj(vec![("error", Json::Str(format!("{}: {}", kind, why)))])
        }
        Err(Err::Uncaught(kind, why)) => uncaught(kind, &why),
    }
}

fn main() {
    let threads = match std::env::var("MASS_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(n) => n.max(1),
        None => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(8),
    };
    for p in std::env::args().skip(1) {
        let mut out = String::new();
        read(&p, threads).dump(&mut out);
        println!("{}", out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube as a faceted B-rep, in metres: a thousand millimetres a side.
    #[test]
    fn a_faceted_cube() {
        let mut s = String::from("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n");
        s += "#1=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));\n";
        let pts = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ];
        for (i, p) in pts.iter().enumerate() {
            s += &format!(
                "#{}=CARTESIAN_POINT('',({}.,{}.,{}.));\n",
                10 + i,
                p.0,
                p.1,
                p.2
            );
        }
        let faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ];
        for (i, f) in faces.iter().enumerate() {
            s += &format!(
                "#{}=POLY_LOOP('',(#{},#{},#{},#{}));\n",
                20 + i,
                10 + f[0],
                10 + f[1],
                10 + f[2],
                10 + f[3]
            );
            s += &format!("#{}=FACE_OUTER_BOUND('',#{},.T.);\n", 30 + i, 20 + i);
            s += &format!("#{}=FACE('',(#{}));\n", 40 + i, 30 + i);
        }
        s += "#50=CLOSED_SHELL('',(#40,#41,#42,#43,#44,#45));\n#51=FACETED_BREP('',#50);\nENDSEC;\nEND-ISO-10303-21;\n";
        for threads in [1, 3] {
            let mut out = String::new();
            measure(s.as_bytes(), threads).ok().unwrap().dump(&mut out);
            let number = |key: &str| -> f64 {
                let at = out.find(key).unwrap() + key.len();
                out[at..]
                    .split([',', '}'])
                    .next()
                    .unwrap()
                    .trim()
                    .parse()
                    .unwrap()
            };
            assert!(
                (number("\"volume\": ") / 1e9 - 1.0).abs() < 1e-12,
                "{}",
                out
            );
            assert!((number("\"area\": ") / 6e6 - 1.0).abs() < 1e-12, "{}", out);
            assert!(
                out.contains("\"centroid\": [500.0, 500.0, 500.0]"),
                "{}",
                out
            );
            assert!(out.ends_with("\"solids\": 1, \"faces\": 6}"), "{}", out);
        }
    }
}
