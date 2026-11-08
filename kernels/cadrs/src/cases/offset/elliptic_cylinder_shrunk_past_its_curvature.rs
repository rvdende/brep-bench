//! The elliptic cylinder of `geometry/elliptic-cylinder` (x²/25 + y²/9 ≤ 1 from z = 0 to 10)
//! shrunk by 2.5, more than the ellipse's least radius of curvature (1.8, at the ends of its long
//! axis): its side moved in along its normals and trimmed where it crosses itself, meeting itself
//! in a sharp upright edge at each end.
//!
//! The ellipse sketched and extruded (as `geometry/elliptic-cylinder`), then offset inward by 2.5
//! (cadrs's offset of a solid).
use crate::bench::{extrude, sketch, xy};
use crate::shared::geometry::ellipse;
use crate::shared::offset::shrunk;
use crate::shared::sweep::first;
use cadrs_kernel::Kernel;
use crate::Refusal;
use cadrs_kernel::{BodyId, DefaultKernel};

pub fn build(k: &mut DefaultKernel) -> Result<BodyId, Refusal> {
    let s = sketch(xy(0.0), vec![ellipse((0.0, 0.0), 5.0, 3.0)]);
    let c = extrude(k, &s, 10.0)?;
    // Exactly where cadrs can; else meshed (the side's offset folds in swallowtails, which the
    // level set of the body's distance trims of itself).
    match shrunk(k, c, 2.5) {
        Ok(b) => Ok(b),
        Err(_) => first(k.offset_meshed(c, -2.5)?),
    }
}
