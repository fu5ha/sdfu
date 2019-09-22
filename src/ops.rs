//! Operations you can perform to combine two SDFs.
use super::*;
use crate::mathtypes::*;
use std::ops::*;

/// The union of two SDFs.
#[derive(Clone, Copy, Debug)]
pub struct Union<S1, S2, M> {
    pub sdf1: S1,
    pub sdf2: S2,
    pub min_func: M,
}

/// A function which can get the minimum between two SDFs.
/// This is useful because sometimes we want to be able to
/// interpolate between the minimums for 'soft blending' between
/// two SDFs.
pub trait MinFunction<T> {
    fn min(&self, a: T, b: T) -> T;
}

impl<S1, S2, M> Union<S1, S2, M> {
    pub fn new(sdf1: S1, sdf2: S2, min_func: M) -> Self { Union { sdf1, sdf2, min_func } }
}

impl<T, V, S1, S2, M> SDF<T, V> for Union<S1, S2, M>
where V: Vec<T>,
    S1: SDF<T, V>,
    S2: SDF<T, V>,
    M: MinFunction<T> + Copy,
{
    fn dist(&self, p: V) -> T {
        self.min_func.min(self.sdf1.dist(p), self.sdf2.dist(p))
    }
}
pub fn union<T: MaxMin>(dist1: T, dist2: T) -> T {
    dist1.min(dist2)
}

/// Get the subtracion of two SDFs. Note that this operation is *not* commutative,
/// i.e. `subtraction(a, b) =/= subtracion(b, a)`.
pub fn subtraction<T: Neg<Output=T> + MaxMin>(dist1: T, dist2: T) -> T {
    -dist1.max(dist2)
}

/// Get the intersection of two SDFs.
pub fn intersection<T: MaxMin>(dist1: T, dist2: T) -> T {
    dist1.max(dist2)
}
