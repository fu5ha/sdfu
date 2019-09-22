//! Operations you can perform to combine two SDFs.
use super::*;
use crate::mathtypes::*;
use std::ops::*;

/// A function which can get the minimum between two SDFs.
/// This is useful because sometimes we want to be able to
/// interpolate between the minimums for 'soft blending' between
/// two SDFs.
pub trait MinFunction<T> {
    fn min(&self, a: T, b: T) -> T;
}

/// Takes the absolute minimum of two values
/// and returns them directly. A standard min function.
pub struct HardMin<T> {
    _pd: std::marker::PhantomData<T>,
}

impl<T> Default for HardMin<T> {
    fn default() -> Self {
        HardMin { _pd: std::marker::PhantomData }
    }
}

impl<T: MaxMin> MinFunction<T> for HardMin<T> {
    fn min(&self, a: T, b: T) -> T {
        a.min(b)
    }
}

/// Takes the minimum of two values, smoothing between them
/// when they are close. 
/// 
/// This uses an exponential function to smooth between the two
/// values, and `k` controls the radius/distance of the
/// smoothing.
pub struct ExponentialSmoothMin<T> {
    pub k: T,
}

impl Default for ExponentialSmoothMin<f32> {
    fn default() -> Self {
        ExponentialSmoothMin { k: 32.0 }
    }
}

impl Default for ExponentialSmoothMin<f64> {
    fn default() -> Self {
        ExponentialSmoothMin { k: 32.0 }
    }
}

impl<T> MinFunction<T> for ExponentialSmoothMin<T> 
where T: Copy + Neg<Output=T> + Mul<T, Output=T> + Add<T, Output=T> + Div<T, Output=T> + Log2 + Exp2 {
    fn min(&self, a: T, b: T) -> T {
        let res = (-self.k * a).exp2() + (-self.k * b).exp2();
        -res.log2() / self.k
    }
}

/// Takes the minimum of two values, smoothing between them
/// when they are close. 
/// 
/// This uses an exponential function to smooth between the two
/// values, and `k` controls the radius/distance of the
/// smoothing.
pub struct PolySmoothMin<T> {
    pub k: T,
}

impl Default for PolySmoothMin<f32> {
    fn default() -> Self {
        PolySmoothMin { k: 0.1 }
    }
}

impl Default for PolySmoothMin<f64> {
    fn default() -> Self {
        PolySmoothMin { k: 0.1 }
    }
}

impl<T> MinFunction<T> for PolySmoothMin<T> 
where T: Neg<Output=T> + Mul<T, Output=T> + Add<T, Output=T>
        + Div<T, Output=T> + Sub<T, Output=T> + Mul<T, Output=T>
        + PointFive + One + Zero + Clamp + Copy
{
    fn min(&self, a: T, b: T) -> T {
        let t = T::POINT_FIVE + T::POINT_FIVE * (b - a) / self.k;
        let h = t.clamp(T:: ZERO, T::ONE);
        b.lerp(a, h) - self.k * h * (T::ONE - h)
    }
}


/// The union of two SDFs.
#[derive(Clone, Copy, Debug)]
pub struct Union<T, S1, S2, M> {
    pub sdf1: S1,
    pub sdf2: S2,
    pub min_func: M,
    _pd: std::marker::PhantomData<T>,
}

impl<T, S1, S2> Union<T, S1, S2, HardMin<T>>
{
    pub fn hard(sdf1: S1, sdf2: S2) -> Self { Union { sdf1, sdf2, min_func: HardMin::default(), _pd: std::marker::PhantomData } }
}

impl<T, S1, S2, M> Union<T, S1, S2, M>
where M: MinFunction<T> + Default
{
    pub fn new(sdf1: S1, sdf2: S2) -> Self { Union { sdf1, sdf2, min_func: M::default(), _pd: std::marker::PhantomData } }
}

impl<T, S1, S2, M> Union<T, S1, S2, M>
where M: MinFunction<T>
{
    pub fn new_with(sdf1: S1, sdf2: S2, min_func: M) -> Self { Union { sdf1, sdf2, min_func, _pd: std::marker::PhantomData } }
}

impl<T, V, S1, S2, M> SDF<T, V> for Union<T, S1, S2, M>
where T: Copy,
    V: Vec<T>,
    S1: SDF<T, V>,
    S2: SDF<T, V>,
    M: MinFunction<T> + Copy,
{
    fn dist(&self, p: V) -> T {
        self.min_func.min(self.sdf1.dist(p), self.sdf2.dist(p))
    }
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
