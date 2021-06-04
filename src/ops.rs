//! Operations you can perform to combine two SDFs.
use super::*;
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
#[derive(Clone, Copy, Debug)]
pub struct HardMin<T> {
    _pd: std::marker::PhantomData<T>,
}

impl<T> Default for HardMin<T> {
    fn default() -> Self {
        HardMin {
            _pd: std::marker::PhantomData,
        }
    }
}

impl<T: MaxMin> MinFunction<T> for HardMin<T> {
    #[inline]
    fn min(&self, a: T, b: T) -> T {
        a.min(b)
    }
}

/// Takes the minimum of two values, smoothing between them
/// when they are close.
///
/// This uses an exponential function to smooth between the two
/// values, and `k` controls the radius/distance of the
/// smoothing. 32 is a good default value for `k` for this
/// smoothing function.
#[derive(Clone, Copy, Debug)]
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
where
    T: Copy
        + Neg<Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Div<T, Output = T>
        + Log2
        + Exp2,
{
    #[inline]
    fn min(&self, a: T, b: T) -> T {
        let res = (-self.k * a).exp2() + (-self.k * b).exp2();
        -res.log2() / self.k
    }
}

/// Takes the minimum of two values, smoothing between them
/// when they are close.
///
/// This uses a polynomial function to smooth between the two
/// values, and `k` controls the radius/distance of the
/// smoothing. 0.1 is a good default value for `k` for this
/// smoothing function.
#[derive(Clone, Copy, Debug)]
pub struct PolySmoothMin<T> {
    pub k: T,
}

impl<T> PolySmoothMin<T> {
    pub fn new(k: T) -> Self {
        PolySmoothMin { k }
    }
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
where
    T: Neg<Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Div<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + PointFive
        + One
        + Zero
        + Clamp
        + Copy,
{
    #[inline]
    fn min(&self, a: T, b: T) -> T {
        let t = T::point_five() + T::point_five() * (b - a) / self.k;
        let h = t.clamp(T::zero(), T::one());
        b.lerp(a, h) - self.k * h * (T::one() - h)
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

impl<T, S1, S2> Union<T, S1, S2, HardMin<T>> {
    pub fn hard(sdf1: S1, sdf2: S2) -> Self {
        Union {
            sdf1,
            sdf2,
            min_func: HardMin::default(),
            _pd: std::marker::PhantomData,
        }
    }
}

impl<T, S1, S2> Union<T, S1, S2, PolySmoothMin<T>> {
    pub fn smooth(sdf1: S1, sdf2: S2, smoothness: T) -> Self {
        Union {
            sdf1,
            sdf2,
            min_func: PolySmoothMin::new(smoothness),
            _pd: std::marker::PhantomData,
        }
    }
}

impl<T, S1, S2, M> Union<T, S1, S2, M>
where
    M: MinFunction<T>,
{
    pub fn new(sdf1: S1, sdf2: S2, min_func: M) -> Self {
        Union {
            sdf1,
            sdf2,
            min_func,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<T, V, S1, S2, M> SDF<T, V> for Union<T, S1, S2, M>
where
    T: Copy,
    V: Vec<T>,
    S1: SDF<T, V>,
    S2: SDF<T, V>,
    M: MinFunction<T> + Copy,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        self.min_func.min(self.sdf1.dist(p), self.sdf2.dist(p))
    }
}

/// Get the subtraction of two SDFs. Note that this operation is *not* commutative,
/// i.e. `Subtraction::new(a, b) =/= Subtraction::new(b, a)`.
#[derive(Clone, Copy, Debug)]
pub struct Subtraction<S1, S2> {
    pub sdf1: S1,
    pub sdf2: S2,
}

impl<S1, S2> Subtraction<S1, S2> {
    /// Get the subtraction of two SDFs. Note that this operation is *not* commutative,
    /// i.e. `Subtraction::new(a, b) =/= Subtraction::new(b, a)`.
    pub fn new(sdf1: S1, sdf2: S2) -> Self {
        Subtraction { sdf1, sdf2 }
    }
}

impl<T, V, S1, S2> SDF<T, V> for Subtraction<S1, S2>
where
    T: Copy + Neg<Output = T> + MaxMin,
    V: Vec<T>,
    S1: SDF<T, V>,
    S2: SDF<T, V>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        (-self.sdf1.dist(p)).max(self.sdf2.dist(p))
    }
}

/// Get the intersection of two SDFs.
#[derive(Clone, Copy, Debug)]
pub struct Intersection<S1, S2> {
    pub sdf1: S1,
    pub sdf2: S2,
}

impl<S1, S2> Intersection<S1, S2> {
    /// Get the intersection of two SDFs.
    pub fn new(sdf1: S1, sdf2: S2) -> Self {
        Intersection { sdf1, sdf2 }
    }
}

impl<T, V, S1, S2> SDF<T, V> for Intersection<S1, S2>
where
    T: Copy + MaxMin,
    V: Vec<T>,
    S1: SDF<T, V>,
    S2: SDF<T, V>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        self.sdf1.dist(p).max(self.sdf2.dist(p))
    }
}
