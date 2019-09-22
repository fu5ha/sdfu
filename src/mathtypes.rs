use std::ops::*;

pub struct XYZ<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub struct XY<T> {
    pub x: T,
    pub y: T,
}

pub trait MaxMin {
    fn max(&self, other: Self) -> Self;
    fn min(&self, other: Self) -> Self;
}

impl MaxMin for f32 {
    fn max(&self, other: Self) -> Self {
        f32::max(*self, other)
    }
    fn min(&self, other: Self) -> Self {
        f32::min(*self, other)
    }
}

impl MaxMin for f64 {
    fn max(&self, other: Self) -> Self {
        f64::max(*self, other)
    }
    fn min(&self, other: Self) -> Self {
        f64::min(*self, other)
    }
}

pub trait One {
    const ONE: Self;
}

impl One for f32 {
    const ONE: Self = 1.0;
}

impl One for f64 {
    const ONE: Self = 1.0;
}

pub trait Zero {
    const ZERO: Self;
}

impl Zero for f32 {
    const ZERO: Self = 1.0;
}

impl Zero for f64 {
    const ZERO: Self = 1.0;
}

pub trait PointFive {
    const POINT_FIVE: Self;
}

impl PointFive for f32 {
    const POINT_FIVE: Self = 0.5;
}

impl PointFive for f64 {
    const POINT_FIVE: Self = 1.0;
}

/// Linear interpolate between self and other with a factor
/// between Self::ZERO and Self::ONE.
pub trait Lerp {
    fn lerp(&self, other: Self, factor: Self) -> Self;
}

impl<T> Lerp for T
where T: Copy + Mul<T, Output=T> + Sub<T, Output=T> + Add<T, Output=T> + One
{
    fn lerp(&self, other: Self, factor: Self) -> Self {
        *self * (T::ONE - factor) + other * factor
    }
}

pub trait Clamp {
    fn clamp(&self, low: Self, high: Self) -> Self;
}

impl Clamp for f32 {
    fn clamp(&self, low: Self, high: Self) -> Self {
        self.max(low).min(high)
    }
}

impl Clamp for f64 {
    fn clamp(&self, low: Self, high: Self) -> Self {
        self.max(low).min(high)
    }
}

/// Raises `2^(self)`
pub trait Exp2 {
    fn exp2(&self) -> Self;
}

/// Returns log base 2 of self.
pub trait Log2 {
    fn log2(&self) -> Self;
}

pub trait Vec<T>: Sized + Copy + Neg<Output=Self> + Mul<T, Output=Self> + Add<Self, Output=Self> + Sub<Self, Output=Self> + MaxMin + Zero + One + Clamp {
    type Dimension: Dimension;
    type Vec2: Vec2<T>;
    type Vec3: Vec3<T>;
    fn dot(&self, other: Self) -> T;
    fn magnitude(&self) -> T;
    fn abs(&self) -> Self;
    fn normalized(&self) -> Self;
}

pub trait Vec3<T>: Vec<T> + Deref<Target=XYZ<T>> {
    fn new(x: T, y: T, z: T) -> Self;
}

pub trait Vec2<T>: Vec<T> + Deref<Target=XY<T>> {
    fn new(x: T, y: T) -> Self;
}

pub trait Dimension {}

#[derive(Clone, Copy, Debug)]
pub struct Dim2D {}
#[derive(Clone, Copy, Debug)]
pub struct Dim3D {}

impl Dimension for Dim2D {}
impl Dimension for Dim3D {}
