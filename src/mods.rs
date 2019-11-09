//! Modifiers for SDFs.
use super::*;
use crate::mathtypes::*;
use std::ops::*;

/// Make an SDF have rounded outside edges.
#[derive(Clone, Copy, Debug)]
pub struct Round<T, S> {
    pub sdf: S,
    pub radius: T,
}

impl<T, S> Round<T, S> {
    pub fn new(sdf: S, radius: T) -> Self {
        Round { sdf, radius }
    }
}

impl<T, V, S> SDF<T, V> for Round<T, S>
where
    T: Copy + Sub<T, Output = T>,
    V: Vec<T>,
    S: SDF<T, V>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        self.sdf.dist(p) - self.radius
    }
}

/// Elongate an SDF along a single axis. The elongation is
/// symmetrical around the origin.
#[derive(Clone, Copy, Debug)]
pub struct Elongate<T, S, D> {
    pub sdf: S,
    pub axis: Axis,
    pub elongation: T,
    _pd: std::marker::PhantomData<D>,
}

impl<T, S, D> Elongate<T, S, D> {
    /// Elongate an SDF along a single axis by `elongation`.
    pub fn new(sdf: S, axis: Axis, elongation: T) -> Self {
        Elongate {
            sdf,
            axis,
            elongation,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<T, V, S> SDF<T, V> for Elongate<T, S, Dim3D>
where
    T: Copy + Add<T, Output = T> + Sub<T, Output = T> + Zero,
    V: Vec3<T>,
    S: SDF<T, V>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let h = match self.axis {
            Axis::X => V::new(self.elongation, T::zero(), T::zero()),
            Axis::Y => V::new(T::zero(), self.elongation, T::zero()),
            Axis::Z => V::new(T::zero(), T::zero(), self.elongation),
        };
        let q = p - p.clamp(-h, h);
        self.sdf.dist(q)
    }
}

impl<T, V, S> SDF<T, V> for Elongate<T, S, Dim2D>
where
    T: Copy + Add<T, Output = T> + Sub<T, Output = T> + Zero,
    V: Vec2<T>,
    S: SDF<T, V>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let h = match self.axis {
            Axis::X => V::new(self.elongation, T::zero()),
            Axis::Y => V::new(T::zero(), self.elongation),
            Axis::Z => panic!("Attempting to use Z axis to elongate 2d SDF"),
        };
        let q = p - p.clamp(-h, h);
        self.sdf.dist(q)
    }
}

/// Elongate an SDF along multiple axes.
#[derive(Clone, Copy, Debug)]
pub struct ElongateMulti<V, S, D> {
    pub sdf: S,
    pub elongation: V,
    _pd: std::marker::PhantomData<D>,
}

impl<V, S, D> ElongateMulti<V, S, D> {
    pub fn new(sdf: S, elongation: V) -> Self {
        ElongateMulti {
            sdf,
            elongation,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<T, V, S> SDF<T, V> for ElongateMulti<V, S, Dim3D>
where
    T: Copy + Add<T, Output = T> + Sub<T, Output = T> + Zero + MaxMin,
    V: Vec3<T>,
    S: SDF<T, V>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let q = p.abs() - self.elongation;
        let t = q.y().max(q.z()).max(q.x()).min(T::zero());
        self.sdf.dist(q.max(V::zero())) + t
    }
}

impl<T, V, S> SDF<T, V> for ElongateMulti<V, S, Dim2D>
where
    T: Copy + Add<T, Output = T> + Sub<T, Output = T> + Zero + MaxMin,
    V: Vec2<T>,
    S: SDF<T, V>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let q = p.abs() - self.elongation;
        let t = q.x().max(q.y()).min(T::zero());
        self.sdf.dist(q.max(V::zero())) + t
    }
}

/// Translate an SDF.
#[derive(Clone, Copy, Debug)]
pub struct Translate<V, S> {
    pub sdf: S,
    pub translation: V,
}

impl<V, S> Translate<V, S> {
    pub fn new(sdf: S, translation: V) -> Self {
        Translate { sdf, translation }
    }
}

impl<T, V, S> SDF<T, V> for Translate<V, S>
where
    T: Copy,
    V: Vec<T>,
    S: SDF<T, V>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        self.sdf.dist(p - self.translation)
    }
}

/// Rotate an SDF.
#[derive(Clone, Copy, Debug)]
pub struct Rotate<R, S> {
    pub sdf: S,
    pub rotation: R,
}

impl<R, S> Rotate<R, S> {
    pub fn new(sdf: S, rotation: R) -> Self {
        Rotate { sdf, rotation }
    }
}

impl<T, V, R, S> SDF<T, V> for Rotate<R, S>
where
    T: Copy,
    V: Vec<T>,
    S: SDF<T, V>,
    R: Rotation<V> + Copy,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        self.sdf.dist(self.rotation.rotate_vec(p))
    }
}

/// Rotate an SDF.
#[derive(Clone, Copy, Debug)]
pub struct Scale<T, S> {
    pub sdf: S,
    pub scaling: T,
}

impl<T, S> Scale<T, S> {
    pub fn new(sdf: S, scaling: T) -> Self {
        Scale { sdf, scaling }
    }
}

impl<T, V, S> SDF<T, V> for Scale<T, S>
where
    T: Copy + Mul<T, Output = T>,
    V: Vec<T>,
    S: SDF<T, V>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        self.sdf.dist(p / self.scaling) * self.scaling
    }
}
