//! Modifiers for SDFs.
use crate::mathtypes::*;
use super::*;
use std::ops::*;

/// Make an SDF have Round corners. `d` is distance obtained from SDF.
#[derive(Clone, Copy, Debug)]
pub struct Round<T, S> {
    pub sdf: S,
    pub radius: T,
}

impl<T, S> Round<T, S> {
    pub fn new(sdf: S, radius: T) -> Self { Round { sdf, radius } }
}

impl<T, V, S> SDF<T, V> for Round<T, S>
where T: Copy + Sub<T, Output=T>,
    V: Vec<T>,
    S: SDF<T, V>
{
    fn dist(&self, p: V) -> T {
        self.sdf.dist(p) - self.radius
    }
}

/// Elongate an SDF along a single axis.
#[derive(Clone, Copy, Debug)]
pub struct Elongate<T, S, D> {
    pub sdf: S,
    pub axis: Axis,
    pub elongation: T,
    _pd: std::marker::PhantomData<D>
}

impl<T, S, D> Elongate<T, S, D> {
    pub fn new(sdf: S, axis: Axis, elongation: T) -> Self { Elongate { sdf, axis, elongation, _pd: std::marker::PhantomData } }
}

impl<T, V, S> SDF<T, V> for Elongate<T, S, Dim3D>
where T: Copy + Add<T, Output=T> + Sub<T, Output=T> + Zero,
    V: Vec3<T>,
    S: SDF<T, V>
{
    fn dist(&self, p: V) -> T {
        let h = match self.axis {
            Axis::X => V::new(self.elongation, T::ZERO, T::ZERO),
            Axis::Y => V::new(T::ZERO, self.elongation, T::ZERO),
            Axis::Z => V::new(T::ZERO, T::ZERO, self.elongation),
        };
        let q = p - p.clamp(-h, h);
        self.sdf.dist(q)
    }
}

impl<T, V, S> SDF<T, V> for Elongate<T, S, Dim2D>
where T: Copy + Add<T, Output=T> + Sub<T, Output=T> + Zero,
    V: Vec2<T>,
    S: SDF<T, V>
{
    fn dist(&self, p: V) -> T {
        let h = match self.axis {
            Axis::X => V::new(self.elongation, T::ZERO),
            Axis::Y => V::new(T::ZERO, self.elongation),
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
    _pd: std::marker::PhantomData<D>
}

impl<V, S, D> ElongateMulti<V, S, D> {
    pub fn new(sdf: S, elongation: V) -> Self { ElongateMulti { sdf, elongation, _pd: std::marker::PhantomData } }
}

impl<T, V, S> SDF<T, V> for ElongateMulti<V, S, Dim3D>
where T: Copy + Add<T, Output=T> + Sub<T, Output=T> + Zero + MaxMin,
    V: Vec3<T>,
    S: SDF<T, V>
{
    fn dist(&self, p: V) -> T {
        let q = p.abs() - self.elongation;
        let t = q.y.max(q.z).max(q.x).min(T::ZERO);
        self.sdf.dist(q.max(V::ZERO)) + t
    }
}

impl<T, V, S> SDF<T, V> for ElongateMulti<V, S, Dim2D>
where T: Copy + Add<T, Output=T> + Sub<T, Output=T> + Zero + MaxMin,
    V: Vec2<T>,
    S: SDF<T, V>
{
    fn dist(&self, p: V) -> T {
        let q = p.abs() - self.elongation;
        let t = q.x.max(q.y).min(T::ZERO);
        self.sdf.dist(q.max(V::ZERO)) + t
    }
}
