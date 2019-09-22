//! A collection of primitive SDFs that may the modified using functions in `mods`
//! or combined using functions in `ops`. Note that these primitives are always
//! centered around the origin and that you must transform the point you are sampling
//! into 'primitive-local' space. Functions are provided in `mods` to do this easier.
//! 
//! Also note that while all translation and rotation transformations of the input point
//! will work properly, scaling modifies the Euclidian space and therefore does not work
//! normally.
use crate::mathtypes::*;
use crate::SDF;
use std::ops::*;

/// A shere centered at origin with a radius.
#[derive(Clone, Copy, Debug)]
pub struct Sphere<T> {
    pub radius: T,
}

impl<T> Sphere<T> {
    pub fn new(radius: T) -> Self { Sphere { radius } }
}

impl<T, V> SDF<T, V> for Sphere<T>
    where T: Sub<T, Output=T> + Copy,
        V: Vec3<T>
{
    fn dist(&self, p: V) -> T {
        p.magnitude() - self.radius
    }
}

/// A box centered at origin with axis-aligned dimensions.
#[derive(Clone, Copy, Debug)]
pub struct Box<V> {
    pub dims: V
}

impl<V> Box<V> {
    pub fn new(dims: V) -> Self { Box { dims } }
}

impl<T, V> SDF<T, V> for Box<V>
    where T: Add<T, Output=T> + MaxMin + Zero + Copy,
        V: Vec3<T> + Copy
{
    fn dist(&self, p: V) -> T {
        let d = p.abs() - self.dims;
        d.max(V::ZERO).magnitude()
            + d.y.max(d.z).max(d.x).min(T::ZERO)
    }
}

/// A torus that sits on the XZ plane. Thickness is the radius of
/// the wrapped cylinder while radius is the radius of the donut
/// shape.
#[derive(Clone, Copy, Debug)]
pub struct Torus<T> {
    pub radius: T,
    pub thickness: T,
}

impl<T> Torus<T> {
    pub fn new(radius: T, thickness: T) -> Self { Torus { radius, thickness } }
}

impl<T, V> SDF<T, V> for Torus<T>
    where T: Sub<T, Output=T> + Copy,
        V: Vec3<T>,
{
    fn dist(&self, p: V) -> T {
        let q = V::Vec2::new(V::Vec2::new(p.x, p.z).magnitude() - self.thickness, p.y);
        q.magnitude() - self.radius
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Axis {
    X, Y, Z
}

/// An infinite cylinder extending along an axis.
#[derive(Clone, Copy, Debug)]
pub struct Cylinder<T> {
    pub radius: T,
    pub axis: Axis,
}

impl<T> Cylinder<T> {
    pub fn new(radius: T, axis: Axis) -> Self { Cylinder { radius, axis } }
}

impl<T, V> SDF<T, V> for Cylinder<T>
    where T: Sub<T, Output=T> + Copy,
        V: Vec3<T>,
{
    fn dist(&self, p: V) -> T {
        let (a, b) = match self.axis {
            Axis::X => (p.y, p.z),
            Axis::Y => (p.x, p.z),
            Axis::Z => (p.x, p.y),
        };
        V::Vec2::new(a, b).magnitude() - self.radius
    }
}

/// A capped cylinder extending along an axis.
#[derive(Clone, Copy, Debug)]
pub struct CappedCylinder<T> {
    pub radius: T,
    pub height: T,
    pub axis: Axis,
}

impl<T> CappedCylinder<T> {
    pub fn new(radius: T, height: T, axis: Axis) -> Self { CappedCylinder { radius, height, axis } }
}

impl<T, V> SDF<T, V> for CappedCylinder<T>
    where T: Sub<T, Output=T> + Add<T, Output=T> + Zero + MaxMin + Copy,
        V: Vec3<T>,
{
    fn dist(&self, p: V) -> T {
        let (a, b, c) = match self.axis {
            Axis::X => (p.y, p.z, p.x),
            Axis::Y => (p.x, p.z, p.y),
            Axis::Z => (p.x, p.y, p.z),
        };
        let d = V::Vec2::new(V::Vec2::new(a, b).magnitude(), c).abs() - V::Vec2::new(self.radius, self.height);
        d.x.max(d.y).min(T::ZERO) + d.max(V::Vec2::ZERO).magnitude()
    }
}

/// A capsule extending from `a` to `b` with radius `radius`.
#[derive(Clone, Copy, Debug)]
pub struct Capsule<T, V> {
    pub a: V,
    pub b: V,
    pub radius: T,
}

impl<T, V> Capsule<T, V> {
    pub fn new(a: V, b: V, radius: T) -> Self { Capsule { a, b, radius } }
}

impl<T, V> SDF<T, V> for Capsule<T, V>
where T: Sub<T, Output=T> + Mul<T, Output=T> + Div<T, Output=T> + Zero + One + Clamp + Copy,
    V: Vec3<T> + Copy,
{
    fn dist(&self, p: V) -> T {
        let pa = p - self.a;
        let ba = self.b - self.a;
        let t = pa.dot(ba) / ba.dot(ba);
        let h = t.clamp(T::ZERO, T::ZERO);
        (pa - (ba * h)).magnitude() - self.radius
    }
}