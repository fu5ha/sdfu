pub mod mathtypes;

pub use primitives::*;
/// A collection of primitive SDFs that may the modified using functions in `mods`
/// or combined using functions in `ops`. Note that these primitives are always
/// centered around the origin and that you must transform the point you are sampling
/// into 'primitive-local' space. Functions are provided in `mods` to do this easier.
/// 
/// Also note that while all translation and rotation transformations of the input point
/// will work properly, scaling modifies the Euclidian space and therefore does not work
/// normally.
pub mod primitives {
    use crate::mathtypes::*;
    use std::ops::*;

    /// A sphere centered at origin.
    pub fn sphere<T: Sub<T, Output=T>, V: Vec3<T>>(p: V, radius: T) -> T {
        p.magnitude() - radius
    }

    /// A box centered at origin with axis-aligned dimensions.
    pub fn sd_box<T, V>(p: V, dims: V) -> T 
        where T: Add<T, Output=T> + MaxMin + Zero,
              V: Vec3<T> 
    {
        let d = p.abs() - dims;
        d.max(V::zero()).magnitude()
            + d.y().max(d.z()).max(d.x()).min(T::zero())
    }

    /// A torus that sits on the XZ plane.
    pub fn torus<T, V3, V2>(p: V3, radius: T, thickness: T) -> T 
        where T: Sub<T, Output=T>,
              V3: Vec3<T>,
            V2: Vec2<T>
    {
        let q = V2::new(V2::new(p.x(), p.z()).magnitude() - thickness, p.y());
        q.magnitude() - radius
    }

    /// An infinite cylinder along the X axis.
    pub fn cylinder_x<T, V3, V2>(p: V3, radius: T) -> T
        where T: Sub<T, Output=T>,
            V3: Vec3<T>,
            V2: Vec2<T>
    {
        V2::new(p.y(), p.z()).magnitude() - radius
    }

    /// An infinite cylinder along the Y axis.
    pub fn cylinder_y<T, V3, V2>(p: V3, radius: T) -> T
        where T: Sub<T, Output=T>,
            V3: Vec3<T>,
            V2: Vec2<T>
    {
        V2::new(p.x(), p.z()).magnitude() - radius
    }

    /// An infinite cylinder along the Z axis.
    pub fn cylinder_z<T, V3, V2>(p: V3, radius: T) -> T
        where T: Sub<T, Output=T>,
            V3: Vec3<T>,
            V2: Vec2<T>
    {
        V2::new(p.x(), p.y()).magnitude() - radius
    }
    
    /// A capped cylinder along the X axis.
    pub fn capped_cylinder_x<T, V3, V2>(p: V3, radius: T, height: T) -> T
        where T: Add<T, Output=T> + Sub<T, Output=T> + MaxMin + Zero,
            V3: Vec3<T>,
            V2: Vec2<T>
    {
        let d = V2::new(V2::new(p.y(), p.z()).magnitude(), p.x()).abs() - V2::new(radius, height);
        d.x().max(d.y()).min(T::zero()) + d.max(V2::zero()).magnitude()
    }

    /// A capped cylinder along the Y axis.
    pub fn capped_cylinder_y<T, V3, V2>(p: V3, radius: T, height: T) -> T
        where T: Add<T, Output=T> + Sub<T, Output=T> + MaxMin + Zero,
            V3: Vec3<T>,
            V2: Vec2<T>
    {
        let d = V2::new(V2::new(p.x(), p.z()).magnitude(), p.y()).abs() - V2::new(radius, height);
        d.x().max(d.y()).min(T::zero()) + d.max(V2::zero()).magnitude()
    }

    /// A capped cylinder along the Z axis.
    pub fn capped_cylinder_z<T, V3, V2>(p: V3, radius: T, height: T) -> T
        where T: Add<T, Output=T> + Sub<T, Output=T> + MaxMin + Zero,
            V3: Vec3<T>,
            V2: Vec2<T>
    {
        let d = V2::new(V2::new(p.x(), p.y()).magnitude(), p.z()).abs() - V2::new(radius, height);
        d.x().max(d.y()).min(T::zero()) + d.max(V2::zero()).magnitude()
    }

    /// A capsule that extends from `a` to `b` with radius `r`.
    pub fn capsule<T, V>(p: V, a: V, b: V, r: T) -> T
        where T: Sub<T, Output=T> + Mul<T, Output=T> + Div<T, Output=T> + Zero + One + Clamp + Copy,
            V: Vec3<T> + Copy,
    {
        let pa = p - a;
        let ba = b - a;
        let t = pa.dot(ba) / ba.dot(ba);
        let h = t.clamp(T::zero(), T::one());
        (pa - (ba * h)).magnitude() - r
    }
}

/// Other random utilities that are helpful when using SDFs in computer graphics applications.
pub mod util {
    use crate::mathtypes::*;
    use std::ops::*;

    /// Estimate the normal of the SDF `sdf` at point `p`
    /// with epsilon `eps`. A good default value for `eps` is 0.0001.
    pub fn estimate_normal<T, V, F>(sdf: F, p: V, eps: T) -> V 
        where T: Add<T, Output=T> + Sub<T, Output=T> + Copy,
            V: Vec3<T>,
            F: Fn(V) -> T,
    {
        V::new(
            sdf(V::new(p.x() + eps, p.y(), p.z())) - sdf(V::new(p.x() - eps, p.y(), p.z())),
            sdf(V::new(p.x(), p.y() + eps, p.z())) - sdf(V::new(p.x(), p.y() - eps, p.z())),
            sdf(V::new(p.x(), p.y(), p.z() + eps)) - sdf(V::new(p.x(), p.y(), p.z() - eps))
        ).normalized()
    }
}

/// Modifiers for SDFs.
pub mod mods {
    use crate::mathtypes::*;
    use std::ops::*;

    /// Make an SDF have rounded corners. `d` is distance obtained from SDF.
    pub fn round<T: Sub<T, Output=T>>(d: T, radius: T) -> T {
        d - radius
    }

    /// Elongate an SDF along the C axis.
    pub fn elongate_x<T, V, F>(sdf: F, p: V, by: T) -> T
        where T: Add<T, Output=T> + Sub<T, Output=T> + Zero + Copy,
            V: Vec3<T> + Copy,
            F: Fn(V) -> T,
    {
        let h = V::new(by, T::zero(), T::zero());
        let q = p - p.clamp(-h, h);
        sdf(q)
    }

    /// Elongate an SDF along the Y axis.
    pub fn elongate_y<T, V, F>(sdf: F, p: V, by: T) -> T
        where T: Add<T, Output=T> + Sub<T, Output=T> + Zero + Copy,
            V: Vec3<T> + Copy,
            F: Fn(V) -> T,
    {
        let h = V::new(T::zero(), by, T::zero());
        let q = p - p.clamp(-h, h);
        sdf(q)
    }

    /// Elongate an SDF along the Z axis.
    pub fn elongate_z<T, V, F>(sdf: F, p: V, by: T) -> T
        where T: Add<T, Output=T> + Sub<T, Output=T> + Zero + Copy,
            V: Vec3<T> + Copy,
            F: Fn(V) -> T,
    {
        let h = V::new(T::zero(), T::zero(), by);
        let q = p - p.clamp(-h, h);
        sdf(q)
    }

    /// Elongate an SDF along multiple axes at the same time.
    pub fn elongate_multi<T, V, F>(sdf: F, p: V, by: V) -> T
        where T: Add<T, Output=T> + Sub<T, Output=T> + MaxMin + Zero + Copy,
            V: Vec3<T> + Copy,
            F: Fn(V) -> T,
    {
        let q = p.abs() - by;
        let t = q.y().max(q.z()).max(q.x()).min(T::zero());
        sdf(q.max(V::zero())) + t
    }
}

/// Operations you can perform to combine two SDFs.
pub mod ops {
    use crate::mathtypes::*;
    use std::ops::*;

    /// Get the union of two SDFs.
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
}
