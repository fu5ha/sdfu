pub mod mathtypes;

/// The core trait of this crate; an implementor of this trait is able
/// to take in a vector and return the min distance from that vector to
/// its field.
pub trait SDF<T, V>: Copy {
    fn dist(&self, p: V) -> T;
}

pub mod primitives;
pub use primitives::*;

pub mod util;

///// Modifiers for SDFs.
// pub mod mods {
//     use crate::mathtypes::*;
//     use std::ops::*;

//     /// Make an SDF have rounded corners. `d` is distance obtained from SDF.
//     pub fn round<T: Sub<T, Output=T>>(d: T, radius: T) -> T {
//         d - radius
//     }

//     /// Elongate an SDF along the C axis.
//     pub fn elongate_x<T, V, F>(sdf: F, p: V, by: T) -> T
//         where T: Add<T, Output=T> + Sub<T, Output=T> + Zero + Copy,
//             V: Vec3<T> + Copy,
//             F: Fn(V) -> T,
//     {
//         let h = V::new(by, T::zero(), T::zero());
//         let q = p - p.clamp(-h, h);
//         sdf(q)
//     }

//     /// Elongate an SDF along the Y axis.
//     pub fn elongate_y<T, V, F>(sdf: F, p: V, by: T) -> T
//         where T: Add<T, Output=T> + Sub<T, Output=T> + Zero + Copy,
//             V: Vec3<T> + Copy,
//             F: Fn(V) -> T,
//     {
//         let h = V::new(T::zero(), by, T::zero());
//         let q = p - p.clamp(-h, h);
//         sdf(q)
//     }

//     /// Elongate an SDF along the Z axis.
//     pub fn elongate_z<T, V, F>(sdf: F, p: V, by: T) -> T
//         where T: Add<T, Output=T> + Sub<T, Output=T> + Zero + Copy,
//             V: Vec3<T> + Copy,
//             F: Fn(V) -> T,
//     {
//         let h = V::new(T::zero(), T::zero(), by);
//         let q = p - p.clamp(-h, h);
//         sdf(q)
//     }

//     /// Elongate an SDF along multiple axes at the same time.
//     pub fn elongate_multi<T, V, F>(sdf: F, p: V, by: V) -> T
//         where T: Add<T, Output=T> + Sub<T, Output=T> + MaxMin + Zero + Copy,
//             V: Vec3<T> + Copy,
//             F: Fn(V) -> T,
//     {
//         let q = p.abs() - by;
//         let t = q.y().max(q.z()).max(q.x()).min(T::zero());
//         sdf(q.max(V::zero())) + t
//     }
// }

// /// Operations you can perform to combine two SDFs.
// pub mod ops {
//     use crate::mathtypes::*;
//     use std::ops::*;

//     /// Get the union of two SDFs.
//     pub fn union<T: MaxMin>(dist1: T, dist2: T) -> T {
//         dist1.min(dist2)
//     }

//     /// Get the subtracion of two SDFs. Note that this operation is *not* commutative,
//     /// i.e. `subtraction(a, b) =/= subtracion(b, a)`.
//     pub fn subtraction<T: Neg<Output=T> + MaxMin>(dist1: T, dist2: T) -> T {
//         -dist1.max(dist2)
//     }

//     /// Get the intersection of two SDFs.
//     pub fn intersection<T: MaxMin>(dist1: T, dist2: T) -> T {
//         dist1.max(dist2)
//     }
// }
