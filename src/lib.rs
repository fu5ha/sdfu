//! # `sdfu` - Signed Distance Field Utilities
//!
//! This is a small crate designed to help when working with signed distance fields
//! in the context of computer graphics, especially ray-marching based renderers. Most
//! of what is here is based on [Inigo Quilez' excellent articles](http://www.iquilezles.org/www/index.htm).
//!
//! If you're using one of the more popular math libraries in Rust, then just enable the corresponding
//! feature (currently, [`ultraviolet`](https://github.com/termhn/ultraviolet), `nalgebra` and `vek`
//! are supported) and hopefully all the necessary traits are already implemented for you so that
//! you can just start passing in your `Vec3`s or whatever your lib calls them and you're off to the
//! races! If not, then you can implement the necessary traits in the `mathtypes` module and still use
//! this library with your own math lib.
//!
//! This crate is built around the central trait `SDF`. This trait is structured in a similar way to
//! how `std::iter::Iterator` works. Anything that implements `SDF` is able to return a distance from
//! a point to its distance field. SDFs can be combined, modified, and otherwise used for various tasks
//! by using the combinator methods on the `SDF` trait, or by directly using the structs that actually
//! implement those combinators.
//!
//! Most `SDF`s will be build up from one or more primitives being modified and combined together--the
//! distance fields in the `primitive` module provide good starting points for this.
//!
//! # Demo
//!
//! ![demo image](https://raw.githubusercontent.com/termhn/sdfu/master/demo.png)
//!
//! The image above was rendered with my own path tracing renderer, [`rayn`](https://github.com/termhn/rayn),
//! by leveraging `sdfu`. The SDF that is rendered above was created with the following code:
//!
//! ```rust
//! # #[cfg(feature = "ultraviolet")]
//! # fn main() {
//! use sdfu::SDF;
//! use ultraviolet::Vec3;
//!
//! let sdf = sdfu::Sphere::new(0.45)
//!     .subtract(
//!         sdfu::Box::new(Vec3::new(0.25, 0.25, 1.5)))
//!     .union_smooth(
//!         sdfu::Sphere::new(0.3).translate(Vec3::new(0.3, 0.3, 0.0)),
//!         0.1)
//!     .union_smooth(
//!         sdfu::Sphere::new(0.3).translate(Vec3::new(-0.3, 0.3, 0.0)),
//!         0.1)
//!     .subtract(
//!         sdfu::Box::new(Vec3::new(0.125, 0.125, 1.5)).translate(Vec3::new(-0.3, 0.3, 0.0)))
//!     .subtract(
//!         sdfu::Box::new(Vec3::new(0.125, 0.125, 1.5)).translate(Vec3::new(0.3, 0.3, 0.0)))
//!     .subtract(
//!         sdfu::Box::new(Vec3::new(1.5, 0.1, 0.1)).translate(Vec3::new(0.0, 0.3, 0.0)))
//!     .subtract(
//!         sdfu::Box::new(Vec3::new(0.2, 2.0, 0.2)))
//!     .translate(Vec3::new(0.0, 0.0, -1.0));
//! # }
//! # #[cfg(not(feature = "ultraviolet"))]
//! # fn main() {}
//! ```
pub mod mathtypes;
use mathtypes::*;
pub use mathtypes::{Dim2D, Dim3D, Dimension};
pub mod primitives;
pub use primitives::*;

pub mod util;
use util::*;
pub mod ops;
use ops::*;
pub mod mods;
use mods::*;

/// The core trait of this crate; an implementor of this trait is able
/// to take in a vector and return the min distance from that vector to
/// a distance field.
pub trait SDF<T, V: Vec<T>>: Copy {
    /// Get distance from `p` to this SDF.
    fn dist(&self, p: V) -> T;

    /// Estimate the normals of this SDF using the default `NormalEstimator`.
    ///
    /// `eps` is the amount to change the point by for each sample.
    /// 0.001 is a good default value to try; you will ideally vary this based on distance.
    fn normals(self, eps: T) -> EstimateNormalDefault<T, V, Self>
    where
        CentralDifferenceEstimator<T, V, <V as Vec<T>>::Dimension>: NormalEstimator<T, V>,
    {
        EstimateNormal::new(self, CentralDifferenceEstimator::new(eps))
    }

    /// Estimate the normals of this SDF using a fast, `TetrahedralEstimator`. Only
    /// works for 3d SDFs.
    ///
    /// `eps` is the amount to change the point by for each sample.
    /// 0.001 is a good default value to try; you will ideally vary this based on distance.
    fn normals_fast(self, eps: T) -> EstimateNormalFast<T, V, Self>
    where
        TetrahedralEstimator<T, V>: NormalEstimator<T, V>,
    {
        EstimateNormal::new(self, TetrahedralEstimator::new(eps))
    }

    /// Estimate the normals of this SDF using a provided `NormalEstimator`.
    fn normals_with<E: NormalEstimator<T, V>>(self, estimator: E) -> EstimateNormal<T, V, Self, E> {
        EstimateNormal::new(self, estimator)
    }

    /// Get the union of this SDF and another one()using a standard
    /// hard minimum, creating a sharp crease at the boundary between the
    /// two fields.
    fn union<O: SDF<T, V>>(self, other: O) -> Union<T, Self, O, HardMin<T>> {
        Union::hard(self, other)
    }

    /// Get the union of this SDF and another one, blended together
    /// with a smooth minimum function. This uses a polynomial smooth min
    /// function by default, and the smoothing factor is controlled by the
    /// `smoothness` parameter. For even more control, see `union_with`.
    fn union_smooth<O: SDF<T, V>>(
        self,
        other: O,
        softness: T,
    ) -> Union<T, Self, O, PolySmoothMin<T>> {
        Union::smooth(self, other, softness)
    }

    /// Get the union of this SDF and another one()using a provided
    /// minimum function. See the documentation of `MinFunction` for more.
    fn union_with<O: SDF<T, V>, M: MinFunction<T>>(
        self,
        other: O,
        min_function: M,
    ) -> Union<T, Self, O, M> {
        Union::new(self, other, min_function)
    }
    /// Get the subtracion of another SDF from this one. Note that this operation is *not* commutative,
    /// i.e. `a.subtraction(b) =/= b.subtraction(a)`.
    fn subtract<O: SDF<T, V>>(self, other: O) -> Subtraction<O, Self> {
        Subtraction::new(other, self)
    }

    /// Get the intersection of this SDF and another one.
    fn intersection<O: SDF<T, V>>(self, other: O) -> Intersection<Self, O> {
        Intersection::new(self, other)
    }

    /// Round the corners of this SDF with a radius.
    fn round(self, radius: T) -> Round<T, Self> {
        Round::new(self, radius)
    }

    /// Elongate this SDF along one()axis. The elongation is symmetrical about the origin.
    fn elongate(self, axis: Axis, elongation: T) -> Elongate<T, Self, <V as Vec<T>>::Dimension>
    where
        Elongate<T, Self, <V as Vec<T>>::Dimension>: SDF<T, V>,
    {
        Elongate::new(self, axis, elongation)
    }

    /// Elongate this SDF along one()axis. The elongation is symmetrical about the origin.
    fn elongate_multi_axis(self, elongation: V) -> ElongateMulti<V, Self, <V as Vec<T>>::Dimension>
    where
        ElongateMulti<V, Self, <V as Vec<T>>::Dimension>: SDF<T, V>,
    {
        ElongateMulti::new(self, elongation)
    }

    /// Translate the SDF by a vector.
    fn translate(self, translation: V) -> Translate<V, Self> {
        Translate::new(self, translation)
    }

    /// Rotate the SDF by a rotation.
    fn rotate<R: Rotation<V>>(self, rotation: R) -> Rotate<R, Self> {
        Rotate::new(self, rotation)
    }
    /// Scale the SDF by a uniform scaling factor.
    fn scale(self, scaling: T) -> Scale<T, Self> {
        Scale::new(self, scaling)
    }
}
