pub mod mathtypes;
use mathtypes::*;
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
/// its field.
pub trait SDF<T, V: Vec<T>>: Copy {
    /// Get distance from `p` to this SDF.
    fn dist(&self, p: V) -> T;

    /// Estimate the normals of this SDF using the default `NormalEstimator`.
    fn normals(self) -> EstimateNormal<T, V, Self, CentralDifferenceEstimator<T, V, <V as Vec<T>>::Dimension>>
    where CentralDifferenceEstimator<T, V, <V as Vec<T>>::Dimension>: NormalEstimator<T, V> + Default
    {
        EstimateNormal::new_default(self)
    }

    /// Estimate the normals of this SDF using a fast, `TetrahedralEstimator`. Only
    /// works for 3d SDFs.
    fn normals_fast(self) -> EstimateNormal<T, V, Self, TetrahedralEstimator<T, V>>
    where TetrahedralEstimator<T, V>: NormalEstimator<T, V> + Default
    {
        EstimateNormal::new(self, Default::default())
    }

    /// Estimate the normals of this SDF using a provided `NormalEstimator`.
    fn normals_with<E: NormalEstimator<T, V>>(self, estimator: E) -> EstimateNormal<T, V, Self, E> {
        EstimateNormal::new(self, estimator)
    }

    /// Get the union of this SDF and another one using a standard
    /// hard minimum, creating a sharp crease at the boundary between the
    /// two fields.
    fn union<O: SDF<T, V>>(self, other: O) -> Union<T, Self, O, HardMin<T>> {
        Union::hard(self, other)
    }

    /// Get the union of this SDF and another one, blended together
    /// with a smooth minimum function. This uses a polynomial smooth min
    /// function by default, and the smoothing factor is controlled by the
    /// `smoothness` parameter. For even more control, see `union_with`.
    fn union_smooth<O: SDF<T, V>>(self, other: O, softness: T) -> Union<T, Self, O, PolySmoothMin<T>> {
        Union::smooth(self, other, softness)
    }

    /// Get the union of this SDF and another one using a provided
    /// minimum function. See the documentation of `MinFunction` for more.
    fn union_with<O: SDF<T, V>, M: MinFunction<T>>(self, other: O, min_function: M) -> Union<T, Self, O, M> {
        Union::new(self, other, min_function)
    }
    
    /// Get the subtracion of another SDF from this one. Note that this operation is *not* commutative,
    /// i.e. `a.subtraction(b) =/= b.subtraction(a)`.
    fn subtraction<O: SDF<T, V>>(self, other: O) -> Subtraction<Self, O> {
        Subtraction::new(self, other)
    }

    /// Get the intersection of this SDF and another one.
    fn intersection<O: SDF<T, V>>(self, other: O) -> Intersection<Self, O> {
        Intersection::new(self, other)
    }

    /// Round the corners of this SDF with a radius.
    fn round(self, radius: T) -> Round<T, Self> {
        Round::new(self, radius)
    }

    /// Elongate this SDF along one axis. The elongation is symmetrical about the origin.
    fn elongate(self, axis: Axis, elongation: T) -> Elongate<T, Self, <V as Vec<T>>::Dimension>
    where Elongate<T, Self, <V as Vec<T>>::Dimension>: SDF<T, V>
    {
        Elongate::new(self, axis, elongation)
    }

    /// Elongate this SDF along one axis. The elongation is symmetrical about the origin.
    fn elongate_multi_axis(self, elongation: V) -> ElongateMulti<V, Self, <V as Vec<T>>::Dimension>
    where ElongateMulti<V, Self, <V as Vec<T>>::Dimension>: SDF<T, V>
    {
        ElongateMulti::new(self, elongation)
    }
}
