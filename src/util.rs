//! Other random utilities that are helpful when using SDFs in computer graphics applications,
//! such as estimating normals.
use super::*;
use std::ops::*;

#[cfg(feature = "ultraviolet")]
use ultraviolet::f32x4;

/// Estimates the normal of an `sdf` using an `estimator`, by default a `CentralDifferenceEstimator`,
/// which provides a good default estimator that works for both 2D and 3D SDFs. See the documentation
/// of `NormalEstimator` for more information.
pub struct EstimateNormal<
    T,
    V: Vec<T>,
    S,
    E = CentralDifferenceEstimator<T, V, <V as Vec<T>>::Dimension>,
> {
    pub sdf: S,
    pub estimator: E,
    _pdt: std::marker::PhantomData<T>,
    _pdv: std::marker::PhantomData<V>,
}

impl<T, V, S, E> EstimateNormal<T, V, S, E>
where
    E: NormalEstimator<T, V> + Default,
    S: SDF<T, V>,
    V: Vec<T>,
{
    /// Creates a new `EstimateNormal` with an SDF using the default version of the
    /// estimator type.
    pub fn new_default(sdf: S) -> Self {
        EstimateNormal {
            sdf,
            estimator: E::default(),
            _pdt: std::marker::PhantomData,
            _pdv: std::marker::PhantomData,
        }
    }
}

impl<T, V, S, E> EstimateNormal<T, V, S, E>
where
    E: NormalEstimator<T, V>,
    S: SDF<T, V>,
    V: Vec<T>,
{
    /// Creates a new `EstimateNormal` with an SDF and a provided estimator.
    pub fn new(sdf: S, estimator: E) -> Self {
        EstimateNormal {
            sdf,
            estimator,
            _pdt: std::marker::PhantomData,
            _pdv: std::marker::PhantomData,
        }
    }

    /// Estimates the normal of the owned SDF at point p.
    pub fn normal_at(&self, p: V) -> V {
        self.estimator.estimate_normal(self.sdf, p)
    }
}

/// `NormalEstimator`s provide a way to estimate the normal of the SDF `sdf` at point `p`.
pub trait NormalEstimator<T, V: Vec<T>> {
    fn estimate_normal<S: SDF<T, V>>(&self, sdf: S, p: V) -> V;
}

/// Estimates the normal of an SDF by estimating the gradient of the SDF.
///
/// The gradient is estimated by taking two samples of the SDF in each dimension,
/// one slightly above (by `eps` distance) the point in question and one slightly below it and taking their
/// difference, hence the 'central difference'. This estimation is relatively robust and accurate, and can
/// work in both two and three dimensions, but is also relatively slow since it takes 6 samples of the SDF.
/// See the `TetrahedralEstimator` for an estimator which is 3d only and slightly less robust/accurate but
/// also slightly faster.
///
/// See [this article](http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm)
/// for more.
pub struct CentralDifferenceEstimator<T, V, D> {
    pub eps: T,
    _pdv: std::marker::PhantomData<V>,
    _pdd: std::marker::PhantomData<D>,
}

impl<T, V: Vec<T>> CentralDifferenceEstimator<T, V, <V as Vec<T>>::Dimension> {
    /// Creates a `CentralDifferenceEstimator` with a given epsilon value.
    pub fn new(eps: T) -> Self {
        CentralDifferenceEstimator {
            eps,
            _pdv: std::marker::PhantomData,
            _pdd: std::marker::PhantomData,
        }
    }
}

impl<T, V> NormalEstimator<T, V> for CentralDifferenceEstimator<T, V, Dim3D>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Copy,
    V: Vec3<T>,
{
    fn estimate_normal<S: SDF<T, V>>(&self, sdf: S, p: V) -> V {
        let eps = self.eps;
        V::new(
            sdf.dist(V::new(p.x() + eps, p.y(), p.z()))
                - sdf.dist(V::new(p.x() - eps, p.y(), p.z())),
            sdf.dist(V::new(p.x(), p.y() + eps, p.z()))
                - sdf.dist(V::new(p.x(), p.y() - eps, p.z())),
            sdf.dist(V::new(p.x(), p.y(), p.z() + eps))
                - sdf.dist(V::new(p.x(), p.y(), p.z() - eps)),
        )
        .normalized()
    }
}

impl<T, V> NormalEstimator<T, V> for CentralDifferenceEstimator<T, V, Dim2D>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Copy,
    V: Vec2<T>,
{
    fn estimate_normal<S: SDF<T, V>>(&self, sdf: S, p: V) -> V {
        let eps = self.eps;
        V::new(
            sdf.dist(V::new(p.x() + eps, p.y())) - sdf.dist(V::new(p.x() - eps, p.y())),
            sdf.dist(V::new(p.x(), p.y() + eps)) - sdf.dist(V::new(p.x(), p.y() - eps)),
        )
        .normalized()
    }
}

#[cfg(feature = "ultraviolet")]
impl<V: Vec<f32x4>> Default for CentralDifferenceEstimator<f32x4, V, <V as Vec<f32x4>>::Dimension> {
    fn default() -> Self {
        Self::new(f32x4::from(0.000))
    }
}

impl<V: Vec<f32>> Default for CentralDifferenceEstimator<f32, V, <V as Vec<f32>>::Dimension> {
    fn default() -> Self {
        Self::new(0.001)
    }
}

impl<V: Vec<f64>> Default for CentralDifferenceEstimator<f64, V, <V as Vec<f64>>::Dimension> {
    fn default() -> Self {
        Self::new(0.001)
    }
}

/// Estimates the normal of an SDF by estimating the gradient of the SDF.
///
/// The gradient is estimated by taking four samples of the SDF in a tetrahedron around the
/// point of interest. By doing so, it only needs to take four instead of 6 samples of the SDF,
/// like the CentralDifferenceEstimator does, so it is slightly faster. However, it only works
/// for 3d SDFs and it is slightly less robust than the traditional way.
///
/// See [this article](http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm)
/// for more.
pub struct TetrahedralEstimator<T, V> {
    pub eps: T,
    _pdv: std::marker::PhantomData<V>,
}

impl<T, V: Vec<T>> TetrahedralEstimator<T, V> {
    /// Creates a `TetrahedralEstimator` with a given epsilon value.
    pub fn new(eps: T) -> Self {
        TetrahedralEstimator {
            eps,
            _pdv: std::marker::PhantomData,
        }
    }
}

impl<T, V> NormalEstimator<T, V> for TetrahedralEstimator<T, V>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Neg<Output = T> + One + Copy,
    V: Vec3<T>,
{
    fn estimate_normal<S: SDF<T, V>>(&self, sdf: S, p: V) -> V {
        let xyy = V::new(T::one(), -T::one(), -T::one());
        let yyx = V::new(-T::one(), -T::one(), T::one());
        let yxy = V::new(-T::one(), T::one(), -T::one());
        let xxx = V::one();

        (xyy * sdf.dist(p + xyy * self.eps)
            + yyx * sdf.dist(p + xyy * self.eps)
            + yxy * sdf.dist(p + xyy * self.eps)
            + xxx * sdf.dist(p + xxx * self.eps))
        .normalized()
    }
}

#[cfg(feature = "ultraviolet")]
impl<V: Vec<f32x4>> Default for TetrahedralEstimator<f32x4, V> {
    fn default() -> Self {
        Self::new(f32x4::from(0.001))
    }
}

impl<V: Vec<f32>> Default for TetrahedralEstimator<f32, V> {
    fn default() -> Self {
        Self::new(0.001)
    }
}

impl<V: Vec<f64>> Default for TetrahedralEstimator<f64, V> {
    fn default() -> Self {
        Self::new(0.001)
    }
}
