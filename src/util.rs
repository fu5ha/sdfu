use super::*;
use crate::mathtypes::*;
use std::ops::*;

/// Estimates the normal of an `sdf` using an `estimator`, by default a CentralDifferenceEstimator.
pub struct EstimateNormal<T, V: Vec<T>, S, E=CentralDifferenceEstimator<T, V, <V as Vec<T>>::Dimension>> {
    pub sdf: S,
    pub estimator: E,
    _pdt: std::marker::PhantomData<T>,
    _pdv: std::marker::PhantomData<V>,
}

impl<T, V, S, E> EstimateNormal<T, V, S, E>
where E: NormalEstimator<T, V> + Default,
    S: SDF<T, V>,
    V: Vec<T>
{
    /// Creates a new `EstimateNormal` with an SDF using the default version of the
    /// estimator type.
    pub fn new(sdf: S) -> Self {
        EstimateNormal {
            sdf,
            estimator: E::default(),
            _pdt: std::marker::PhantomData,
            _pdv: std::marker::PhantomData,
        }
    }
}

impl<T, V, S, E> EstimateNormal<T, V, S, E>
where E: NormalEstimator<T, V>,
    S: SDF<T, V>,
    V: Vec<T>,
{
    /// Creates a new `EstimateNormal` with an SDF and a provided estimator.
    pub fn new_with_estimator(sdf: S, estimator: E) -> Self {
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

/// Estimate the normal of the SDF `sdf` at point `p`
pub trait NormalEstimator<T, V> {
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

impl<T, V: Vec<T>> CentralDifferenceEstimator<T, V, <V as Vec<T>>::Dimension>
{
    /// Creates a `CentralDifferenceEstimator` with a given epsilon value.
    pub fn new(eps: T) -> Self { CentralDifferenceEstimator { eps, _pdv: std::marker::PhantomData, _pdd: std::marker::PhantomData } }
}

trait CentralDifferenceEstimatorImpl<T, V, D> {
    fn internal_estimate_normal<S: SDF<T, V>>(&self, sdf: S, p: V, eps: T) -> V;
}

impl<T, V> CentralDifferenceEstimatorImpl<T, V, Dim3D> for CentralDifferenceEstimator<T, V, Dim3D>
where T: Add<T, Output=T> + Sub<T, Output=T> + Copy,
    V: Vec3<T>,
{
    fn internal_estimate_normal<S: SDF<T, V>>(&self, sdf: S, p: V, eps: T) -> V {
        V::new(
            sdf.dist(V::new(p.x + eps, p.y, p.z)) - sdf.dist(V::new(p.x - eps, p.y, p.z)),
            sdf.dist(V::new(p.x, p.y + eps, p.z)) - sdf.dist(V::new(p.x, p.y - eps, p.z)),
            sdf.dist(V::new(p.x, p.y, p.z + eps)) - sdf.dist(V::new(p.x, p.y, p.z - eps))
        ).normalized()
    }
}

impl<T, V> CentralDifferenceEstimatorImpl<T, V, Dim2D> for CentralDifferenceEstimator<T, V, Dim2D>
where T: Add<T, Output=T> + Sub<T, Output=T> + Copy,
    V: Vec2<T>,
{
    fn internal_estimate_normal<S: SDF<T, V>>(&self, sdf: S, p: V, eps: T) -> V {
        V::new(
            sdf.dist(V::new(p.x + eps, p.y)) - sdf.dist(V::new(p.x - eps, p.y)),
            sdf.dist(V::new(p.x, p.y + eps)) - sdf.dist(V::new(p.x, p.y - eps)),
        ).normalized()
    }
}

impl<T, V> NormalEstimator<T, V> for CentralDifferenceEstimator<T, V, Dim3D> 
where T: Add<T, Output=T> + Sub<T, Output=T> + Copy,
    V: Vec3<T>,
{
    fn estimate_normal<S: SDF<T, V>>(&self, sdf: S, p: V) -> V 
    {
        CentralDifferenceEstimatorImpl::internal_estimate_normal(self, sdf, p, self.eps)
    }
}

impl<T, V> NormalEstimator<T, V> for CentralDifferenceEstimator<T, V, Dim2D> 
where T: Add<T, Output=T> + Sub<T, Output=T> + Copy,
    V: Vec2<T>,
{
    fn estimate_normal<S: SDF<T, V>>(&self, sdf: S, p: V) -> V 
    {
        CentralDifferenceEstimatorImpl::internal_estimate_normal(self, sdf, p, self.eps)
    }
}

impl<V: Vec<f32>> Default for CentralDifferenceEstimator<f32, V, <V as Vec<f32>>::Dimension> {
    fn default() -> Self {
        Self::new(0.0001)
    }
}

impl<V: Vec<f64>> Default for CentralDifferenceEstimator<f64, V, <V as Vec<f64>>::Dimension> {
    fn default() -> Self {
        Self::new(0.0001)
    }
}

/// Estimates the normal of an SDF by estimating the gradient of the SDF.
/// 
/// The gradient is estimated by taking four samples of the SDF
/// one slightly above (by `eps` distance) the point in question and one slightly below it and taking their
/// difference, hence the 'central difference'. This estimation is relatively robust and accurate,
/// but is also relatively slow since it takes 6 samples of the SDF. See the `TetrahedralEstimator` for
/// an estimator which is slightly less robust/accurate but also slightly faster.
/// 
/// See [this article](http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm)
/// for more.
pub struct TetrahedralEstimator<T, V> {
    pub eps: T,
    _pdv: std::marker::PhantomData<V>,
}

impl<T, V: Vec<T>> TetrahedralEstimator<T, V>
{
    /// Creates a `CentralDifferenceEstimator` with a given epsilon value.
    pub fn new(eps: T) -> Self { TetrahedralEstimator { eps, _pdv: std::marker::PhantomData } }
}

impl<T, V> NormalEstimator<T, V> for TetrahedralEstimator<T, V> 
where T: Add<T, Output=T> + Sub<T, Output=T> + Neg<Output=T> + One + Copy,
    V: Vec3<T>,
{
    fn estimate_normal<S: SDF<T, V>>(&self, sdf: S, p: V) -> V 
    {
        let xyy = V::new(T::one(), -T::one(), -T::one());
        let yyx = V::new(-T::one(), -T::one(),  T::one());
        let yxy = V::new(-T::one(),  T::one(), -T::one());
        let xxx = V::one();

        (xyy * sdf.dist(p + xyy * self.eps) +
            yyx * sdf.dist(p + xyy * self.eps) +
            yxy * sdf.dist(p + xyy * self.eps) +
            xxx * sdf.dist(p + xxx * self.eps)).normalized()
    }
}

impl<V: Vec<f32>> Default for TetrahedralEstimator<f32, V> {
    fn default() -> Self {
        Self::new(0.0001)
    }
}

impl<V: Vec<f64>> Default for TetrahedralEstimator<f64, V> {
    fn default() -> Self {
        Self::new(0.0001)
    }
}
