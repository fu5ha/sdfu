use super::*;
use crate::mathtypes::*;
use std::ops::*;

/// Estimate the normal of the SDF `sdf` at point `p`
pub trait NormalEstimator<T, V> {
    fn estimate_normal<S: SDF<T, V>>(&self, sdf: S, p: V) -> V;
}

pub struct CentralDifferenceEstimator<T, V, D> {
    pub eps: T,
    _pdv: std::marker::PhantomData<V>,
    _pdd: std::marker::PhantomData<D>,
}

impl<T, V: Vec<T>> CentralDifferenceEstimator<T, V, <V as Vec<T>>::Dimension>
{
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
    pub fn normal_at(&self, p: V) -> V {
        self.estimator.estimate_normal(self.sdf, p)
    }
}
