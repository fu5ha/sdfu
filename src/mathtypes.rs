//! Traits that have to implemented by vector and scalar traits to be used by this library.

use std::ops::*;

#[cfg(feature = "ultraviolet")]
use ultraviolet::f32x4;
#[cfg(feature = "ultraviolet")]
use ultraviolet::f32x8;

/// Functionality that should be shared between all vector types.
pub trait Vec<T>:
    Sized
    + Copy
    + Neg<Output = Self>
    + Mul<T, Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<T, Output = Self>
    + Div<T, Output = Self>
    + MaxMin
    + Zero
    + One
    + Clamp
{
    type Dimension: Dimension;
    type Vec2: Vec2<T>;
    type Vec3: Vec3<T>;
    fn dot(&self, other: Self) -> T;
    fn magnitude(&self) -> T;
    fn abs(&self) -> Self;
    fn normalized(&self) -> Self;
}

/// Functionality that must be implemented by 3D vectors.
pub trait Vec3<T>: Vec<T> {
    fn new(x: T, y: T, z: T) -> Self;
    fn x(&self) -> T;
    fn y(&self) -> T;
    fn z(&self) -> T;
}

/// Functionality that must be implemented by 2D vectors.
pub trait Vec2<T>: Vec<T> {
    fn new(x: T, y: T) -> Self;
    fn x(&self) -> T;
    fn y(&self) -> T;
}

/// A trait used to mark the dimensionality of a vector/SDF/implementation
/// of an SDF combinator.
pub trait Dimension {}

/// 2D marker struct.
#[derive(Clone, Copy, Debug)]
pub struct Dim2D {}

/// 3D marker struct.
#[derive(Clone, Copy, Debug)]
pub struct Dim3D {}

impl Dimension for Dim2D {}
impl Dimension for Dim3D {}

/// Return the maximum or minimum of `self` and `other`.
pub trait MaxMin {
    fn max(&self, other: Self) -> Self;
    fn min(&self, other: Self) -> Self;
}

macro_rules! impl_max_min {
    ($($scalar_t:ty),+) => {
        $(impl MaxMin for $scalar_t {
            #[inline]
            fn max(&self, other: Self) -> Self {
                <$scalar_t>::max(*self, other)
            }
            #[inline]
            fn min(&self, other: Self) -> Self {
                <$scalar_t>::min(*self, other)
            }
        })+
    }
}

impl_max_min!(f32, f64);
#[cfg(feature = "ultraviolet")]
impl_max_min!(f32x4, f32x8);

/// The multiplicative identity.
pub trait One {
    fn one() -> Self;
}

/// The additive identity.
pub trait Zero {
    fn zero() -> Self;
}

/// Multiply by half.
pub trait PointFive {
    fn point_five() -> Self;
}

macro_rules! impl_number_factory {
    ($trait_name:ident, $function_name:ident, $number:literal, $($scalar_t:ty),+) => {
        $(impl $trait_name for $scalar_t {
            #[inline]
            fn $function_name() -> Self {
                <$scalar_t>::from($number)
            }
        })+
    }
}

impl_number_factory!(One, one, 1.0, f32, f64);
#[cfg(feature = "ultraviolet")]
impl_number_factory!(One, one, 1.0, f32x4, f32x8);

impl_number_factory!(Zero, zero, 0.0, f32, f64);
#[cfg(feature = "ultraviolet")]
impl_number_factory!(Zero, zero, 0.0, f32x4, f32x8);

impl_number_factory!(PointFive, point_five, 0.5, f32, f64);
#[cfg(feature = "ultraviolet")]
impl_number_factory!(PointFive, point_five, 0.5, f32x4, f32x8);

/// Linear interpolate between self and other with a factor
/// between `Self::zero()` and `Self::one()`.
pub trait Lerp {
    fn lerp(&self, other: Self, factor: Self) -> Self;
}

impl<T> Lerp for T
where
    T: Copy + Mul<T, Output = T> + Sub<T, Output = T> + Add<T, Output = T> + One,
{
    #[inline]
    fn lerp(&self, other: Self, factor: Self) -> Self {
        *self * (T::one() - factor) + other * factor
    }
}

/// Clamp the value(s) of self to between `low` and `high`.
pub trait Clamp {
    fn clamp(&self, low: Self, high: Self) -> Self;
}

macro_rules! impl_clamp {
    ($($scalar_t:ty),+) => {
        $(impl Clamp for $scalar_t {
            #[inline]
            fn clamp(&self, low: Self, high: Self) -> Self {
                self.max(low).min(high)
            }
        })+
    }
}

impl_clamp!(f32, f64);
#[cfg(feature = "ultraviolet")]
impl_clamp!(f32x4, f32x8);

/// Raises `2^(self)`
pub trait Exp2 {
    fn exp2(&self) -> Self;
}

macro_rules! impl_exp2 {
    ($($scalar_t:ty),+) => {
        $(impl Exp2 for $scalar_t {
            #[inline]
            fn exp2(&self) -> Self {
                <$scalar_t>::exp2(*self)
            }
        })+
    }
}

impl_exp2!(f32, f64);

#[cfg(feature = "ultraviolet")]
impl Exp2 for f32x4 {
    #[inline]
    fn exp2(&self) -> Self {
        f32x4::from(2.0).pow_f32x4(*self)
    }
}

#[cfg(feature = "ultraviolet")]
impl Exp2 for f32x8 {
    #[inline]
    fn exp2(&self) -> Self {
        f32x8::from(2.0).pow_f32x8(*self)
    }
}

/// Returns log base 2 of self.
pub trait Log2 {
    fn log2(&self) -> Self;
}

macro_rules! impl_log2 {
    ($($scalar_t:ty),+) => {
        $(impl Log2 for $scalar_t {
            #[inline]
            fn log2(&self) -> Self {
                <$scalar_t>::log2(*self)
            }
        })+
    }
}

impl_log2!(f32, f64);
#[cfg(feature = "ultraviolet")]
impl_log2!(f32x4, f32x8);

/// This is a trait for types that can rotate an SDF.
/// Note that the implementation should actually rotate the vec
/// by the *inverse* of the native rotation that the type would
/// normally produce.
pub trait Rotation<V> {
    fn rotate_vec(&self, v: V) -> V;
}

#[cfg(feature = "vek")]
#[doc(hidden)]
pub mod vek_integration {
    use super::*;

    impl<T: vek::ops::Clamp + Copy> Clamp for vek::vec::Vec2<T> {
        #[inline]
        fn clamp(&self, low: Self, high: Self) -> Self {
            use vek::Clamp;
            self.clamped(low, high)
        }
    }

    impl<T: vek::ops::Clamp + Copy> Clamp for vek::vec::Vec3<T> {
        #[inline]
        fn clamp(&self, low: Self, high: Self) -> Self {
            use vek::Clamp;
            self.clamped(low, high)
        }
    }

    impl<T: Zero> Zero for vek::vec::Vec2<T> {
        #[inline]
        fn zero() -> Self {
            vek::vec::Vec2::new(T::zero(), T::zero())
        }
    }

    impl<T: Zero> Zero for vek::vec::Vec3<T> {
        #[inline]
        fn zero() -> Self {
            vek::vec::Vec3::new(T::zero(), T::zero(), T::zero())
        }
    }

    impl<T: One> One for vek::vec::Vec2<T> {
        #[inline]
        fn one() -> Self {
            vek::vec::Vec2::new(T::one(), T::one())
        }
    }

    impl<T: One> One for vek::vec::Vec3<T> {
        #[inline]
        fn one() -> Self {
            vek::vec::Vec3::new(T::one(), T::one(), T::one())
        }
    }

    impl<T: PartialOrd + Copy> MaxMin for vek::vec::Vec2<T> {
        #[inline]
        fn max(&self, other: Self) -> Self {
            vek::vec::Vec2::partial_max(*self, other)
        }

        #[inline]
        fn min(&self, other: Self) -> Self {
            vek::vec::Vec2::partial_min(*self, other)
        }
    }

    impl<T: PartialOrd + Copy> MaxMin for vek::vec::Vec3<T> {
        #[inline]
        fn max(&self, other: Self) -> Self {
            vek::vec::Vec3::partial_max(*self, other)
        }

        #[inline]
        fn min(&self, other: Self) -> Self {
            vek::vec::Vec3::partial_min(*self, other)
        }
    }

    macro_rules! impl_vec2 {
        ($($inner_t:ty),+) => {
            $(impl Vec2<$inner_t> for vek::vec::Vec2<$inner_t> {
                #[inline]
                fn new(x: $inner_t, y: $inner_t) -> Self {
                    vek::vec::Vec2::new(x, y)
                }
                #[inline]
                fn x(&self) -> $inner_t { self.x }
                #[inline]
                fn y(&self) -> $inner_t { self.y }
            })+
        }
    }
    impl_vec2!(f32, f64);

    macro_rules! impl_vec3 {
        ($($inner_t:ty),+) => {
            $(impl Vec3<$inner_t> for vek::vec::Vec3<$inner_t> {
                #[inline]
                fn new(x: $inner_t, y: $inner_t, z: $inner_t) -> Self {
                    vek::vec::Vec3::new(x, y, z)
                }
                #[inline]
                fn x(&self) -> $inner_t { self.x }
                #[inline]
                fn y(&self) -> $inner_t { self.y }
                #[inline]
                fn z(&self) -> $inner_t { self.z }
            })+
        }
    }
    impl_vec3!(f32, f64);

    macro_rules! impl_vec_vec2 {
        ($($inner_t:ty),+) => {
            $(impl Vec<$inner_t> for vek::vec::Vec2<$inner_t>
            {
                type Dimension = Dim2D;
                type Vec2 = Self;
                type Vec3 = vek::vec::Vec3<$inner_t>;
                #[inline]
                fn dot(&self, other: Self) -> $inner_t {
                    vek::vec::Vec2::dot(*self, other)
                }

                #[inline]
                fn magnitude(&self) -> $inner_t {
                    vek::vec::Vec2::magnitude(*self)
                }

                #[inline]
                fn abs(&self) -> Self {
                    vek::vec::Vec2::new(self.x.abs(), self.y.abs())
                }

                #[inline]
                fn normalized(&self) -> Self {
                    vek::vec::Vec2::normalized(*self)
                }
            })+
        }
    }
    impl_vec_vec2!(f32, f64);

    macro_rules! impl_vec_vec3 {
        ($($inner_t:ty),+) => {
            $(impl Vec<$inner_t> for vek::vec::Vec3<$inner_t>
            {
                type Dimension = Dim3D;
                type Vec2 = vek::vec::Vec2<$inner_t>;
                type Vec3 = Self;
                #[inline]
                fn dot(&self, other: Self) -> $inner_t {
                    vek::vec::Vec3::dot(*self, other)
                }

                #[inline]
                fn magnitude(&self) -> $inner_t {
                    vek::vec::Vec3::magnitude(*self)
                }

                #[inline]
                fn abs(&self) -> Self {
                    vek::vec::Vec3::new(self.x.abs(), self.y.abs(), self.z.abs())
                }

                #[inline]
                fn normalized(&self) -> Self {
                    vek::vec::Vec3::normalized(*self)
                }
            })+
        }
    }
    impl_vec_vec3!(f32, f64);

    macro_rules! impl_rotation_mat {
        {$($outer_t:ty => $inner_t:ty),+} => {
            $(impl Rotation<$inner_t> for $outer_t {
                #[inline]
                fn rotate_vec(&self, v: $inner_t) -> $inner_t {
                    <$inner_t>::from(vek::mat::Mat4::from(*self).inverted() * vek::vec::Vec4::from(v))
                }
            })+
        }
    }

    impl_rotation_mat! {
        vek::mat::Mat3<f32> => vek::vec::Vec2<f32>,
        vek::mat::Mat2<f32> => vek::vec::Vec2<f32>,
        vek::mat::Mat3<f32> => vek::vec::Vec3<f32>,
        vek::mat::Mat4<f32> => vek::vec::Vec3<f32>,
        vek::mat::Mat3<f64> => vek::vec::Vec2<f64>,
        vek::mat::Mat2<f64> => vek::vec::Vec2<f64>,
        vek::mat::Mat3<f64> => vek::vec::Vec3<f64>,
        vek::mat::Mat4<f64> => vek::vec::Vec3<f64>
    }

    macro_rules! impl_rotation_quat {
        {$($outer_t:ty => $inner_t:ty),+} => {
            $(impl Rotation<$inner_t> for $outer_t {
                #[inline]
                fn rotate_vec(&self, v: $inner_t) -> $inner_t {
                    <$inner_t>::from(self.inverse() * vek::vec::Vec4::from(v))
                }
            })+
        }
    }

    impl_rotation_quat! {
        vek::quaternion::Quaternion<f32> => vek::vec::Vec2<f32>,
        vek::quaternion::Quaternion<f32> => vek::vec::Vec3<f32>,
        vek::quaternion::Quaternion<f64> => vek::vec::Vec2<f64>,
        vek::quaternion::Quaternion<f64> => vek::vec::Vec3<f64>
    }
}

#[cfg(feature = "ultraviolet")]
#[doc(hidden)]
pub mod ultraviolet_integration {
    use super::*;
    use ultraviolet as uv;

    macro_rules! impl_numerics {
        ($($vt:ty),+) => {
            $(
            impl Zero for $vt {
                #[inline]
                fn zero() -> Self {
                    Self::zero()
                }
            }

            impl One for $vt {
                #[inline]
                fn one() -> Self {
                    Self::one()
                }
            }

            impl MaxMin for $vt {
                #[inline]
                fn max(&self, other: Self) -> Self {
                    self.max_by_component(other)
                }

                #[inline]
                fn min(&self, other: Self) -> Self {
                    self.min_by_component(other)
                }
            }

            impl Clamp for $vt {
                #[inline]
                fn clamp(&self, low: Self, high: Self) -> Self {
                    self.clamped(low, high)
                }
            }
            )+
        };
    }

    impl_numerics!(
        uv::Vec2,
        uv::Vec2x4,
        uv::Vec2x8,
        uv::Vec3,
        uv::Vec3x4,
        uv::Vec3x8
    );

    macro_rules! impl_vec2 {
        ($($vt:ty => $t:ty),+) => {
            $(
            impl Vec2<$t> for $vt {
                #[inline]
                fn new(x: $t, y: $t) -> Self {
                    <$vt>::new(x, y)
                }
                #[inline]
                fn x(&self) -> $t { self.x }
                #[inline]
                fn y(&self) -> $t { self.y }
            }
            )+
        }
    }
    impl_vec2!(uv::Vec2 => f32, uv::Vec2x4 => f32x4, uv::Vec2x8 => f32x8);

    macro_rules! impl_vec3 {
        ($($vt:ty => $t:ty),+) => {
            $(impl Vec3<$t> for $vt {
                #[inline]
                fn new(x: $t, y: $t, z: $t) -> Self {
                    <$vt>::new(x, y, z)
                }
                #[inline]
                fn x(&self) -> $t { self.x }
                #[inline]
                fn y(&self) -> $t { self.y }
                #[inline]
                fn z(&self) -> $t { self.z }
            })+
        }
    }
    impl_vec3!(uv::Vec3 => f32, uv::Vec3x4 => f32x4, uv::Vec3x8 => f32x8);

    macro_rules! impl_vec_vec2 {
        ($($vt:ty, $v3t:ty => $t:ty),+) => {
            $(impl Vec<$t> for $vt
            {
                type Dimension = Dim2D;
                type Vec2 = Self;
                type Vec3 = $v3t;
                #[inline]
                fn dot(&self, other: Self) -> $t {
                    self.dot(other)
                }

                #[inline]
                fn magnitude(&self) -> $t {
                    self.mag()
                }

                #[inline]
                fn abs(&self) -> Self {
                    <$vt>::new(self.x.abs(), self.y.abs())
                }

                #[inline]
                fn normalized(&self) -> Self {
                    self.normalized()
                }
            })+
        }
    }
    impl_vec_vec2!(uv::Vec2, uv::Vec3 => f32, uv::Vec2x4, uv::Vec3x4 => f32x4, uv::Vec2x8, uv::Vec3x8 => f32x8);

    macro_rules! impl_vec_vec3 {
        ($($vt:ty, $v2t:ty => $t:ty),+) => {
            $(impl Vec<$t> for $vt
            {
                type Dimension = Dim3D;
                type Vec2 = $v2t;
                type Vec3 = Self;
                #[inline]
                fn dot(&self, other: Self) -> $t {
                    self.dot(other)
                }

                #[inline]
                fn magnitude(&self) -> $t {
                    self.mag()
                }

                #[inline]
                fn abs(&self) -> Self {
                    self.abs()
                }

                #[inline]
                fn normalized(&self) -> Self {
                    self.normalized()
                }
            })+
        }
    }
    impl_vec_vec3!(uv::Vec3, uv::Vec2 => f32, uv::Vec3x4, uv::Vec2x4 => f32x4, uv::Vec3x8, uv::Vec2x8 => f32x8);

    macro_rules! impl_rotation_rotor {
        {$($rt:ty => $vt:ty),+} => {
            $(impl Rotation<$vt> for $rt {
                #[inline]
                fn rotate_vec(&self, v: $vt) -> $vt {
                    self.reversed() * v
                }
            })+
        }
    }

    impl_rotation_rotor! {
        uv::Rotor2 => uv::Vec2,
        uv::Rotor2x4 => uv::Vec2x4,
        uv::Rotor2x8 => uv::Vec2x8,
        uv::Rotor3 => uv::Vec3,
        uv::Rotor3x4 => uv::Vec3x4,
        uv::Rotor3x8 => uv::Vec3x8
    }
}

#[cfg(feature = "nalgebra")]
#[doc(hidden)]
pub mod nalgebra_integration {
    use super::*;
    use nalgebra as na;
    use std::fmt::Debug;

    impl<T: PartialOrd + Copy + Debug + 'static> Clamp for na::Vector2<T> {
        #[inline]
        fn clamp(&self, low: Self, high: Self) -> Self {
            na::Vector2::new(
                na::clamp(self.x, low.x, high.x),
                na::clamp(self.y, low.y, high.y),
            )
        }
    }

    impl<T: PartialOrd + Copy + Debug + 'static> Clamp for na::Vector3<T> {
        #[inline]
        fn clamp(&self, low: Self, high: Self) -> Self {
            na::Vector3::new(
                na::clamp(self.x, low.x, high.x),
                na::clamp(self.y, low.y, high.y),
                na::clamp(self.z, low.z, high.z),
            )
        }
    }

    impl<T: Zero + PartialEq + Copy + Debug + 'static> Zero for na::Vector2<T> {
        #[inline]
        fn zero() -> Self {
            na::Vector2::new(T::zero(), T::zero())
        }
    }

    impl<T: Zero + PartialEq + Copy + Debug + 'static> Zero for na::Vector3<T> {
        #[inline]
        fn zero() -> Self {
            na::Vector3::new(T::zero(), T::zero(), T::zero())
        }
    }

    impl<T: One + PartialEq + Copy + Debug + 'static> One for na::Vector2<T> {
        fn one() -> Self {
            na::Vector2::new(T::one(), T::one())
        }
    }

    impl<T: One + PartialEq + Copy + Debug + 'static> One for na::Vector3<T> {
        #[inline]
        fn one() -> Self {
            na::Vector3::new(T::one(), T::one(), T::one())
        }
    }

    impl<T: PartialOrd + Copy + Debug + 'static> MaxMin for na::Vector2<T> {
        #[inline]
        fn max(&self, other: Self) -> Self {
            na::Vector2::new(
                *na::partial_max(&self.x, &other.x).unwrap_or(&self.x),
                *na::partial_max(&self.y, &other.y).unwrap_or(&self.y),
            )
        }

        #[inline]
        fn min(&self, other: Self) -> Self {
            na::Vector2::new(
                *na::partial_min(&self.x, &other.x).unwrap_or(&self.x),
                *na::partial_min(&self.y, &other.y).unwrap_or(&self.y),
            )
        }
    }

    impl<T: PartialOrd + Copy + Debug + 'static> MaxMin for na::Vector3<T> {
        #[inline]
        fn max(&self, other: Self) -> Self {
            na::Vector3::new(
                *na::partial_max(&self.x, &other.x).unwrap_or(&self.x),
                *na::partial_max(&self.y, &other.y).unwrap_or(&self.y),
                *na::partial_max(&self.z, &other.z).unwrap_or(&self.z),
            )
        }

        #[inline]
        fn min(&self, other: Self) -> Self {
            na::Vector3::new(
                *na::partial_min(&self.x, &other.x).unwrap_or(&self.x),
                *na::partial_min(&self.y, &other.y).unwrap_or(&self.y),
                *na::partial_min(&self.z, &other.z).unwrap_or(&self.z),
            )
        }
    }

    macro_rules! impl_vec2 {
        ($($inner_t:ty),+) => {
            $(impl Vec2<$inner_t> for na::Vector2<$inner_t> {
                #[inline]
                fn new(x: $inner_t, y: $inner_t) -> Self {
                    na::Vector2::new(x, y)
                }
                #[inline]
                fn x(&self) -> $inner_t { self.x }
                #[inline]
                fn y(&self) -> $inner_t { self.y }
            })+
        }
    }
    impl_vec2!(f32, f64);

    macro_rules! impl_vec3 {
        ($($inner_t:ty),+) => {
            $(impl Vec3<$inner_t> for na::Vector3<$inner_t> {
                #[inline]
                fn new(x: $inner_t, y: $inner_t, z: $inner_t) -> Self {
                    na::Vector3::new(x, y, z)
                }
                #[inline]
                fn x(&self) -> $inner_t { self.x }
                #[inline]
                fn y(&self) -> $inner_t { self.y }
                #[inline]
                fn z(&self) -> $inner_t { self.z }
            })+
        }
    }
    impl_vec3!(f32, f64);

    macro_rules! impl_vec_vec2 {
        ($($inner_t:ty),+) => {
            $(impl Vec<$inner_t> for na::Vector2<$inner_t>
            {
                type Dimension = Dim2D;
                type Vec2 = Self;
                type Vec3 = na::Vector3<$inner_t>;
                #[inline]
                fn dot(&self, other: Self) -> $inner_t {
                    na::Vector2::dot(self, &other)
                }

                #[inline]
                fn magnitude(&self) -> $inner_t {
                    na::Vector2::magnitude(self)
                }

                #[inline]
                fn abs(&self) -> Self {
                    na::Vector2::new(self.x.abs(), self.y.abs())
                }

                #[inline]
                fn normalized(&self) -> Self {
                    na::Vector2::normalize(self)
                }
            })+
        }
    }
    impl_vec_vec2!(f32, f64);

    macro_rules! impl_vec_vec3 {
        ($($inner_t:ty),+) => {
            $(impl Vec<$inner_t> for na::Vector3<$inner_t>
            {
                type Dimension = Dim3D;
                type Vec2 = na::Vector2<$inner_t>;
                type Vec3 = Self;
                #[inline]
                fn dot(&self, other: Self) -> $inner_t {
                    na::Vector3::dot(self, &other)
                }

                #[inline]
                fn magnitude(&self) -> $inner_t {
                    na::Vector3::magnitude(self)
                }

                #[inline]
                fn abs(&self) -> Self {
                    na::Vector3::new(self.x.abs(), self.y.abs(), self.z.abs())
                }

                #[inline]
                fn normalized(&self) -> Self {
                    na::Vector3::normalize(self)
                }
            })+
        }
    }
    impl_vec_vec3!(f32, f64);

    macro_rules! impl_rot_inner {
        ($($rot_ty:ty => $vec_ty:ty),+) => {
            $(impl Rotation<$vec_ty> for $rot_ty {
                #[inline]
                fn rotate_vec(&self, v: $vec_ty) -> $vec_ty {
                    self.inverse_transform_vector(&v)
                }
            })+
        }
    }

    macro_rules! impl_rot {
        ($($inner_ty:ty),+) => {
            $(impl_rot_inner!(
                na::Rotation2<$inner_ty> => na::Vector2<$inner_ty>,
                na::UnitComplex<$inner_ty> => na::Vector2<$inner_ty>,
                na::Rotation3<$inner_ty> => na::Vector3<$inner_ty>,
                na::UnitQuaternion<$inner_ty> => na::Vector3<$inner_ty>
            );)+
        }
    }
    impl_rot!(f32, f64);
}

#[cfg(feature = "glam")]
#[doc(hidden)]
pub mod glam_integration {
    use super::*;

    use glam as gl;

    impl Zero for gl::Vec2 {
        #[inline]
        fn zero() -> Self {
            Self::ZERO
        }
    }

    impl One for gl::Vec2 {
        #[inline]
        fn one() -> Self {
            gl::Vec2::new(1.0, 1.0)
        }
    }

    impl Clamp for gl::Vec2 {
        #[inline]
        fn clamp(&self, low: Self, high: Self) -> Self {
            self.min(high).max(low)
        }
    }

    impl MaxMin for gl::Vec2 {
        #[inline]
        fn max(&self, other: Self) -> Self {
            gl::Vec2::max(*self, other)
        }
        #[inline]
        fn min(&self, other: Self) -> Self {
            gl::Vec2::min(*self, other)
        }
    }

    impl Zero for gl::Vec3A {
        #[inline]
        fn zero() -> Self {
            gl::Vec3A::ZERO
        }
    }

    impl One for gl::Vec3A {
        #[inline]
        fn one() -> Self {
            gl::Vec3A::new(1.0, 1.0, 1.0)
        }
    }

    impl Clamp for gl::Vec3A {
        #[inline]
        fn clamp(&self, low: Self, high: Self) -> Self {
            self.min(high).max(low)
        }
    }

    impl MaxMin for gl::Vec3A {
        #[inline]
        fn max(&self, other: Self) -> Self {
            gl::Vec3A::max(*self, other)
        }
        #[inline]
        fn min(&self, other: Self) -> Self {
            gl::Vec3A::min(*self, other)
        }
    }

    impl Vec2<f32> for gl::Vec2 {
        #[inline]
        fn new(x: f32, y: f32) -> Self {
            gl::Vec2::new(x, y)
        }
        #[inline]
        fn x(&self) -> f32 {
            self.x
        }
        #[inline]
        fn y(&self) -> f32 {
            self.y
        }
    }

    impl Vec3<f32> for gl::Vec3A {
        #[inline]
        fn new(x: f32, y: f32, z: f32) -> Self {
            gl::Vec3A::new(x, y, z)
        }
        #[inline]
        fn x(&self) -> f32 {
            self.x
        }
        #[inline]
        fn y(&self) -> f32 {
            self.y
        }
        #[inline]
        fn z(&self) -> f32 {
            self.z
        }
    }

    impl Vec<f32> for gl::Vec2 {
        type Dimension = Dim2D;
        type Vec2 = gl::Vec2;
        type Vec3 = gl::Vec3A;

        #[inline]
        fn dot(&self, other: Self) -> f32 {
            gl::Vec2::dot(*self, other)
        }

        #[inline]
        fn abs(&self) -> Self {
            gl::Vec2::abs(*self)
        }

        #[inline]
        fn normalized(&self) -> Self {
            *self / self.length()
        }

        #[inline]
        fn magnitude(&self) -> f32 {
            self.length()
        }
    }

    impl Vec<f32> for gl::Vec3A {
        type Dimension = Dim3D;
        type Vec2 = gl::Vec2;
        type Vec3 = gl::Vec3A;

        #[inline]
        fn dot(&self, other: Self) -> f32 {
            gl::Vec3A::dot(*self, other)
        }

        #[inline]
        fn abs(&self) -> Self {
            gl::Vec3A::abs(*self)
        }

        #[inline]
        fn normalized(&self) -> Self {
            *self / self.length()
        }

        #[inline]
        fn magnitude(&self) -> f32 {
            self.length()
        }
    }
}
