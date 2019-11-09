use std::ops::*;

#[cfg(feature = "ultraviolet")]
use ultraviolet::f32x4;

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

/// Functionality that must be implmeented by 3D vectors.
pub trait Vec3<T>: Vec<T> {
    fn new(x: T, y: T, z: T) -> Self;
    fn x(&self) -> T;
    fn y(&self) -> T;
    fn z(&self) -> T;
}

/// Functionality that must be implmeented by 2D vectors.
pub trait Vec2<T>: Vec<T> {
    fn new(x: T, y: T) -> Self;
    fn x(&self) -> T;
    fn y(&self) -> T;
}

/// A trait used to mark the dimensionality of a vector/SDF/implmentation
/// of an SDF combinator.
pub trait Dimension {}

/// 2D marker struct.
#[derive(Clone, Copy, Debug)]
pub struct Dim2D {}

/// 2D marker struct.
#[derive(Clone, Copy, Debug)]
pub struct Dim3D {}

impl Dimension for Dim2D {}
impl Dimension for Dim3D {}

/// Return the maximum or minimum of `self` and `other`.
pub trait MaxMin {
    fn max(&self, other: Self) -> Self;
    fn min(&self, other: Self) -> Self;
}

impl MaxMin for f32 {
    #[inline]
    fn max(&self, other: Self) -> Self {
        f32::max(*self, other)
    }
    #[inline]
    fn min(&self, other: Self) -> Self {
        f32::min(*self, other)
    }
}

impl MaxMin for f64 {
    #[inline]
    fn max(&self, other: Self) -> Self {
        f64::max(*self, other)
    }
    #[inline]
    fn min(&self, other: Self) -> Self {
        f64::min(*self, other)
    }
}

#[cfg(feature = "ultraviolet")]
impl MaxMin for f32x4 {
    #[inline]
    fn max(&self, other: Self) -> Self {
        f32x4::max(*self, other)
    }
    #[inline]
    fn min(&self, other: Self) -> Self {
        f32x4::min(*self, other)
    }
}

/// The multiplicative identity.
pub trait One {
    fn one() -> Self;
}

#[cfg(feature = "ultraviolet")]
impl One for f32x4 {
    #[inline]
    fn one() -> Self {
        f32x4::from(1.0)
    }
}

impl One for f32 {
    #[inline]
    fn one() -> Self {
        1.0
    }
}

impl One for f64 {
    #[inline]
    fn one() -> Self {
        1.0
    }
}

/// The additive identity.
pub trait Zero {
    fn zero() -> Self;
}

#[cfg(feature = "ultraviolet")]
impl Zero for f32x4 {
    #[inline]
    fn zero() -> Self {
        f32x4::from(0.0)
    }
}

impl Zero for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
}

impl Zero for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
}

/// Multiply by half.
pub trait PointFive {
    fn point_five() -> Self;
}

#[cfg(feature = "ultraviolet")]
impl PointFive for f32x4 {
    #[inline]
    fn point_five() -> Self {
        f32x4::from(0.5)
    }
}

impl PointFive for f32 {
    #[inline]
    fn point_five() -> Self {
        0.5
    }
}

impl PointFive for f64 {
    #[inline]
    fn point_five() -> Self {
        0.5
    }
}

/// Linear interpolate between self and other with a factor
/// between Self::zero() and Self::one.
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

#[cfg(feature = "ultraviolet")]
impl Clamp for f32x4 {
    #[inline]
    fn clamp(&self, low: Self, high: Self) -> Self {
        self.max(low).min(high)
    }
}

/// Raises `2^(self)`
pub trait Exp2 {
    fn exp2(&self) -> Self;
}

#[cfg(feature = "ultraviolet")]
impl Exp2 for f32x4 {
    #[inline]
    fn exp2(&self) -> Self {
        f32x4::exp2(*self)
    }
}

impl Exp2 for f32 {
    #[inline]
    fn exp2(&self) -> Self {
        f32::exp2(*self)
    }
}

impl Exp2 for f64 {
    #[inline]
    fn exp2(&self) -> Self {
        f64::exp2(*self)
    }
}

/// Returns log base 2 of self.
pub trait Log2 {
    fn log2(&self) -> Self;
}

#[cfg(feature = "ultraviolet")]
impl Log2 for f32x4 {
    #[inline]
    fn log2(&self) -> Self {
        f32x4::log2(*self)
    }
}

impl Log2 for f32 {
    #[inline]
    fn log2(&self) -> Self {
        f32::log2(*self)
    }
}

impl Log2 for f64 {
    #[inline]
    fn log2(&self) -> Self {
        f64::log2(*self)
    }
}

/// This is a trait for types that can rotate an SDF.
/// Note that the implementation should actually rotate the vec
/// by the *inverse* of the native rotation that the type would
/// normally produce.
pub trait Rotation<V> {
    fn rotate_vec(&self, v: V) -> V;
}

#[cfg(feature = "vek")]
pub mod vek_integration {
    use super::*;

    impl<T: vek::ops::Clamp + Copy> Clamp for T {
        #[inline]
        fn clamp(&self, low: T, high: T) -> T {
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

    impl_numerics!(uv::Vec2, uv::Wec2, uv::Vec3, uv::Wec3);

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
    impl_vec2!(uv::Vec2 => f32, uv::Wec2 => f32x4);

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
    impl_vec3!(uv::Vec3 => f32, uv::Wec3 => f32x4);

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
    impl_vec_vec2!(uv::Vec2, uv::Vec3 => f32, uv::Wec2, uv::Wec3 => f32x4);

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
    impl_vec_vec3!(uv::Vec3, uv::Vec2 => f32, uv::Wec3, uv::Wec2 => f32x4);

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
        uv::WRotor2 => uv::Wec2,
        uv::Rotor3 => uv::Vec3,
        uv::WRotor3 => uv::Wec3
    }
}

#[cfg(not(feature = "vek"))]
impl Clamp for f32 {
    #[inline]
    fn clamp(&self, low: Self, high: Self) -> Self {
        self.max(low).min(high)
    }
}

#[cfg(not(feature = "vek"))]
impl Clamp for f64 {
    #[inline]
    fn clamp(&self, low: Self, high: Self) -> Self {
        self.max(low).min(high)
    }
}

#[cfg(feature = "nalgebra")]
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
