#[cfg(feature = "nalgebra")]
#[test]
fn test_nalgebra() {
    use sdfu::SDF;
    let sdf = sdfu::Sphere::new(1.0);
    let dist: f32 = sdf.dist(nalgebra::Vector3::zeros());
    assert_eq!(dist, -1.0);
}

#[cfg(feature = "ultraviolet")]
#[test]
fn test_ultraviolet() {
    use sdfu::SDF;
    let sdf = sdfu::Sphere::new(1.0);
    let dist: f32 = sdf.dist(ultraviolet::Vec3::zero());
    assert_eq!(dist, -1.0);
}

#[cfg(feature = "vek")]
#[test]
fn test_vek() {
    use sdfu::SDF;
    let sdf = sdfu::Sphere::new(1.0);
    let dist: f32 = sdf.dist(vek::vec::Vec3::zero());
    assert_eq!(dist, -1.0);
}

#[cfg(feature = "glam")]
#[test]
fn test_glam() {
    use sdfu::SDF;
    let sdf = sdfu::Sphere::new(1.0);
    let dist: f32 = sdf.dist(glam::Vec3A::ZERO);
    assert_eq!(dist, -1.0);
}
