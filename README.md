# `sdfu` - Signed Distance Field Utilities 

This is a small crate designed to help when working with signed distance fields
in the context of computer graphics, especially ray-marching based renderers. Most
of what is here is based on [Inigo Quilez' excellent articles](http://www.iquilezles.org/www/index.htm).

If you're using one of the more popular math libraries in Rust, then just enable the corresponding
feature (currently, [`ultraviolet`](https://github.com/termhn/ultraviolet), `nalgebra` and `vek`
are supported) and hopefully all the necessary traits are already implemented for you so that
you can just start passing in your `Vec3`s or whatever your lib calls them and you're off to the
races! If not, then you can implement the necessary traits in the `mathtypes` module and still use
this library with your own math lib.

# Demo

![demo image](/demo.png?raw=true)

The image above was rendered with my own path tracing renderer, [`rayn`](https://github.com/termhn/rayn)
by leveraging `sdfu`. The SDF that is rendered above was created with the following code:

```rust
use sdfu::SDF;
use ultraviolet::Vec3;

let sdf = sdfu::Sphere::new(0.45)
    .subtract(
        sdfu::Box::new(Vec3::new(0.25, 0.25, 1.5)))
    .union_smooth(
        sdfu::Sphere::new(0.3).translate(Vec3::new(0.3, 0.3, 0.0)),
        0.1)
    .union_smooth(
        sdfu::Sphere::new(0.3).translate(Vec3::new(-0.3, 0.3, 0.0)),
        0.1)
    .subtract(
        sdfu::Box::new(Vec3::new(0.125, 0.125, 1.5)).translate(Vec3::new(-0.3, 0.3, 0.0)))
    .subtract(
        sdfu::Box::new(Vec3::new(0.125, 0.125, 1.5)).translate(Vec3::new(0.3, 0.3, 0.0)))
    .subtract(
        sdfu::Box::new(Vec3::new(1.5, 0.1, 0.1)).translate(Vec3::new(0.0, 0.3, 0.0)))
    .subtract(
        sdfu::Box::new(Vec3::new(0.2, 2.0, 0.2)))
    .translate(Vec3::new(0.0, 0.0, -1.0));
```