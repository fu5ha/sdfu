# `sdfu` - Signed Distance Field Utilities 

This is a small crate designed to help when working with signed distance fields
in the context of computer graphics, especially ray-marching based renderers. Most
of what is here is based on [Inigo Quilez' excellent articles](http://www.iquilezles.org/www/index.htm).

If you're using one of the more popular math libraries in Rust, then just enable
the corresponding feature and hopefully all the necessary traits are already implemented
for you so that you can just start passing in your `Vec3`s or whatever your lib calls them
and you're off to the races! If not, then you can implement the necessary traits in the
`mathtypes` module and still use this library with your own math lib.