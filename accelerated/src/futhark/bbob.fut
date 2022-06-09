import "raw"
import "math"

module t = import "transform"

entry sphere (x: []f64) (xopt: []f64) (fopt: f64): f64 =
    x
    |> map2 subbed xopt
    |> sphere_raw
    |> (+fopt)

entry ellipsoidal (x: []f64) (xopt: []f64) (fopt: f64): f64 =
    x
    |> map2 subbed xopt
    |> t.x_osz
    |> ellipsoidal_raw
    |> (+fopt)
