import "math"

-- f01: sphere

def sphere (x: []f64): f64 =
    f64.sum (map2 (*) x x)

-- f02: ellipsoidal

local def ellipsoidal_factor (x: []f64) (i: i64): f64 =
    (10 ** (6 * (idr i (length x)))) * (x[i] ** 2)

def ellipsoidal (x: []f64): f64 =
    f64.sum (map (ellipsoidal_factor x) (indices x))

-- f03: rastrigin

def rastrigin (x: []f64): f64 =
    10 * ((dim x) - f64.sum (x |> map (2 * f64.pi *) |> map f64.cos)) + sphere(x)

-- f07: step_ellipsoidal

local def step_ellipsoidal_factor (x: []f64) (i: i64): f64 =
    (10 ** (2 * (idr i (length x)))) * (x[i] ** 2)

def step_ellipsoidal (x: []f64): f64 =
    f64.sum (map (step_ellipsoidal_factor x) (indices x))

-- f08: rosenbrock

def rosenbrock (x: []f64): f64 =
    let n = length x in
    (0 ..< (n - 1))
    |> map (\i -> 100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2)
    |> f64.sum

-- f11: discus

def discus (x: []f64): f64 =
    (10**6) * (x[0]**2) + sphere x[1:]

-- f11: bent_cigar

def bent_cigar (x: []f64): f64 =
    (x[0]**2) + (10**6) * (sphere x[1:])

-- f13: sharp_ridge

def sharp_ridge (x: []f64): f64 =
    (x[0]**2) + 100 * f64.sqrt (sphere x[1:])

-- f14: different_powers

local def different_powers_factor (x: []f64) (i: i64): f64 =
    (f64.abs x[i]) ** (2 + 4 * (idr i (length x)))

def different_powers (x: []f64): f64 =
    f64.sqrt (f64.sum (map (different_powers_factor x) (indices x)))
