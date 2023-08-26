use ordered_float::OrderedFloat;

use crate::bbob::Function;

#[derive(Debug, Clone)]
pub enum Params {
    Basic {
        fopt: f64,
        xopt: Vec<f64>,
    },
    Rotated {
        fopt: f64,
        xopt: Vec<f64>,
        R: coco_legacy::Matrix,
    },
    FixedRotated {
        fopt: f64,
        R: coco_legacy::Matrix,
    },
    DoubleRotated {
        fopt: f64,
        xopt: Vec<f64>,
        R: coco_legacy::Matrix,
        Q: coco_legacy::Matrix,
    },
    Gallagher {
        fopt: f64,
        peaks: usize,
        y: Vec<f64>,
        w: Vec<f64>,
        c: Vec<f64>,
        R: coco_legacy::Matrix,
    },
}

impl Params {
    pub fn fopt(&self) -> f64 {
        *match self {
            Params::Basic { fopt, .. } => fopt,
            Params::Rotated { fopt, .. } => fopt,
            Params::FixedRotated { fopt, .. } => fopt,
            Params::DoubleRotated { fopt, .. } => fopt,
            Params::Gallagher { fopt, .. } => fopt,
        }
    }
}

fn precompute_matrix_multiplication_with_conditioning(
    rot1: &coco_legacy::Matrix,
    rot2: &coco_legacy::Matrix,
    conditioning: f64,
) -> coco_legacy::Matrix {
    assert_eq!(rot1.dimension, rot2.dimension);

    let dim = rot1.dimension;
    let mut m = coco_legacy::Matrix::new(dim);

    for i in 0..dim {
        let current_row = &mut m[i];
        for j in 0..dim {
            current_row[j] = 0.0;
            for k in 0..dim {
                let base = f64::sqrt(conditioning);
                let exponent = k as f64 / (dim - 1) as f64;
                current_row[j] += rot1[i][k] * f64::powf(base, exponent) * rot2[k][j];
            }
        }
    }

    m
}

impl Params {
    pub fn new(function: Function, dimension: usize, instance: usize) -> Self {
        let rseed: usize = function as usize + 10000 * instance;
        let rseed_3: usize = 3 + 10000 * instance;
        let rseed_17: usize = 17 + 10000 * instance;

        // These (as well as bent_cigar) use a different rseed.
        let rseed = match function {
            Function::BuecheRastrigin => rseed_3,
            Function::Schaffers2 => rseed_17,
            _ => rseed,
        };

        let mut xopt = coco_legacy::compute_xopt(rseed, dimension);
        let fopt = coco_legacy::compute_fopt(function as usize, instance);

        // Special cases for some functions.
        match function {
            Function::BuecheRastrigin => {
                // OME: This step is in the legacy C code but _not_ in the function description.
                for xi in xopt.iter_mut().step_by(2) {
                    *xi = xi.abs();
                }
            }
            Function::LinearSlope => {
                for xi in &mut xopt {
                    *xi = if *xi >= 0.0 { 5.0 } else { -5.0 };
                }
            }
            Function::Rosenbrock => {
                // According to the documentation, xopt should be within [-3, 3],
                // but this is not enough to satisfy that condition...
                for xi in &mut xopt {
                    *xi *= 0.75;
                }
            }
            Function::BentCigar => {
                // No clue why they did this, probably it was a typo?
                xopt = coco_legacy::compute_xopt(rseed + 1000000, dimension);
            }
            Function::LunacekBiRastrigin => {
                xopt = coco_legacy::compute_gauss(rseed, dimension);
                for xi in &mut xopt {
                    *xi = if *xi >= 0.0 { 1.0 } else { -1.0 };
                }
            }
            Function::Schwefel => {
                xopt = coco_legacy::compute_unif(rseed, dimension);
                for xi in &mut xopt {
                    *xi = if *xi >= 0.5 { 1.0 } else { -1.0 };
                    *xi *= 0.5 * 4.2096874637;
                }
            }
            _ => {}
        }

        match function {
            Function::Sphere
            | Function::Ellipsoid
            | Function::Rastrigin
            | Function::BuecheRastrigin
            | Function::LinearSlope
            | Function::Rosenbrock
            | Function::Schwefel => Params::Basic { fopt, xopt },

            Function::EllipsoidRotated
            | Function::Discus
            | Function::BentCigar
            | Function::DifferentPowers => {
                let R = coco_legacy::compute_rotation(rseed + 1000000, dimension);
                Params::Rotated { fopt, xopt, R }
            }

            Function::StepEllipsoid
            | Function::RastriginRotated
            | Function::Schaffers1
            | Function::Schaffers2
            | Function::Katsuura => {
                let R = coco_legacy::compute_rotation(rseed + 1000000, dimension);
                let Q = coco_legacy::compute_rotation(rseed, dimension);
                Params::DoubleRotated { fopt, xopt, R, Q }
            }

            Function::LunacekBiRastrigin => {
                let R = coco_legacy::compute_rotation(rseed + 1000000, dimension);
                let Q = coco_legacy::compute_rotation(rseed, dimension);
                let M = precompute_matrix_multiplication_with_conditioning(&R, &Q, 100.0);
                Params::Rotated { fopt, xopt, R: M }
            }

            Function::AttractiveSector | Function::SharpRidge => {
                let R = coco_legacy::compute_rotation(rseed + 1000000, dimension);
                let Q = coco_legacy::compute_rotation(rseed, dimension);
                let M = precompute_matrix_multiplication_with_conditioning(&R, &Q, 10.0);
                Params::Rotated { fopt, xopt, R: M }
            }

            Function::Weierstrass => {
                let R = coco_legacy::compute_rotation(rseed + 1000000, dimension);
                let Q = coco_legacy::compute_rotation(rseed, dimension);
                let M = precompute_matrix_multiplication_with_conditioning(&R, &Q, 0.01);
                Params::DoubleRotated {
                    fopt,
                    xopt,
                    R,
                    Q: M,
                }
            }

            Function::RosenbrockRotated | Function::GriewankRosenbrock => {
                let Q = coco_legacy::compute_rotation(rseed, dimension);
                Params::FixedRotated { fopt, R: Q }
            }

            Function::Gallagher1 | Function::Gallagher2 => {
                let R = coco_legacy::compute_rotation(rseed, dimension);

                let peaks = match function {
                    Function::Gallagher1 => 101,
                    Function::Gallagher2 => 21,
                    _ => unreachable!(),
                };

                // a == arrCondition
                let mut rperm = coco_legacy::compute_unif(rseed, peaks - 1)
                    .into_iter()
                    .enumerate()
                    .collect::<Vec<_>>();
                rperm.sort_unstable_by_key(|&(_, x)| OrderedFloat(x));
                let mut a = rperm
                    .into_iter()
                    .map(|(i, _)| i as f64)
                    .map(|j| 1000f64.powf(j / ((peaks - 2) as f64)))
                    .collect::<Vec<_>>();

                match function {
                    Function::Gallagher1 => a.insert(0, f64::sqrt(1000.0)), // might be 100
                    Function::Gallagher2 => a.insert(0, 1000.0),            // might be 1000
                    _ => unreachable!(),
                }

                // w = peak_values
                let mut w = (0..peaks)
                    .into_iter()
                    .map(|i| i as f64)
                    .map(|i| 1.1 + 8.0 * (i - 1.0) / ((peaks - 2) as f64))
                    .collect::<Vec<_>>();
                w[0] = 10.0;

                // c == array_scales
                let mut c = vec![0.0; peaks * dimension];
                for i in 0..peaks {
                    let mut aperm = coco_legacy::compute_unif(rseed + (1000 * i), dimension)
                        .into_iter()
                        .enumerate()
                        .collect::<Vec<_>>();
                    aperm.sort_unstable_by_key(|&(_, x)| OrderedFloat(x));
                    let aperm = aperm.into_iter().map(|(i, _)| i as f64).collect::<Vec<_>>();

                    for j in 0..dimension {
                        c[i * dimension + j] =
                            f64::powf(a[i], aperm[j] / (dimension - 1) as f64 - 0.5)
                    }
                }

                let (pb, pc) = match function {
                    Function::Gallagher1 => (10.0, 5.0),
                    Function::Gallagher2 => (9.8, 4.9),
                    _ => unreachable!(),
                };

                // y == x_local
                let random_numbers = coco_legacy::compute_unif(rseed, dimension * peaks);
                let mut y = vec![0.0; dimension * peaks];
                for i in 0..dimension {
                    for j in 0..peaks {
                        y[i * peaks + j] = 0.0;

                        for k in 0..dimension {
                            y[i * peaks + j] +=
                                R[i][k] * (pb * random_numbers[j * dimension + k] - pc);
                        }

                        if j == 0 {
                            y[i * peaks + j] *= 0.8;
                        }
                    }
                }

                Params::Gallagher {
                    fopt,
                    peaks,
                    y,
                    w,
                    c,
                    R,
                }
            }
        }
    }
}