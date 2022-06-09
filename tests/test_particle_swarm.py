"""Tests for the Dual Annealing algorithm."""

import numpy as np
from numpy.testing import assert_allclose
from pyswarms.single.global_best import GlobalBestPSO

import lmfit


def eggholder(x):
    return (-(x[1] + 47.0) * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
            - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0)))))

def create_pyswarms_particle_function_wrapper(func: callable):
    def function_accepting_particles_position(x_mat, **kwargs):
        f_out = np.empty(x_mat.shape[0])
        for i in range(x_mat.shape[0]):
            f_out[i] = func(x_mat[i,:], **kwargs)
        return f_out
    return function_accepting_particles_position

def eggholder_lmfit(params):
    x0 = params['x0'].value
    x1 = params['x1'].value

    return (-(x1 + 47.0) * np.sin(np.sqrt(abs(x0/2.0 + (x1 + 47.0))))
            - x0 * np.sin(np.sqrt(abs(x0 - (x1 + 47.0)))))


def test_pso_pyswarms_vs_lmfit():
    """Test Particle Swarm algorithm from pyswarms wrapped in lmfit versus in pyswarms."""

    # hyperparameters
    iters = 1000
    n_particles = 32
    options = {'c1': 0.5, 'c2': 0.3, 'w': 1e-3}

    # pyswarms
    np.random.seed(42)

    x_max = np.array([512, 512])
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=2, options=options, bounds=bounds)

    eggholder_pyswarms = create_pyswarms_particle_function_wrapper(eggholder)

    cost_pyswarms, pos_pyswarms = optimizer.optimize(eggholder_pyswarms, iters)

    # lmfit with pyswarms
    np.random.seed(42)

    opt_args = dict(
        iters=iters,
    )

    pars = lmfit.Parameters()
    pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
    mini = lmfit.Minimizer(eggholder_lmfit, pars, options=options)
    result = mini.minimize(method='particle_swarm', n_particles=n_particles, opt_args=opt_args)
    out_x = np.array([result.params['x0'].value, result.params['x1'].value])

    assert_allclose(cost_pyswarms, result.residual)
    assert_allclose(pos_pyswarms, out_x)


# correct result for Alpine02 function
global_optimum = [7.91705268, 4.81584232]
fglob = -6.12950


def test_da_Alpine02(minimizer_Alpine02):
    """Test particle_swarm algorithm on Alpine02 function."""
    out = minimizer_Alpine02.minimize(method='particle_swarm')
    out_x = np.array([out.params['x0'].value, out.params['x1'].value])

    assert_allclose(out.residual, fglob, rtol=1e-5)
    assert_allclose(min(out_x), min(global_optimum), rtol=1e-3)
    assert_allclose(max(out_x), max(global_optimum), rtol=1e-3)
    assert out.method == 'particle_swarm'


def test_da_bounds(minimizer_Alpine02):
    """Test particle_swarm algorithm with bounds."""
    pars_bounds = lmfit.Parameters()
    pars_bounds.add_many(('x0', 1., True, 5.0, 15.0),
                         ('x1', 1., True, 2.5, 7.5))

    out = minimizer_Alpine02.minimize(params=pars_bounds,
                                      method='particle_swarm')
    assert 5.0 <= out.params['x0'].value <= 15.0
    assert 2.5 <= out.params['x1'].value <= 7.5
