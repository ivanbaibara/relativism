from src.solver import Solver
from src.asolution import ASolution
from src.utils import compare_density_plots

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----- Init common objects ------
solver = Solver()
asolultion = ASolution()

# ----- Set system configuration ------
N = 25000
charge = 1.6021892e-16
mass = 1.673e-24
r_max = 1e-14
t_max = 1e-22 * 0.5
dt = t_max * 0.01

c = 3e8
e0 = 8.852e-12
k = 1 / (4 * np.pi * e0)

sigma = 0.2
mu = 0
tau = 3

solver.set_constants(N, charge, mass, k, c, r_max, t_max, dt, sigma, tau, mu)
asolultion.set_constants(N, charge, mass, r_max, t_max, c, e0, sigma, mu, tau / r_max)

# ------ Init physical system (with lognorm distribution) ------
solver.fill_particles()
initial_perticle_r = solver.final().copy()

# ------ Run simulatiuon ------
solver.run()

# ------ Save data ------
ROOT_DIR = Path(__file__).resolve().parent.parent
data_dir = Path(ROOT_DIR).joinpath('solved')
data_dir.mkdir(parents=False, exist_ok=True)
solver.save_data(ROOT_DIR)

# ------ Display results and compare with exact solution ------
final_particle_r = solver.final()

ex_x1, ex_y1 = asolultion.get_function(100, 0)
ex_x2, ex_y2 = asolultion.get_function(100, t_max)

compare_density_plots(
    initial_perticle_r,
    final_particle_r,
    charge / N,
    xn1=ex_x1,
    yn1=ex_y1,
    xn2=ex_x2,
    yn2=ex_y2,
    x_lim=[0, r_max * 1.3],
    y_lim=[0, 1e27],
    analytical_marker='',
    analytical_color1='grey',
    analytical_color2='cyan',
    analytical_size=2
)
