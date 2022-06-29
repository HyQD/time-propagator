import numpy as np
import matplotlib.pyplot as plt

from time_propagator0 import TimePropagator

inputs = {
    "molecule": "h 0 0 0; h 0 0 1.4",
    "final_time": 100,
    "time_step": 0.1,
    "sample_dipole_moment": True,
    "sample_kinetic_momentum": True,
    "pulses": ["p1"],
    "p1": {
        "laser_class": "sin_sin2",
        "amplitude": 0.01,
        "omega": 0.3,
        "ncycles": 3,
        "polarization": [0, 0, 1],
        "k_direction": [1, 0, 0],
    },
}


# function can take t (time), y (state vector), tdcc object and pulses object
def trace_rho(tp):
    rho = tp.tdcc.compute_one_body_density_matrix(tp.r.t, tp.r.y)
    return np.trace(rho)


tp = TimePropagator("rcis", inputs=inputs)
tp.build()
tp.add_custom_sampling_operator("trace_rho", trace_rho, dim=(1,))
output = tp.propagate()

import matplotlib.pyplot as plt

s = output["samples"]
a = s["trace_rho"]
t = s["time_points"]

plt.plot(t, a.real)
plt.axis([-1, 101, 0, 5])
plt.show()
