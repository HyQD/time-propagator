import numpy as np
import matplotlib.pyplot as plt

from time_propagator0 import TimePropagator


#inputs from dict
if False:
    inputs =    {   'gauge':'length',
                    'final_time':20.0,
                    'basis':'aug-cc-pvdz',

                    'pulses':['pulse1'],

                    'pulse1':{  'pulse_class':'square_length_dipole',
                                'field_strength':0.01,
                                'omega':1.0,
                                'polarization':[0.0,0.0,1.0],
                                'ncycles':2,
                            },
                }

    tp = TimePropagator('rcis',**inputs)
    tp.setup_quantum_system(program='pyscf',molecule='lih.xyz',charge=0)
    tp.setup_ground_state()
    tp.build()
    output = tp.propagate()


#from input file
if False:
    tp = TimePropagator('rhf',inputs='example_inputs.py')
    tp.setup_quantum_system(program='pyscf',molecule='lih.xyz',charge=0)
    tp.setup_ground_state()
    tp.build()
    output = tp.propagate()



if True:
    tp = TimePropagator('rhf',inputs='example_inputs.py')
    tp.setup_quantum_system(program='pyscf',molecule='He:0 0 0',charge=0)
    tp.setup_ground_state()
    tp.build()
    output = tp.propagate()



#from file, plane_wave
if False:
    tp = TimePropagator('rhf',inputs='example_inputs.py',laser_approx='plane_wave')
    tp.setup_quantum_system(program='dalton',molecule='lih.xyz',charge=0)
    tp.setup_ground_state()
    tp.build()
    output = tp.propagate()



if True:
    t = output['time_points']
    d = output['dipole_moment']
    plt.plot(t,d.real)
    plt.show()

