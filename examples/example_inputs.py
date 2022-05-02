inputs =    {   'gauge':'velocity',
                'final_time':20.0,
                'basis':'aug-cc-pvdz',

                'integrator_params':{'s':3,'eps':1e-10},

                'pulses':['pulse1','pulse2'],

                'pulse1':{  'pulse_class':'square_velocity_dipole',
                            'field_strength':0.01,
                            'omega':1.0,
                            'polarization':[0.0,0.0,1.0],
                            'k_direction':[1.0,0.0,0.0],
                            'ncycles':2,
                        },
                'pulse2':{  'pulse_class':'square_velocity_dipole',
                            'field_strength':0.01,
                            'omega':2.0,
                            'polarization':[0.0,0.0,1.0],
                            'k_direction':[1.0,0.0,0.0],
                            'ncycles':2,
                            't0':10,
                        },
            }