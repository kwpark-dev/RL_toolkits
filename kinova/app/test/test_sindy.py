#!/usr/bin/env python3

import numpy as np
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary


#t = np.linspace(0, 1, 100)
dt = 0.01
#x = 3 * np.exp(-2 * t)
#y = 0.5 * np.exp(t)
x = np.random.rand(20, 7)

lib = PolynomialLibrary(degree=1)
model = SINDy(feature_library=lib)
model.fit(x, t=dt)

model.print()

init = np.zeros(7)
t = np.linspace(0, 1, 10)
sim = model.simulate(init, t=t)

