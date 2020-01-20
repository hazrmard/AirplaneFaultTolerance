"""
Unit tests for code. Run using:

```
python test.py
```
"""



from unittest import TestCase, main

import numpy as np
from pystatespace import Trapezoid

from tanks import TanksFactory



class TestTanks(TestCase):


    def setUp(self):
        self.tanks = TanksFactory()
        self.n = len(self.tanks.pumps)
        self.e = len(self.tanks.engines)
        self.system = Trapezoid(dims=self.n, outputs=self.n, dx=self.tanks.dxdt,
                                out=self.tanks.y, tstep=1e-1)


    def test_conservation_open_valves(self):
        t = np.linspace(0, 10, num=100, endpoint=False)
        u = np.zeros((len(t), self.n))
        x, y = self.system.predict(t, np.ones(self.n), 0)
        total_level = np.sum(x, axis=1)