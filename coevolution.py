import logging
import threading
import time
import math

import GP_regression as GPR

f1 = lambda x: (1/x) + math.sin(x)
f2 = lambda x: (2*x) + (x**2) + 3
f = lambda x: f1(x) if x > 0 else f2(x)

GPR.run(f1)

# two individual threads that both look to a solution
# then we evaluate here