import numpy as np
from numpy.testing import assert_allclose
import sys
from scipy.spatial.distance import cosine

a = np.load(sys.argv[1]).astype(np.float32)
b = np.load(sys.argv[2]).astype(np.float32)

np.seterr(over="raise")
cos_sim = 1 - cosine(a.flatten(), b.flatten())
print(f"cosin similarity is {cos_sim}")

assert_allclose(a, b)