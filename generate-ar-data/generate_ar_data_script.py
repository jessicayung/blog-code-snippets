"""
Code to generate autoregressive data.
Replicates plots in blog post linked below.

Blog post: http://www.jessicayung.com/generating-autoregressive-data-for-experiments=

Author: Jessiac Yung
Sept 2018
"""

from generate_data import ARData, fixed_ar_coefficients
import matplotlib.pyplot as plt
import numpy as np

##################################
# Generate AR(4) with unstable poles
##################################

unstable = False
# Generate coefficients until we have at least one unstable pole
while not unstable:
    unstable_coeffs = np.random.random(5)  # 5 = 4 prev terms + 1
    # Calculate pole magnitudes
    root_magnitudes = np.abs(np.roots(unstable_coeffs))
    # check if max pole magnitude > 1 (unstable)
    if np.max(root_magnitudes) > 1:
        unstable = True
        print("Poles: {}".format(np.roots(unstable_coeffs)))

# plot unstable AR data
unstable_ar = ARData(num_datapoints=50, coeffs=unstable_coeffs, num_prev=len(unstable_coeffs), noise_var=0)

plt.plot(unstable_ar.y[:10])
plt.xlabel('t')
plt.ylabel('x_t')
plt.title("Unstable AR data (first 10 dp)")
# plt.savefig('unstable_ar_first10.jpg')
plt.show()

plt.plot(unstable_ar.y)
plt.xlabel('t')
plt.ylabel('x_t')
plt.title("Unstable AR data")
# plt.savefig('unstable_ar.jpg')
plt.show()


##################################
# Generate AR(5) with stable poles
##################################

# Fix coefficients used so can compare plots with and without noise.
c = fixed_ar_coefficients

# Generate AR(5) with stable poles, no noise
stable_ar = ARData(num_datapoints=50, coeffs=c[5], num_prev=5, noise_var=0)

plt.plot(stable_ar.y)
plt.xlabel('t')
plt.ylabel('x_t')
plt.title("Stable AR data (no noise)")
# plt.savefig('stable_ar.jpg')
plt.show()

# Generate AR(5) with stable poles, Gaussian noise
stable_ar = ARData(num_datapoints=50, coeffs=c[5], num_prev=5, noise_var=1)

plt.plot(stable_ar.y)
plt.xlabel('t')
plt.ylabel('x_t')
plt.title("Stable AR data (noise var = 1)")
# plt.savefig('stable_ar_noisy.jpg')
plt.show()