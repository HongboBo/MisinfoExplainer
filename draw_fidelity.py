import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as ticker

plt.style.use('bmh')
matplotlib.rcParams.update({'font.size': 17})

#F+:
# x = np.array([1, 2, 3, 4, 5, 6, 7])
#
# Perturbation = np.array([0.051, 0.089, 0.128, 0.155, 0.218, 0.202, 0.227])
# Gradient = np.array([0.051, 0.065, 0.114, 0.124, 0.145, 0.161, 0.171])
# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.set_ylim(0, 0.08)
#
# ax1.plot(x, Perturbation/x, label='Perturbation', marker='o')
# ax1.plot(x, Gradient/x, label='Gradient', marker='o')
#
#
# ax1.set_ylabel('Fidelity+')
# ax1.set_xlabel('N')
# ax1.legend(loc=3, fontsize=15)
# ax1.set_title('Fidelity+', fontsize=17)
#
# # ax1.fill_between(x, F1, F2, facecolor='blue', alpha=0.3)
# plt.tight_layout()
# plt.savefig('fidelity+.pdf', bbox_inches='tight')
# plt.show()

#F-:
x = np.array([12, 13, 14, 15, 16, 17, 18])

Perturbation = np.array([0.191, 0.225, 0.104, 0.073, 0.069, 0.061, 0.058])
Gradient = np.array([1.760, 1.720, 1.770, 1.593, 1.594, 1.186, 1.202])
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.set_ylim(0, 0.2)

ax1.plot(x, Perturbation/x, label='Perturbation', marker='o')
ax1.plot(x, Gradient/x, label='Gradient', marker='o')


ax1.set_ylabel('Fidelity-')
ax1.set_xlabel('N')
ax1.legend(loc=1, fontsize=15)
ax1.set_title('Fidelity-', fontsize=17)

# ax1.fill_between(x, F1, F2, facecolor='blue', alpha=0.3)
plt.tight_layout()
plt.savefig('fidelity-.pdf', bbox_inches='tight')
plt.show()