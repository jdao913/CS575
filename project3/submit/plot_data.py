import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("./sim_data.csv", skip_header=1, delimiter=",", dtype=float)
# print(data)

num_months = data.shape[0]
# Convert data to Celcius and cm
data[:, 2] = (data[:, 2] - 32) * (5/9)
data[:, 3:5] *= 2.54

x = np.arange(num_months)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, data[:, 2], label="Temp ($^\circ$C)", color="tab:red")
ax.plot(x, data[:, 3], label="Precip (cm)", color="deepskyblue")
ax.plot(x, data[:, 4], label="Height (cm)", color="forestgreen", lw=3)
ax.plot(x, data[:, 5], label="NumDeer", color="saddlebrown", lw=3)
ax.plot(x, data[:, 6], label="NumMice", color="gray", lw=3)
yticks = np.linspace(0, 50, 11, dtype=int)
xticks = np.linspace(0, 75, 16, dtype=int)
ax.set_yticks(yticks)
ax.set_xticks(xticks)
ax.grid(alpha=0.5)
ax.set_xlabel("Month Number")
ax.set_title("Simulation Values Over Time")
plt.legend()
# plt.show()
plt.savefig("./sim_graph.png")

