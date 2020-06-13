import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate


def plot_data():
    data = np.genfromtxt("./mc_sim_data.csv", skip_header=1, delimiter=",", dtype=float)

    block_sizes = np.array([16, 32, 64, 128])
    num_trials = np.array([16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 1024*1024])
    # Reshape performance to be in format (num_trial, block_size)
    perf = data[:, 2].reshape(4, 7)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i in range(len(num_trials)):
        # Make interp function
        xdata = np.linspace(array_sizes[0], array_sizes[-1], 2000)
        
        t, c, k = interpolate.splrep(array_sizes[:cutoff], thread_speedups[:cutoff, i], s=0, k=2)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        ax.plot(xdata, spline(xdata), label="{} threads".format(num_threads[i]))
        ax.scatter(array_sizes[:cutoff], thread_speedups[:cutoff, i])#, label="{} threads".form
        
        ax.plot(block_sizes, perf[:, i], label="Num Trials {}".format(num_trials[i]))
        ax.scatter(block_sizes, perf[:, i])
    ax.grid(alpha=0.5)
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Performance")
    ax.set_title("Block Size vs. Performance for various Number of Trials")
    ax.legend()
    # plt.show()
    plt.savefig("./block_size.png")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i in range(len(block_sizes)):
        ax.plot(num_trials, perf[i, :], label="Block Size {}".format(block_sizes[i]))
        ax.scatter(num_trials, perf[i, :])
    ax.grid(alpha=0.5)
    ax.set_xlabel("Num Trials")
    ax.set_ylabel("Performance")
    ax.set_title("Num Trials vs. Performance for various Block Sizes")
    ax.legend()
    # plt.show()
    plt.savefig("./num_trials.png")

def print_table():
    data = np.genfromtxt("./mc_sim_data.csv", skip_header=1, delimiter=",", dtype=float)

    block_sizes = np.array([16, 32, 64, 128])
    num_trials = np.array([16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 1024*1024])
    # Reshape performance to be in format (num_trial, block_size)
    perf = data[:, 2].reshape(4, 7)

    header = "{}".format(num_trials[0])
    for i in range(1, len(num_trials)):
        header += " & {}".format(num_trials[i])
    header += " \\\\ \\hline\n"

    outstring = ""
    # Write data
    for i in range(len(block_sizes)):
        outstring += "{}".format(int(block_sizes[i]))
        for j in range(len(num_trials)):
            outstring += " & {:.2f}".format(perf[i, j])
        outstring += " \\\\ \\hline\n"

    print(header)
    print(outstring)

# plot_data()
print_table()