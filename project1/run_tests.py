import subprocess
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.interpolate as interpolate

def run_experiments():
    num_trials = np.array([10, 100, 1000, 5000, 10000, 50000, 100000])
    num_threads = np.array([1, 2, 4, 6, 8])
    data = np.zeros((len(num_threads), len(num_trials)))
    avg_data = np.zeros((len(num_threads), len(num_trials)))

    start_t = time.time()
    for i in range(len(num_trials)):
        for j in range(len(num_threads)):
            print("Running {} trials using {} threads".format(num_trials[i], num_threads[j]))
            result = subprocess.run(["./mc_sim", str(num_threads[j]), str(num_trials[i])], capture_output=True)
            res_split = result.stdout.decode("utf-8").split("\n")
            max_perf = float(res_split[1].split("\t")[1][17:])
            avg_perf = float(res_split[1].split("\t")[2][16:])
            data[j, i] = max_perf
            avg_data[j, i] = avg_perf
            print("max_perf: {}\tavg perf: {}".format(max_perf, avg_perf))
    print("Total run time: ", time.time() - start_t)
    np.savez("run_data.npz", perf_data=data, avg_perf=avg_data, num_threads=num_threads, num_trials=num_trials)

def plot_data(filename):
    data = np.load(filename)
    perf_data = data["perf_data"]
    avg_perf = data["avg_perf"]
    num_threads = data["num_threads"]
    num_trials = data["num_trials"]

    # Num trials graph:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(num_threads)):
        # Make interp function
        xdata = np.linspace(num_trials[0], num_trials[-1], 2000)
        t, c, k = interpolate.splrep(num_trials[:], perf_data[i, :], s=15, k=2)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        ax.plot(xdata, spline(xdata), label="{} threads".format(num_threads[i]))
        ax.scatter(num_trials[:], perf_data[i, :])
    ax.legend()
    ax.set_ylabel("MegaTrials/Second")
    ax.set_xlabel("Number of Monte Carlo Trials")
    ax.set_title("Monte Carlo Performance")
    yticks = np.linspace(0, 250, 11, dtype=int)
    xticks = np.linspace(0, num_trials[-1], 11, dtype=int)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.grid(alpha=0.3)
    # plt.show()
    plt.savefig("./trials_graph.png")

    # Num thread graph:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(num_trials)):
        # Make interp function
        xdata = np.linspace(num_threads[0], num_threads[-1], 2000)
        t, c, k = interpolate.splrep(num_threads[:], perf_data[:, i], s=5, k=2)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        ax.plot(xdata, spline(xdata), label="{} trials".format(num_trials[i]))
        ax.scatter(num_threads[:], perf_data[:, i])
    ax.legend()
    ax.set_ylabel("MegaTrials/Second")
    ax.set_xlabel("Number of Threads")
    ax.set_title("Monte Carlo Performance")
    yticks = np.linspace(0, 250, 11, dtype=int)
    ax.set_yticks(yticks)
    ax.grid(alpha=0.3)
    # plt.show()
    plt.savefig("./threads_graph.png")

# run_experiments()
plot_data("./run_data.npz")