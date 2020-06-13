import numpy as np
import matplotlib.pyplot as plt
import subprocess

def plot_perf():
    # 1-thread test
    result = subprocess.run(["./sum_mp", str(1)], capture_output=True)
    result = result.stdout.decode("utf-8")
    mp1_perf = float(result.split("\t")[1].split(" ")[1])

    # 8-thread test
    result = subprocess.run(["./sum_mp", str(8)], capture_output=True)
    result = result.stdout.decode("utf-8")
    mp8_perf = float(result.split("\t")[1].split(" ")[1])

    # SIMD test
    result = subprocess.run(["./sum_simd"], capture_output=True)
    result = result.stdout.decode("utf-8")
    simd_perf = float(result.split(" ")[1])

    # OpenCL test
    result = subprocess.run(["./sum_cl", str(64)], capture_output=True)
    result = result.stdout.decode("utf-8")
    cl_perf = float(result.split("\t")[1].split(" ")[1])

    x = np.arange(4)
    data = np.array([mp1_perf, mp8_perf, simd_perf, cl_perf])
    labels = ["1-thread OpenMP", "8-thread OpenMP", "SIMD", "OpenCL"]
    width = 0.5

    fig, ax = plt.subplots()
    rects = ax.bar(x, data, width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale('log')
    ax.set_yticks([10, 100, 1e3, 1e4, 1e4, 1e5])
    ax.set_ylabel("Performance (MegaMultsPerSecond)")
    ax.set_title("Autocorrelation Performance Comparison")
    # plt.show()
    plt.savefig("bar_plot.png")



def plot_autocorr():
    data = np.genfromtxt("./auto_corr.csv", max_rows=512)
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(x[1:], data[1:])
    xticks = np.arange(525, step=25)
    ax.set_xticks(xticks)
    # plt.show()
    plt.savefig("autocorr.png")

# plot_autocorr()
plot_perf()