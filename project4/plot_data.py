import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

def plot_simd_data():
    data = np.genfromtxt("./simd_data_best.csv", skip_header=1, delimiter=",", dtype=float)

    array_sizes = data[:, 0]
    SSE_perf = data[:, 1]
    loop_perf = data[:, 3]

    speedup = np.divide(SSE_perf, loop_perf)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(array_sizes, speedup)
    ax.scatter(array_sizes, speedup)
    yticks = np.linspace(2.4, 2.6, 11, dtype=float)
    xticks = np.linspace(0, 1e7, 11, dtype=int)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.grid(alpha=0.5)
    ax.set_xlabel("Array Size")
    ax.set_ylabel("SSE Speedup")
    ax.set_title("Array Size vs. SSE Speedup")
    # plt.show()
    plt.savefig("./simd_data.png")

def simd_table():
    data = np.genfromtxt("./simd_data_best.csv", skip_header=1, delimiter=",", dtype=float)

    array_sizes = data[:, 0]
    SSE_perf = data[:, 1]
    loop_perf = data[:, 3]
    speedup = np.divide(SSE_perf, loop_perf)

    outstring = ""
    # Write data
    for i in range(len(array_sizes)):
        outstring += "{} & ".format(int(array_sizes[i]))
        outstring += "{:.2f} & {:.2f} & {:.2f} \\\\ \\hline\n".format(SSE_perf[i], loop_perf[i], speedup[i])
        # for j in range(perf_data.shape[1]):
        #     outstring += str(perf_data[i, j])
        #     if (j == perf_data.shape[1]-1):
        #         outstring += " \\\\\n"
        #     else:
        #         outstring += " & "
    print(outstring)

def plot_simd_multi():
    data = np.genfromtxt("./simd_multi1_data.csv", skip_header=1, delimiter=",", dtype=float)

    array_sizes = data[:, 0]
    SSE_perf = np.zeros((len(array_sizes), 5))
    loop_perf = np.zeros((len(array_sizes), 5))
    num_threads = [1, 2, 4]

    for i in range(len(num_threads)):
        data = np.genfromtxt("./simd_multi{}_data.csv".format(num_threads[i]), skip_header=1, delimiter=",", dtype=float)
        SSE_perf[:, i] = data[:, 1]
        loop_perf[:, i] = data[:, 3]

    baseline_perf = loop_perf[:, 0]
    SIMD_speedups = np.divide(SSE_perf[:, 0], baseline_perf)
    thread_speedups = loop_perf / baseline_perf[:, None]
    thread_SIMD_speedups = SSE_perf[:, 1:] / baseline_perf[:, None]
    cutoff = None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(array_sizes[:cutoff], SIMD_speedups[:cutoff], label="SIMD Only")
    ax.scatter(array_sizes[:cutoff], SIMD_speedups[:cutoff])
    for i in range(len(num_threads)):
        # Make interp function
        xdata = np.linspace(array_sizes[0], array_sizes[-1], 2000)
        
        t, c, k = interpolate.splrep(array_sizes[:cutoff], thread_speedups[:cutoff, i], s=0, k=2)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        ax.plot(xdata, spline(xdata), label="{} threads".format(num_threads[i]))
        ax.scatter(array_sizes[:cutoff], thread_speedups[:cutoff, i])#, label="{} threads".format(num_threads[i]))
    for i in range(len(num_threads)-1):
        # Make interp function
        xdata = np.linspace(array_sizes[0], array_sizes[-1], 2000)
        
        t, c, k = interpolate.splrep(array_sizes[:cutoff], thread_SIMD_speedups[:cutoff, i], s=0, k=2)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        ax.plot(xdata, spline(xdata), label="{} threads + SIMD".format(num_threads[i+1]))
        ax.scatter(array_sizes[:cutoff], thread_SIMD_speedups[:cutoff, i])#, label="{} threads + SIMD".format(num_threads[i+1]))
    ax.legend()
    yticks = np.linspace(0, 12, 13, dtype=float)
    xticks = np.linspace(0, 1e7, 11, dtype=int)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.grid(alpha=0.5)
    ax.set_xlabel("Array Size")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup for Varying Number of Threads with and without SIMD vs. Array Size")
    # plt.show() 
    plt.savefig("./simd_multi.png")

def simd_multi_table():
    data = np.genfromtxt("./simd_multi1_data.csv", skip_header=1, delimiter=",", dtype=float)

    array_sizes = data[:, 0]
    SSE_perf = np.zeros((len(array_sizes), 5))
    loop_perf = np.zeros((len(array_sizes), 5))
    num_threads = [1, 2, 4]

    for i in range(len(num_threads)):
        data = np.genfromtxt("./simd_multi{}_data.csv".format(num_threads[i]), skip_header=1, delimiter=",", dtype=float)
        SSE_perf[:, i] = data[:, 1]
        loop_perf[:, i] = data[:, 3]

    baseline_perf = loop_perf[:, 0]
    SIMD_speedups = np.divide(SSE_perf[:, 0], baseline_perf)
    thread_speedups = loop_perf / baseline_perf[:, None]
    thread_SIMD_speedups = SSE_perf[:, 1:] / baseline_perf[:, None]

    print("SIMD performance table")
    outstring = ""
    # Write data
    for i in range(len(array_sizes)):
        outstring += "{} & ".format(int(array_sizes[i]))
        outstring += "{:.2f} & {:.2f} & {:.2f} \\\\ \\hline\n".format(SSE_perf[i, 0], SSE_perf[i, 1], SSE_perf[i, 2])

    print(outstring)

    print("Loop performance table")
    outstring = ""
    # Write data
    for i in range(len(array_sizes)):
        outstring += "{} & ".format(int(array_sizes[i]))
        outstring += "{:.2f} & {:.2f} & {:.2f} \\\\ \\hline\n".format(loop_perf[i, 0], loop_perf[i, 1], loop_perf[i, 2])

    print(outstring)

    print("Speedup table")
    outstring = ""
    # Write data
    for i in range(len(array_sizes)):
        outstring += "{} & ".format(int(array_sizes[i]))
        outstring += "{:.2f} & {:.2f} \\\\ \\hline\n".format(thread_SIMD_speedups[i, 0], thread_SIMD_speedups[i, 1])

    print(outstring)

# simd_table()
# simd_multi_table()
# plot_simd_data()
plot_simd_multi()
