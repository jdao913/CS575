import subprocess
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.interpolate as interpolate
from prettytable import PrettyTable
import time

def run_experiments():
    num_trials = 5
    num_threads = np.array([1, 2, 4, 6, 8])
    num_nodes = np.array([16, 64, 256, 512, 1024, 2048, 3072, 4096, 5120])
    data = np.zeros((len(num_threads), len(num_nodes)))
    avg_data = np.zeros((len(num_threads), len(num_nodes)))

    start_t = time.time()
    for i in range(len(num_nodes)):
        for j in range(len(num_threads)):
            time.sleep(2)
            print("Running {} trials using {} threads for {} nodes".format(num_trials, num_threads[j], num_nodes[i]))
            result = subprocess.run(["./integrate", str(num_threads[j]), str(num_nodes[i]), str(num_trials)], capture_output=True)
            res_split = result.stdout.decode("utf-8").split("\n")
            max_perf = float(res_split[1].split("\t")[1][17:])
            avg_perf = float(res_split[1].split("\t")[2][16:])
            data[j, i] = max_perf
            avg_data[j, i] = avg_perf
            print("max_perf: {}\tavg perf: {}".format(max_perf, avg_perf))
    print("Total run time: ", time.time() - start_t)
    np.savez("run_data_even8.npz", perf_data=data, avg_perf=avg_data, num_threads=num_threads, num_nodes=num_nodes)

def plot_data(filename):
    data = np.load(filename)
    perf_data = data["perf_data"]
    avg_perf = data["avg_perf"]
    num_threads = data["num_threads"]
    num_nodes = data["num_nodes"]

    # Num nodes graph:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(num_threads)):
        # Make interp function
        xdata = np.linspace(num_nodes[0], num_nodes[-1], 2000)
        
        t, c, k = interpolate.splrep(num_nodes[:], perf_data[i, :], s=10, k=3)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        ax.plot(xdata, spline(xdata), label="{} threads".format(num_threads[i]))
        ax.scatter(num_nodes[:], perf_data[i, :])
    ax.legend()
    ax.set_ylabel("MegaVolumes/Second")
    ax.set_xlabel("Number of Nodes Used")
    ax.set_title("Integration Performance vs. Number of Nodes")
    yticks = np.linspace(0, 35, 8, dtype=int)
    xticks = np.linspace(0, 5500, 12, dtype=int)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.grid(alpha=0.3)
    # plt.show()
    plt.savefig("./nodes_graph.png")

    # Num thread graph:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(num_nodes)):
        # Make interp function
        xdata = np.linspace(num_threads[0], num_threads[-1], 2000)
        t, c, k = interpolate.splrep(num_threads[:], perf_data[:, i], s=0, k=2)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        ax.plot(xdata, spline(xdata), label="{} nodes".format(num_nodes[i]))
        ax.scatter(num_threads[:], perf_data[:, i])
    ax.legend()
    ax.set_ylabel("MegaVolumes/Second")
    ax.set_xlabel("Number of Threads")
    ax.set_title("Integration Performance vs. Number of Nodes")
    ax.grid(alpha=0.3)
    # plt.show()
    plt.savefig("./threads_graph.png")

def print_table(filename):
    data = np.load(filename)
    perf_data = np.around(data["perf_data"], decimals=2)
    avg_perf = np.around(data["avg_perf"], decimals=2)
    num_threads = data["num_threads"]
    num_nodes = data["num_nodes"]

    header = [str(num_nodes[i]) + " Nodes" for i in range(len(num_nodes))]
    header = [''] + header
    t = PrettyTable(header)
    for i in range(len(num_threads)):
        data = perf_data[i, :].tolist()
        data = [str(perf_data[i, j]) + " (" + str(avg_perf[i, j]) + ")" for j in range(len(num_nodes))]
        data = ["{} Thread(s)".format(num_threads[i])] + data
        t.add_row(data)
    print("Average Performance in ( )")
    print(t)

def print_table_latex(filename):
    data = np.load(filename)
    perf_data = np.around(data["perf_data"], decimals=2)
    avg_perf = np.around(data["avg_perf"], decimals=2)
    num_threads = data["num_threads"]
    num_nodes = data["num_nodes"]

    print("Data table: (Threads row, Nodes column)")
    outstring = "& "
    # Make header
    for j in range(len(num_nodes)):
        outstring += "{}".format(num_nodes[j])
        if (j == len(num_nodes)-1):
            outstring += " \\\\\n"
        else:
            outstring += " & "
    # Write data
    for i in range(perf_data.shape[0]):
        outstring += "{} & ".format(num_threads[i])
        for j in range(perf_data.shape[1]):
            outstring += str(perf_data[i, j])
            if (j == perf_data.shape[1]-1):
                outstring += " \\\\\n"
            else:
                outstring += " & "
    print(outstring)

run_experiments()
print_table("./run_data_even.npz")
print_table_latex("./run_data_even.npz")
plot_data("./run_data_even.npz")