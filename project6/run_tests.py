import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import scipy.interpolate as interpolate

arr_sizes = np.array([1024, 16*1024, 64*1024, 100*1024, 250*1024, 500*1024,  
 					1024*1024, 2*1024*1024, 4*1024*1024, 6*1024*1024, 8*1024*1024])
local_sizes = np.array([8, 16, 64, 128, 256, 512])
reduce_local_sizes = np.array([32, 64, 128, 256])
num_trials = 10

def run_arr_mult():
    perf_data = np.zeros((len(arr_sizes), len(local_sizes)))
    for i in range(len(arr_sizes)):
        for j in range(len(local_sizes)):
            avg_perf = 0
            max_perf = 0
            for k in range(num_trials):
                output = subprocess.run(["./arr_mult", str(arr_sizes[i]), str(local_sizes[j])], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
                # print(output)
                perf = float(output.split()[3])
                avg_perf += perf
                if perf > max_perf:
                    max_perf = perf
            perf_data[i][j] = max_perf
            avg_perf /= num_trials
            print("Num elements: {}\tLocal size: {}\t Avg perf: {}\t Max Perf: {}".format(arr_sizes[i], local_sizes[j], avg_perf, max_perf))

    np.save("./arr_mult_data.npy", perf_data)

def run_arr_mult_add():
    perf_data = np.zeros((len(arr_sizes), len(local_sizes)))
    for i in range(len(arr_sizes)):
        for j in range(len(local_sizes)):
            avg_perf = 0
            max_perf = 0
            for k in range(num_trials):
                output = subprocess.run(["./arr_mult_add", str(arr_sizes[i]), str(local_sizes[j])], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
                # print(output)
                perf = float(output.split()[3])
                avg_perf += perf
                if perf > max_perf:
                    max_perf = perf
            perf_data[i][j] = max_perf
            avg_perf /= num_trials
            print("Num elements: {}\tLocal size: {}\t Avg perf: {}\t Max Perf: {}".format(arr_sizes[i], local_sizes[j], avg_perf, max_perf))
    np.save("./arr_mult_add_data.npy", perf_data)

def run_arr_mult_reduce():
    perf_data = np.zeros((len(arr_sizes), len(reduce_local_sizes)))
    for i in range(len(arr_sizes)):
        for j in range(len(reduce_local_sizes)):
            avg_perf = 0
            max_perf = 0
            for k in range(num_trials):
                output = subprocess.run(["./arr_mult_reduce", str(arr_sizes[i]), str(reduce_local_sizes[j])], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
                perf = float(output.split()[3])
                avg_perf += perf
                if perf > max_perf:
                    max_perf = perf
            perf_data[i][j] = max_perf
            avg_perf /= num_trials
            print("Num elements: {}\tLocal size: {}\t Avg perf: {}\t Max Perf: {}".format(arr_sizes[i], reduce_local_sizes[j], avg_perf, max_perf))
    np.save("./arr_mult_reduce_data.npy", perf_data)

def plot_data():
    mult_data = np.load("./arr_mult_data.npy")
    mult_add_data = np.load("./arr_mult_add_data.npy")
    mult_reduce_data = np.load("./arr_mult_reduce_data.npy")
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    for i in range(len(local_sizes)):
        # Make interp function
        xdata = np.linspace(arr_sizes[0], arr_sizes[-1], 2000)
        t, c, k = interpolate.splrep(arr_sizes, mult_data[:, i], s=0, k=2)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        ax[0].plot(xdata, spline(xdata), label="Local Size: {}".format(local_sizes[i]))
        ax[0].scatter(arr_sizes, mult_data[:, i])

        # Make interp function
        xdata = np.linspace(arr_sizes[0], arr_sizes[-1], 2000)
        t, c, k = interpolate.splrep(arr_sizes, mult_add_data[:, i], s=0, k=2)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        ax[1].plot(xdata, spline(xdata), label="Local Size: {}".format(local_sizes[i]))
        ax[1].scatter(arr_sizes, mult_add_data[:, i])
    ax[0].set_ylabel("Performace (GigaMultsPerSecond)")
    ax[1].set_ylabel("Performace (GigaMultAddsPerSecond)")
    for i in range(2):
        ax[i].legend(loc="upper left")
        # ax[i].set_ylabel("Performace (GigaMultsPerSecond")
        ax[i].set_xlabel("Global Dataset Size")
        ax[i].grid()
        # ax[i].set_ylim(0, 6)
    ax[0].set_title("Array Multiply Performance for Varying Work Sizes")
    ax[1].set_title("Array Multiple Add Performance for Varying Work Sizes")
    # plt.show()
    plt.tight_layout()
    plt.savefig("./mult_multadd_localwork.png")

    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    for i in range(len(arr_sizes)):
        # Make interp function
        xdata = np.linspace(local_sizes[0], local_sizes[-1], 2000)
        t, c, k = interpolate.splrep(local_sizes, mult_data[i, :], s=0, k=2)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        ax[0].plot(xdata, spline(xdata), label="Data Size: {}".format(arr_sizes[i]))
        ax[0].scatter(local_sizes, mult_data[i, :])

        # Make interp function
        xdata = np.linspace(local_sizes[0], local_sizes[-1], 2000)
        t, c, k = interpolate.splrep(local_sizes, mult_add_data[i, :], s=0, k=2)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        ax[1].plot(xdata, spline(xdata), label="Data Size: {}".format(arr_sizes[i]))
        ax[1].scatter(local_sizes, mult_add_data[i, :])
    ax[0].set_ylabel("Performace (GigaMultsPerSecond)")
    ax[1].set_ylabel("Performace (GigaMultAddsPerSecond)")
    for i in range(2):
        ax[i].legend(loc="upper left")
        ax[i].set_xlabel("Global Dataset Size")
        ax[i].grid()
        # ax[i].set_ylim(0, 6)
    ax[0].set_title("Array Multiply Performance for Varying Global Dataset Sizes")
    ax[1].set_title("Array Multiple Add Performance for Varying Global Dataset Sizes")
    # plt.show()
    plt.tight_layout()
    plt.savefig("./mult_multadd_dataset.png")

    # Mult reduce graph
    fig, ax = plt.subplots(figsize=(11, 5))
    for i in range(len(reduce_local_sizes)):
        # Make interp function
        xdata = np.linspace(arr_sizes[0], arr_sizes[-1], 2000)
        t, c, k = interpolate.splrep(arr_sizes, mult_reduce_data[:, i], s=0, k=2)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        ax.plot(xdata, spline(xdata), label="Local Size: {}".format(reduce_local_sizes[i]))
        ax.scatter(arr_sizes, mult_reduce_data[:, i])

    ax.set_ylabel("Performace (GigaMult-ReducsPerSecond)")
    ax.legend(loc="upper left")
    ax.set_xlabel("Global Dataset Size")
    ax.grid()
    ax.set_title("Array Multiply Reduction Performance for Varying Work Sizes")
    # plt.show()
    plt.tight_layout()
    plt.savefig("./mult_multreduce_localwork.png")

def print_tables():
    mult_data = np.load("./arr_mult_data.npy")
    mult_add_data = np.load("./arr_mult_add_data.npy")
    mult_reduce_data = np.load("./arr_mult_reduce_data.npy")

    # Mult table
    print("Multiple array table")
    outstring = " "
    # Write header
    for i in range(len(local_sizes)):
        outstring += "& {}".format(local_sizes[i])
    outstring += "\\\\ \\hline\n"
    # Write data
    for i in range(len(arr_sizes)):
        outstring += "{} ".format(int(arr_sizes[i]))
        for j in range(len(local_sizes)):
            outstring += "& {:.2f} ".format(mult_data[i, j])
        outstring += "\\\\ \\hline\n"
    print(outstring)

    # Mult add table
    print("Multiple add array table")
    outstring = " "
    # Write header
    for i in range(len(local_sizes)):
        outstring += "& {}".format(local_sizes[i])
    outstring += "\\\\ \\hline\n"
    # Write data
    for i in range(len(arr_sizes)):
        outstring += "{} ".format(int(arr_sizes[i]))
        for j in range(len(local_sizes)):
            outstring += "& {:.2f} ".format(mult_add_data[i, j])
        outstring += "\\\\ \\hline\n"
    print(outstring)

    # Mult reduce table
    print("Multiple reduce table")
    outstring = " "
    # Write header
    for i in range(len(reduce_local_sizes)):
        outstring += "& {}".format(reduce_local_sizes[i])
    outstring += "\\\\ \\hline\n"
    # Write data
    for i in range(len(arr_sizes)):
        outstring += "{} ".format(int(arr_sizes[i]))
        for j in range(len(reduce_local_sizes)):
            outstring += "& {:.2f} ".format(mult_reduce_data[i, j])
        outstring += "\\\\ \\hline\n"
    print(outstring)

# run_arr_mult()
# run_arr_mult_add()
# run_arr_mult_reduce()
# plot_data()
print_tables()
