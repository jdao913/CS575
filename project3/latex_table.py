import numpy as np

data = np.genfromtxt("./sim_data.csv", skip_header=1, delimiter=",", dtype=float)

print("Data table: (Threads row, Nodes column)")
month_num = np.arange(data.shape[0])
outstring = ""
# Write data
for i in range(data.shape[0]):
    outstring += str(month_num[i]) + " & "
    for j in range(2, data.shape[1]):
        value = data[i, j]
        if value.is_integer():
            value = int(value)
        outstring += str(value)
        if (j == data.shape[1]-1):
            outstring += " \\\\ \\hline\n"
        else:
            outstring += " & "
print(outstring)