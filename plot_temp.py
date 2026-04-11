import csv
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
plt.style.use('seaborn-darkgrid')
#'C:/Users/piero/Desktop/OneDrive-2021-01-03/CT-Log 2021-01-03 22-17-52.csv'
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    flag = 0
    diz = {}
    n_core = []
    for row in reader:
        if 'Session end:' in row:
            flag = -1
        elif 'Processor:' in row:
            title_proc = row

        if flag >= 1:
            test = {}
            for count, iups in enumerate(row):
                test.update({col[count]: iups})

            diz.update({flag: test})
            flag = flag+1

        if 'Time' in row:
            for name_core in row:
                if 'Temp' in name_core:
                    n_core.append(name_core)
            col = row
            flag = 1

    temp = {i: [] for i in n_core}
    temp.update({'Temp mean': []})
    dove = len(diz.keys())-1

    for k in range(dove):
        som = 0
        for num, h in enumerate(n_core):
            temp[h].append(int(diz[k+1][h]))
            som = som+int(diz[k+1][h])
        temp['Temp mean'].append(som/len(n_core))

    x = list(range(1, len(diz.keys())))

    plt.title(title_proc)
    for t in n_core:
        plt.plot(x, temp[t],linewidth=0.5)

    plt.legend(n_core)
    plt.ylim([15, 70])
    plt.xlim(left=0)
    plt.show()