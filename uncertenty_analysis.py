import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import shutil
import os


class Th:
    def __init__(self, path_js, fname):
        self.path = path_js
        self.fname = fname
        self.list_v = []
        self.dizio = ''
        self.vfin, self.thfin = -1, -1
        self.newth = 0
        self.list_ale, self.list_epi, self.list_tot = [], [], []
        self.tot_n = 0

    def openf(self):
        with open(self.path, 'r') as myfile:
            openf = myfile.read()
        self.dizio = json.loads(openf)
    
    def create_list(self):
        self.openf()
        print('open the file')
        for i in self.dizio:
            self.list_ale.append(float(self.dizio[i]['ale']))
            self.list_epi.append(float(self.dizio[i]['epi']))
            self.dizio[i]['Unc_tot'] = float(self.dizio[i]['ale']) + float(self.dizio[i]['epi'])
            self.list_tot.append(self.dizio[i]['Unc_tot'])
        
        self.tot_n = len(self.list_tot)

        return self.list_ale, self.list_epi, self.list_tot

    def otsu(self):
        pixel_number = self.tot_n
        mean_weigth = 1.0/pixel_number
        his, bi = np.histogram(self.list_tot, np.arange(0, 1, 0.01))

        for o, t in enumerate(bi[1:-1], 1):
            pcb = np.sum(his[:o])
            pcf = np.sum(his[o:])
            Wb = pcb * mean_weigth
            Wf = pcf * mean_weigth

            a = np.asarray(bi[1:o+1])
            a1 = np.asarray(bi[o+1:])
            b = np.asarray(his[:o])
            b1 = np.asarray(his[o:])

            s1 = np.sum(np.dot(a, b))
            if s1 == 0 and pcb == 0:
                mub = 0
            else:
                mub = s1/float(pcb)
            s = np.sum(np.dot(a1, b1))
            if s == 0 and pcf == 0:
                muf = 0
            else:
                muf = s/float(pcf)
            value = Wb * Wf * (mub - muf) ** 2
            self.list_v.append(value)
            if value > self.vfin:
                self.thfin = t
                self.vfin = value


    def th_managment(self, manual_th=0):
        minimo = round(self.tot_n * 0.6)
        
        his, bi = np.histogram(self.list_tot, np.arange(0, 1, 0.01))
        pos = np.where(bi == self.thfin)
        number_new_dataset = np.sum(his[:pos[0][0]])

        print('Total number tiles:         {}\n'
              '60% of dataset:             {}\n'
              'Elemets UncT < Otsu Th:     {}'.format(self.tot_n, minimo, number_new_dataset))

        if minimo > number_new_dataset:
            max_pos = np.where(his == np.max(his[pos[0][0]:]))
            der = np.diff(his[pos[0][0]:max_pos[0][0]])
            print(his[pos[0][0]:max_pos[0][0]])
            print(der)
            max_variation = np.where(der == np.max(der))
            print('MAX VAR --', max_variation)
            new_ph = his[pos[0][0]+max_variation[0][0]]
            npos = np.where(his[pos[0][0]:] > new_ph)
            self.newth = bi[npos[0][0] + pos[0][0]]
        else:
            self.newth = self.thfin

        if manual_th == 0:
            pos1 = np.where(bi == self.newth)
            number_new_dataset1 = np.sum(his[:pos1[0][0]])
        else:
            newTh = [i for i in bi if i > manual_th]
            self.newth = newTh[0]
            pos1 = np.where(bi == self.newth)
            number_new_dataset1 = np.sum(his[:pos1[0][0]])

        print('Elements new dataset:          {}\nNew Th: {}'.format(number_new_dataset1, self.newth))
        return self.newth, self.thfin, number_new_dataset1, number_new_dataset

    def clean_json(self, conclusive_path, progress_callback, view):
        dizio_new = {}
        for i in self.dizio:
            if self.dizio[i]['Unc_tot'] < self.newth:
                dizio_new[i] = self.dizio[i]
        
        name = os.path.join(conclusive_path , self.fname + '_cleanss_js.txt')
        print(name)
        with open(name, 'w') as f:
            json.dump(dizio_new, f)

        self.copy(dizio_new, conclusive_path, progress_callback)

    def copy(self, dizio_new, conclusive_path, progress_callback):
        n_elem = len(dizio_new)
        for k, j in enumerate(dizio_new):
            progress_callback.emit(100*k/n_elem)
            destination = os.path.join(conclusive_path, dizio_new[j]['true_class'])
            shutil.copy2(dizio_new[j]['im_path'], destination)

    def removed_class(self):
        cl = {'AC': 0, 'AD': 0, 'H': 0}
        print('REMOVED CLASS')
        for i in self.dizio:
            if self.dizio[i]['Unc_tot'] > self.newth:
                cl[self.dizio[i]['true_class']] += 1

        with sns.axes_style("darkgrid"):
            plt.bar(cl.keys(), cl.values())
            plt.show()

    def pl(self):
        figure(figsize=(6, 3), dpi=200)
        with sns.axes_style("darkgrid"):
            n2, bins2, patches2 = plt.hist(self.list_tot, 1000, alpha=0.7)
            x = np.arange(0.01, 0.99, 0.01)
            plt.plot(x, np.asarray(self.list_v)*40000, linewidth=3, alpha=0.75, label='Varianza inter-classe')
            plt.title('Histogram total uncertainty')
            plt.axvline(x=self.thfin, color='red', label='Otsu Th')
            plt.axvline(x=self.newth, ls='--', color='k', label='New Th')
            plt.legend(prop={'size': 6})

            plt.xlim(0,0.8)
            plt.show


if __name__ == '__main__':
    t1 = Th('D:/Download/tr_js.txt', 'train')
    # t1.create_list()
    # t1.otsu()
    # t1.th_managment()
    # t1.removed_class()
    # t1.pl()
    # t = t1.clean_json()
    t1.clean_json('D:/test/000')

