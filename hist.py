import sys
import matplotlib
import os
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import (QWidget, QMainWindow, QVBoxLayout, QFrame, QSizePolicy, QHBoxLayout,
                             QSplitter, QStyleFactory, QApplication, QLabel, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QFont, QIcon


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, name = 'dadadada'):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        list_ale, list_epi, list_tot = self.load_dict()

        with sns.axes_style("darkgrid"):
            sc1 = MplCanvas(self, width=5, height=4, dpi=100)
            sc1.axes.hist(list_ale, 50, alpha=0.75)

            sc2 = MplCanvas(self, width=5, height=4, dpi=100)
            sc2.axes.hist(list_epi, 50, alpha=0.75)

            sc3 = MplCanvas(self, width=5, height=4, dpi=100)
            sc3.axes.hist(list_tot, 50, alpha=0.75)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar1 = NavigationToolbar(sc1, self)
        toolbar2 = NavigationToolbar(sc2, self)
        toolbar3 = NavigationToolbar(sc3, self)
        image = QImage('C:/Users/piero/Test/31400_2/thumbnail/th.png')

        imageLabel2 = QLabel()
        imageLabel2.setPixmap(QPixmap.fromImage(image))

        scroll = QScrollArea()
        scroll.setWidget(imageLabel2)

        layout = QVBoxLayout()
        layout.addWidget(toolbar1)
        layout.addWidget(sc1)
        layout.addWidget(toolbar2)
        layout.addWidget(sc2)
        layout.addWidget(toolbar3)
        layout.addWidget(sc3)

        widget1 = QWidget()
        widget1.setLayout(layout)
        widget1.setMaximumWidth(500)
        self.setCentralWidget(widget1)

        lh = QHBoxLayout()
        lh.addWidget(scroll)
        lh.addWidget(widget1)


        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QWidget()
        widget.setLayout(lh)
        self.setCentralWidget(widget)
        self.setWindowTitle('testiamo')

        self.show()

    def load_dict(self):
        name_f = os.path.join('C:/Users/piero/Test/66064_2', 'dictionary_js.txt')
        with open(name_f, 'r') as f:
            dictionary = json.load(f)

        dizio = dictionary

        list_ale = []
        list_epi = []
        list_tot = []

        for i in dizio:
            print(i)
            list_ale.append(float(dizio[i]['ale']))
            list_epi.append(float(dizio[i]['epi']))
            dizio[i]['Unc_tot'] = float(dizio[i]['ale']) + float(dizio[i]['epi'])
            list_tot.append(dizio[i]['Unc_tot'])

        return list_ale, list_epi, list_tot

# def load_dict():
#     name_f = os.path.join('C:/Users/piero/Test/66064_2', 'dictionary_js.txt')
#     with open(name_f, 'r') as f:
#         dictionary = json.load(f)
#
#     return dictionary
#
#
# dizio = load_dict()
#
# list_ale = []
# list_epi = []
# list_tot = []
#
# for i in dizio:
#     print(i)
#     list_ale.append(float(dizio[i]['ale']))
#     list_epi.append(float(dizio[i]['epi']))
#     dizio[i]['Unc_tot'] = float(dizio[i]['ale']) + float(dizio[i]['epi'])
#     list_tot.append(dizio[i]['Unc_tot'])
#
# figh = plt.figure()
# figh.set_size_inches(15, 13)
# with sns.axes_style("darkgrid"):
#     plt.subplot(3, 1, 1)
#     n, bins, patches = plt.hist(list_epi, 50, alpha=0.75)
#     plt.xlim(0, 0.7)
#     plt.title('Epistemic uncertenty')
#     plt.subplot(3, 1, 2)
#     plt.hist(list_ale, 50, alpha=0.75)
#     plt.title('Aleatoric uncertenty')
#     plt.xlim(0, 0.7)
#     plt.ylim(0, 200)
#     plt.subplot(3, 1, 3)
#     plt.hist(list_tot, 50, alpha=0.75)
#     plt.title('Total Uncertanty distribution')
#     plt.xlim(0, 0.7)
#     plt.ylim(0, 200)
#
# plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    app.exec_()