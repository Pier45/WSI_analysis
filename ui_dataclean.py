from PyQt5.QtWidgets import (QWidget, QMainWindow, QVBoxLayout, QSizePolicy, QHBoxLayout,
                             QSplitter, QStyleFactory, QApplication, QLabel, QScrollArea, QMenu, QAction,
                             QFileDialog, QProgressBar, QListWidget, QLineEdit)
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QFont, QIcon
from PyQt5.QtWidgets import QPushButton, QAction, QTabWidget, QRadioButton
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject, QRunnable, QThreadPool, Qt
import sys
import matplotlib
import traceback
import os
from multi_processing_analysis import StartAnalysis
from DropOut import ModelDropOut
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, name='dadadada'):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class WorkerSignals(QObject):

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    result1 = pyqtSignal(object)


class WorkerLong(QRunnable):
    """Useful to start more complicated threads, it allows to interact with the state of the thread, useful
    in progress bar and in same case where need to know if the process is ended"""

    def __init__(self, fn, *args, **kwargs):
        super(WorkerLong, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress
        self.kwargs['view'] = self.signals.result1


    @pyqtSlot()
    def run(self):

        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Bayesian Datacleaning'
        self.left = 900
        self.top = 300
        self.width = 900
        self.height = 600
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
        self.createAct()
        self.createMenu()
        self.show()

    def createAct(self):
        self.aboutAct = QAction("&About", self)

    def createMenu(self):
        fileMenu = QMenu("&File", self)
        fileMenu.addAction(self.aboutAct)

        about = QMenu("About", self)

        self.menuBar().addMenu(fileMenu)
        self.menuBar().addMenu(about)


class MyTableWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.start_path = ""
        self.epoch = 400
        self.path_work = 'C:/Users/piero/test2'
        self.layout = QVBoxLayout(self)
        self.Hlay = QHBoxLayout(self)
        self.list = QHBoxLayout(self)
        self.HMonte = QHBoxLayout(self)
        self.set_train = QHBoxLayout(self)
        self.pr = QHBoxLayout(self)

        self.threadPool = QThreadPool()

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        #self.tabs.resize(500, 400)

        # Add tabs
        self.tabs.addTab(self.tab1, "Get Tiles")
        self.tabs.addTab(self.tab2, "Training")
        self.tabs.addTab(self.tab3, "Classify")
        self.tabs.addTab(self.tab4, "Datacleaning")


        # Create first tab
        self.tab1.layout = QVBoxLayout(self)
        self.tab2.layout = QVBoxLayout(self)
        self.tab3.layout = QVBoxLayout(self)
        self.tab4.layout = QVBoxLayout(self)

        # Elements section 1
        self.start = QPushButton("Start")
        self.start.clicked.connect(self.start_tiles)
        self.description = QLabel("Press here to select the train folder: ")
        self.folder = QPushButton("Select folder")
        self.folder.clicked.connect(self.select_folder)
        self.list_ac = QListWidget(self)
        self.list_ad = QListWidget(self)
        self.list_h = QListWidget(self)

        self.Hlay.addWidget(self.description)
        self.Hlay.addWidget(self.folder)

        self.list.addWidget(self.list_ac)
        self.list.addWidget(self.list_ad)
        self.list.addWidget(self.list_h)

        self.prog1 = QProgressBar(self)
        self.prog2 = QProgressBar(self)
        self.prog3 = QProgressBar(self)
        self.pr.addWidget(self.prog1)
        self.pr.addWidget(self.prog2)
        self.pr.addWidget(self.prog3)

        # tab1
        self.tab1.layout.addLayout(self.Hlay)
        self.tab1.layout.addWidget(self.start)
        self.tab1.layout.addLayout(self.list)
        self.tab1.layout.addLayout(self.pr)

        self.tab1.setLayout(self.tab1.layout)

        # Elements section 2
        self.progrestrain = QProgressBar(self)
        self.text = QLineEdit(self)
        self.start_train = QPushButton("Start")
        self.start_train.clicked.connect(self.train)
        self.state_train = QLabel("Press start to train the model.")
        self.state_train.setMargin(10)
        self.scrolltr = QScrollArea()
        self.scrolltr.setAlignment(Qt.AlignTop)
        self.scrolltr.setWidget(self.state_train)

        self.description_tr = QLabel('Insert numer of epochs: ')
        self.description_tr2 = QLabel('Default epochs: 400')

        self.ok_text = QPushButton('Ok')
        self.ok_text.clicked.connect(self.ok_epochs)

        self.set_train.addWidget(self.description_tr)
        self.set_train.addWidget(self.text)
        self.set_train.addWidget(self.ok_text)
        self.set_train.addWidget(self.description_tr2)

        # tab2
        self.tab2.layout.addLayout(self.set_train)
        self.tab2.layout.addWidget(self.start_train)

        self.tab2.layout.addWidget(self.scrolltr)
        self.tab2.layout.addWidget(self.progrestrain)

        self.tab2.setLayout(self.tab2.layout)

        # Elements section 3
        self.description_clas = QLabel('The purpose of this step is to obtain the values of uncertainty values')
        self.description_monte = QLabel('Write here Monte Carlo samples:')
        self.text_monte = QLineEdit()
        self.prog_monte = QProgressBar()
        self.ok_monte = QPushButton('Ok')

        self.HMonte.addWidget(self.description_monte, alignment=Qt.AlignTop)
        self.HMonte.addWidget(self.text_monte, alignment=Qt.AlignTop)
        self.HMonte.addWidget(self.ok_monte, alignment=Qt.AlignTop)

        # tab 3
        self.tab3.layout.addWidget(self.description_clas, alignment=Qt.AlignTop)
        self.tab3.layout.addLayout(self.HMonte)
        self.tab3.layout.addWidget(self.prog_monte)

        self.tab3.setLayout(self.tab3.layout)

        list_ale = [6, 4, 5, 5, 2, 3, 3, 1, 1]
        list_epi = [6, 4, 5, 5, 2, 3, 3, 1, 1]
        list_tot = [6, 4, 5, 5, 2, 3, 3, 1, 1]

        # Elements 4
        hist_ale = MplCanvas(self, width=5, height=4, dpi=100)
        hist_ale.axes.hist(list_ale, 50, alpha=0.75)
        hist_epi = MplCanvas(self, width=5, height=4, dpi=100)
        hist_epi.axes.hist(list_epi, 50, alpha=0.75)
        hist_tot = MplCanvas(self, width=5, height=4, dpi=100)
        hist_tot.axes.hist(list_tot, 50, alpha=0.75)
        self.description_total_before = QLabel('Total number of tiles before datacleaning: 45000')
        self.description_total_after = QLabel('Total number of tiles after datacleaning: 45000')
        self.description_types = QLabel('Select one of the two modes, Auto the software will find the correct'
                                        'threashold to divide the images between certain and Uncartain; in Manual '
                                        'you have to write the disired value in the text box.')
        self.auto = QRadioButton('Auto')
        self.manual = QRadioButton('Manual')
        self.manual_value = QLineEdit()
        self.start_clean = QPushButton('Start analysis')

        self.mode = QHBoxLayout()
        self.mode.addWidget(self.auto)
        self.mode.addWidget(self.manual)
        self.mode.addWidget(self.manual_value)

        self.hist_v = QVBoxLayout()
        self.hist_v.addWidget(hist_ale)
        self.hist_v.addWidget(hist_epi)

        self.hist_o = QHBoxLayout()
        self.hist_o.addWidget(hist_tot)
        self.hist_o.addLayout(self.hist_v)

        self.number = QHBoxLayout()
        self.number.addWidget(self.description_total_before)
        self.number.addWidget(self.description_total_after)
        # tab 4
        self.tab4.layout.addLayout(self.hist_o)
        self.tab4.layout.addLayout(self.number)
        self.tab4.layout.addWidget(self.description_types)
        self.tab4.layout.addLayout(self.mode)
        self.tab4.layout.addWidget(self.start_clean)


        self.tab4.setLayout(self.tab4.layout)

        # Add tabs to widget

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def select_folder(self):
        self.start_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.cl = os.listdir(self.start_path)

        for i in self.cl:
            nfiles = os.listdir(os.path.join(self.start_path, i))
            if i == 'AC' or i == 'ac':
                self.list_ac.addItems(nfiles)
            elif i == 'AD' or i == 'ad':
                self.list_ad.addItems(nfiles)
            elif i == 'H' or i == 'h':
                self.list_h.addItems(nfiles)
            else:
                pass

        print(self.start_path)
        print("we jack")

    def progress_fn(self, n):
        print("%d%% done" % n)

    def print_output(self, s):
        self.state_train.setText(str(s))
        print(s)

    def caso(self, val):
        self.state_train.setText(str(val))

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def thread_train_complete(self):
        self.state_train.setText('Training Completed!')
        print("THREAD COMPLETE!")

    def start_tiles(self):
        vect_pr = [self.prog1, self.prog2, self.prog3]
        for y, k in enumerate(self.cl, 0):
            save_folder = os.path.join(self.path_work, k)
            f_p = os.path.join(self.start_path, k)
            if os.path.exists(save_folder):
                pass
            else:
                os.mkdir(save_folder)

            st_tile = StartAnalysis(tile_size=256,  lev_sec=0)
            print('inizio il threading \n', f_p, save_folder)
            k = WorkerLong(st_tile.list_files, f_p, save_folder)

            k.signals.result.connect(self.print_output)
            k.signals.progress.connect(self.progress_fn)
            k.signals.progress.connect(vect_pr[y].setValue)
            k.signals.finished.connect(self.thread_complete)
            self.threadPool.start(k)

        print('get the tiles')
        pass

    def ok_epochs(self):
        textboxValue = self.text.text()
        if textboxValue.isdecimal() and int(textboxValue) > 0:
            self.description_tr2.setText('Limit epochs: {}'.format(textboxValue))
            self.epoch = textboxValue
        else:
            pass

        print(textboxValue)

    def train(self):
        self.state_train.setText('The training is starting...')
        Obj_model = ModelDropOut(epochs=100, path_train='C:/Users/piero/test2', path_val='C:/Users/piero/test2')

        k = WorkerLong(Obj_model.start_train)

        k.signals.result.connect(self.print_output)
        k.signals.progress.connect(self.progress_fn)
        k.signals.result1.connect(self.caso)
        k.signals.finished.connect(self.thread_train_complete)
        self.threadPool.start(k)

        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())