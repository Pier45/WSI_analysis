from PyQt5.QtWidgets import (QWidget, QMainWindow, QVBoxLayout, QSizePolicy, QHBoxLayout,
                             QSplitter, QStyleFactory, QApplication, QLabel, QScrollArea, QMenu, QAction,
                             QFileDialog, QProgressBar, QListWidget, QLineEdit)
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QFont, QIcon
from PyQt5.QtWidgets import QPushButton, QAction, QTabWidget, QRadioButton, QFrame
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
#import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import time
from Classification import Classification

matplotlib.use('Qt5Agg')


class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class QVLine(QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, name, parent=None, width=10, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_title(name)
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
        self.setWindowTitle(self.title)

        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
        self.createAct()
        self.createMenu()
        self.showMaximized()

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
        self.train_path = "D:/test/train"
        self.val_path = "D:/test/val"
        self.model_path = "C:/Users/piero/Documents/GitHub/WSI_analysis/Model_1_85aug.h5"
        self.epoch = 100
        self.model = 'drop'
        self.batch_dim = 100
        self.monte = 5
        self.path_work = "D:/test"
        self.layout = QVBoxLayout(self)
        self.HMonte = QHBoxLayout(self)

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
        self.first = QLabel("First things first, select a place where put all data sets:")
        self.first_button = QPushButton("Select folder")
        self.first_button.clicked.connect(self.first_selection)

        newfont = QFont("Times", 15, QFont.Bold)
        self.title_train = QLabel('TRAINING SET')
        self.title_train.setFont(newfont)
        self.title_val = QLabel('VALIDATION SET')
        self.title_val.setFont(newfont)

        self.start = QPushButton("Start")
        self.start.clicked.connect(self.start_tiles)
        self.description_t = QLabel("Press here to select the train folder: ")
        self.description_v = QLabel("Press here to select the validation folder: ")
        self.folder_train = QPushButton("Select folder")
        self.folder_train.clicked.connect(self.select_folder_train)
        self.folder_val = QPushButton("Select folder")
        self.folder_val.clicked.connect(self.select_folder_val)
        self.prog1 = QProgressBar(self)
        self.prog2 = QProgressBar(self)
        self.prog3 = QProgressBar(self)
        self.prog1_v = QProgressBar(self)
        self.prog2_v = QProgressBar(self)
        self.prog3_v = QProgressBar(self)
        # train
        self.list_ac = QListWidget(self)
        self.list_ad = QListWidget(self)
        self.list_h = QListWidget(self)
        # validation
        self.list_ac_v = QListWidget(self)
        self.list_ad_v = QListWidget(self)
        self.list_h_v = QListWidget(self)

        self.first_layout = QHBoxLayout(self)
        self.first_layout.addWidget(self.first)
        self.first_layout.addWidget(self.first_button)

        self.h_train = QHBoxLayout(self)
        self.h_train.addWidget(self.description_t)
        self.h_train.addWidget(self.folder_train)

        self.h_val = QHBoxLayout(self)
        self.h_val.addWidget(self.description_v)
        self.h_val.addWidget(self.folder_val)

        self.list_t = QHBoxLayout(self)
        self.list_t.addWidget(self.list_ac)
        self.list_t.addWidget(self.list_ad)
        self.list_t.addWidget(self.list_h)

        self.list_v = QHBoxLayout(self)
        self.list_v.addWidget(self.list_ac_v)
        self.list_v.addWidget(self.list_ad_v)
        self.list_v.addWidget(self.list_h_v)

        self.pr = QHBoxLayout(self)
        self.pr.addWidget(self.prog1)
        self.pr.addWidget(self.prog2)
        self.pr.addWidget(self.prog3)

        self.pr_v = QHBoxLayout(self)
        self.pr_v.addWidget(self.prog1_v)
        self.pr_v.addWidget(self.prog2_v)
        self.pr_v.addWidget(self.prog3_v)

        # tab1
        self.tab1.layout.addLayout(self.first_layout)
        self.tab1.layout.addWidget(QHLine())
        self.tab1.layout.addWidget(self.title_train)
        self.tab1.layout.addLayout(self.h_train)
        self.tab1.layout.addLayout(self.list_t)
        self.tab1.layout.addLayout(self.pr)
        self.tab1.layout.addWidget(QHLine())
        self.tab1.layout.addWidget(self.title_val)
        self.tab1.layout.addLayout(self.h_val)
        self.tab1.layout.addLayout(self.list_v)
        self.tab1.layout.addLayout(self.pr_v)
        self.tab1.layout.addWidget(QHLine())
        self.tab1.layout.addWidget(self.start)

        self.tab1.setLayout(self.tab1.layout)

        # Elements section 2
        self.progrestrain = QProgressBar(self)
        self.text = QLineEdit(self)
        self.text_batch = QLineEdit(self)
        self.start_train = QPushButton("Start")
        self.start_train.clicked.connect(self.train)
        self.state_train = QLabel("Press start to train the model.")
        self.state_train.setMargin(10)
        self.state_train.setFixedWidth(600)
        self.state_train.setFixedHeight(1500)
        self.state_train.setAlignment(Qt.AlignTop)

        self.scrolltr = QScrollArea()
        self.scrolltr.setAlignment(Qt.AlignTop)
        self.scrolltr.setWidget(self.state_train)

        self.description_tr = QLabel('Insert numer of epochs:  ')
        self.description_batch = QLabel('Insert dimension of batch:  ')
        self.description_batch2 = QLabel('Default value: 100 ')

        self.description_model = QLabel('Select one of the 2 available models:')
        self.description_tr2 = QLabel('Default epochs: 100')
        self.kl = QRadioButton('Kl divergence')
        self.kl.toggled.connect(self.load_kl)
        self.drop = QRadioButton('Drop-Out')
        self.drop.setChecked(True)
        self.drop.toggled.connect(self.load_drop)

        self.ok_text = QPushButton('Ok')
        self.ok_text.clicked.connect(self.ok_epochs)

        self.ok_batc = QPushButton('Ok')
        self.ok_batc.clicked.connect(self.ok_batch)

        self.set_epochs = QHBoxLayout(self)
        self.set_epochs.addWidget(self.description_tr)
        self.set_epochs.addWidget(self.text)
        self.set_epochs.addWidget(self.ok_text)
        self.set_epochs.addWidget(self.description_tr2)

        self.set_batch_size = QHBoxLayout(self)
        self.set_batch_size.addWidget(self.description_batch)
        self.set_batch_size.addWidget(self.text_batch)
        self.set_batch_size.addWidget(self.ok_batc)
        self.set_batch_size.addWidget(self.description_batch2)

        self.set_model = QHBoxLayout(self)
        self.set_model.addWidget(self.description_model)
        self.set_model.addWidget(self.kl)
        self.set_model.addWidget(self.drop)

        # tab2
        self.tab2.layout.addLayout(self.set_model)
        self.tab2.layout.addLayout(self.set_epochs)
        self.tab2.layout.addLayout(self.set_batch_size)
        self.tab2.layout.addWidget(self.start_train)

        self.tab2.layout.addWidget(self.scrolltr)
        self.tab2.layout.addWidget(self.progrestrain)

        self.tab2.setLayout(self.tab2.layout)

        # Elements section 3
        self.description_clas = QLabel('The purpose of this step is to obtain the values of uncertainty values')
        self.description_monte = QLabel('Write here Monte Carlo samples:')
        self.title_train_cl = QLabel('TRAINING SET')
        self.title_test_cl = QLabel('VALIDATION SET')
        self.title_train_cl.setFont(newfont)
        self.title_test_cl.setFont(newfont)

        self.description_monte2 = QLabel('Default value: {}'.format(self.monte))
        self.text_monte = QLineEdit()
        self.prog_monte = QProgressBar()
        self.ok_monte = QPushButton('Ok')
        self.ok_monte.clicked.connect(self.ok_m)
        self.start_classify_train = QPushButton('Start')
        self.start_classify_train.clicked.connect(self.cl_train)

        self.start_classify_val = QPushButton('Start')
        self.start_classify_val.clicked.connect(self.cl_test)


        self.HMonte.addWidget(self.description_monte)
        self.HMonte.addWidget(self.text_monte)
        self.HMonte.addWidget(self.ok_monte)
        self.HMonte.addWidget(self.description_monte2)

        # tab 3
        self.tab3.layout.addWidget(self.description_clas, alignment=Qt.AlignTop)
        self.tab3.layout.addWidget(QHLine())
        self.tab3.layout.addLayout(self.HMonte)
        self.tab3.layout.addWidget(self.title_train_cl)
        self.tab3.layout.addWidget(self.start_classify_train)
        self.tab3.layout.addWidget(QHLine())
        self.tab3.layout.addWidget(self.title_test_cl)
        self.tab3.layout.addWidget(self.start_classify_val)

        self.tab3.layout.addStretch(1)
        self.tab3.layout.addWidget(self.prog_monte)

        self.tab3.setLayout(self.tab3.layout)

        list_ale = [6, 4, 5, 5, 2, 3, 3, 1, 1]
        list_epi = [6, 4, 5, 5, 5, 5, 5, 1, 1]
        list_tot = [6, 4, 5, 5, 2, 3, 3, 1, 1]

        # Elements 4
        hist_ale = MplCanvas('Aleatoric uncertanty', self, width=5, height=4, dpi=100)
        hist_ale.axes.hist(list_ale, 50, alpha=0.75)
        hist_epi = MplCanvas('Epistemic uncertanty', self, width=5, height=4, dpi=100)
        hist_epi.axes.hist(list_epi, 50, alpha=0.75)
        hist_tot = MplCanvas('Total uncertanty', self, width=5, height=4, dpi=100)
        hist_tot.axes.hist(list_tot, 50, alpha=0.75)
        hist_removed = MplCanvas('Tiles removed for class', self, width=5, height=4, dpi=100)
        hist_removed.axes.hist(list_tot, 50, alpha=0.75)
        self.description_total_before = QLabel('Total number of tiles before data cleaning: 45000')
        self.description_select_data = QLabel('Select which dataset analyze:')
        self.description_total_after = QLabel('Total number of tiles after data cleaning: 45000')
        self.description_total_after.hide()
        self.description_types = QLabel('Select one of the two modes, Auto the software will find the correct'
                                        'threshold to divide the images between certain and uncertain; in Manual '
                                        'you have to write the desired value in the text box.')
        self.description_folder = QLabel('Folder where the dataset cleaned will be created:')
        self.folder_cl_data = QPushButton('Select a Folder')
        self.auto = QRadioButton('Auto')
        self.auto.toggled.connect(self.update_auto)
        self.manual = QRadioButton('Manual')
        self.manual.toggled.connect(self.update_man)
        self.dataset_train = QRadioButton('Training set')
        self.dataset_val = QRadioButton('Validation set')
        self.manual_value = QLineEdit()
        self.manual_value.hide()
        self.start_clean = QPushButton('Start analysis')
        self.create_new_dataset = QPushButton('Create new data set')

        self.mode = QHBoxLayout()
        self.mode.addWidget(self.auto, alignment=Qt.AlignCenter)
        self.mode.addWidget(self.manual, alignment=Qt.AlignCenter)
        self.mode.addWidget(self.manual_value)

        self.hist_v = QVBoxLayout()
        self.hist_v.addWidget(hist_ale)
        self.hist_v.addWidget(hist_epi)

        self.hist_o = QHBoxLayout()
        self.hist_o.addWidget(hist_tot)
        self.hist_o.addLayout(self.hist_v)

        self.folder_cl = QHBoxLayout()
        self.folder_cl.addWidget(self.description_folder)
        self.folder_cl.addWidget(self.folder_cl_data)

        self.number = QVBoxLayout()
        self.number.addWidget(self.description_total_before)
        self.number.addWidget(self.description_total_after)

        self.h_number = QHBoxLayout()
        self.h_number.addLayout(self.number)
        self.h_number.addWidget(hist_removed)

        self.h_select_dataset = QHBoxLayout()
        self.h_select_dataset.addWidget(self.description_select_data)
        self.h_select_dataset.addWidget(self.dataset_train)
        self.h_select_dataset.addWidget(self.dataset_val)
        # tab 4
        self.tab4.layout.addLayout(self.h_select_dataset)
        self.tab4.layout.addWidget(QHLine())
        self.tab4.layout.addLayout(self.hist_o)
        self.tab4.layout.addLayout(self.h_number)
        self.tab4.layout.addWidget(QHLine())
        self.tab4.layout.addWidget(self.description_types)
        self.tab4.layout.addLayout(self.mode)
        self.tab4.layout.addWidget(self.start_clean)
        self.tab4.layout.addWidget(QHLine())
        self.tab4.layout.addLayout(self.folder_cl)
        self.tab4.layout.addWidget(self.create_new_dataset)
        self.tab4.layout.addStretch(1)

        self.tab4.setLayout(self.tab4.layout)

        # Add tabs to widget

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        start_time = time.asctime(time.localtime(time.time()))
        self.log_epoch = "Start  {}".format(start_time)

    def first_selection(self):
        fl = QFileDialog.getExistingDirectory(self, "Select Directory")
        if fl != '':
            self.path_work = fl
        else:
            pass

    def ok_m(self):
        tex = self.text_monte.text()
        if tex.isdecimal() and int(tex) > 0:
            self.description_monte2.setText('Monte Carlo samples:  {}'.format(tex))
            self.monte = tex
        else:
            pass

    def ok_batch(self):
        tex = self.text_batch.text()
        if tex.isdecimal() and int(tex) > 0:
            self.description_batch2.setText('Batch dimension:  {}'.format(tex))
            self.batch_dim = tex
        else:
            pass

    def load_kl(self):
        self.model = 'kl'

    def load_drop(self):
        self.model = 'drop'

    def update_auto(self):
        if self.auto.isChecked():
            self.manual_value.hide()

    def update_man(self):
        if self.manual.isChecked():
            self.manual_value.show()

    def select_folder_train(self):
        self.train_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.train_path != '':
            self.cl = os.listdir(self.train_path)
            try:
                for i in self.cl:
                    nfiles = os.listdir(os.path.join(self.train_path, i))
                    if i == 'AC' or i == 'ac':
                        self.list_ac.addItems(nfiles)
                    elif i == 'AD' or i == 'ad':
                        self.list_ad.addItems(nfiles)
                    elif i == 'H' or i == 'h':
                        self.list_h.addItems(nfiles)
                    else:
                        pass
                    print(self.train_path)
                    print("we jack")
            except:
                pass
        else:
            pass

    def select_folder_val(self):
        self.val_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.val_path != '':
            self.cl = os.listdir(self.val_path)
            try:
                for i in self.cl:
                    nfiles = os.listdir(os.path.join(self.val_path, i))
                    if i == 'AC' or i == 'ac':
                        self.list_ac_v.addItems(nfiles)
                    elif i == 'AD' or i == 'ad':
                        self.list_ad_v.addItems(nfiles)
                    elif i == 'H' or i == 'h':
                        self.list_h_v.addItems(nfiles)
                    else:
                        pass
                    print(self.val_path)
                    print("we jack")
            except:
                pass
        else:
            pass

    def progress_fn(self, n):
        print("%d%% done" % n)

    def print_output(self, s):
        self.state_train.setText(str(s))
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def thread_train_complete(self):
        self.state_train.setText('Training Completed!')
        print("THREAD COMPLETE!")

    def th_tiles(self, pr, path, name):

        for y, k in enumerate(self.cl, 0):
            save_folder = os.path.join(self.path_work, name, k)
            f_p = os.path.join(path, k)
            if os.path.exists(save_folder):
                pass
            else:
                os.makedirs(save_folder)

            st_tile = StartAnalysis(tile_size=256,  lev_sec=0)
            print('inizio il threading \n', f_p, save_folder)
            name_th = str(k) + name
            name_th = WorkerLong(st_tile.list_files, f_p, save_folder)

            name_th.signals.result.connect(self.print_output)
            name_th.signals.progress.connect(self.progress_fn)
            name_th.signals.progress.connect(pr[y].setValue)
            name_th.signals.finished.connect(self.thread_complete)
            self.threadPool.start(name_th)

    def start_tiles(self):
        pr = [[self.prog1, self.prog2, self.prog3], [self.prog1_v, self.prog2_v, self.prog3_v]]
        ph = [self.train_path, self.val_path]
        datas = ['train', 'val']

        for t, i in enumerate(datas):
            self.th_tiles(pr[t], ph[t], name=i)

    def ok_epochs(self):
        textboxValue = self.text.text()
        if textboxValue.isdecimal() and int(textboxValue) > 0:
            self.description_tr2.setText('Limit epochs:  {}'.format(textboxValue))
            self.epoch = textboxValue
        else:
            pass

        print(textboxValue)

    def train(self):
        self.state_train.setText('The training is starting, in few second other information will be showed...')

        if self.model == 'drop':
            obj_model = ModelDropOut(epochs=self.epoch, path_train=self.train_path, path_val=self.val_path, b_dim=int(self.batch_dim))
        else:
            obj_model = ModelDropOut(epochs=self.epoch, path_train=self.train_path, path_val=self.val_path, b_dim=int(self.batch_dim))

        k = WorkerLong(obj_model.start_train)
        k.signals.result.connect(self.print_output)
        k.signals.progress.connect(self.progrestrain.setValue)
        k.signals.result1.connect(self.tr_view)
        k.signals.finished.connect(self.thread_train_complete)
        self.threadPool.start(k)

    def cl_train(self):
        self.start_an('train')

    def cl_test(self):
        self.start_an('val')

    def start_an(self, data):
        path = os.path.join(self.path_work, data)

        cls = Classification(path, ty='datacleaning')
        worker_cl = WorkerLong(cls.classify, 'datacleaning', self.monte, self.model_path)
        worker_cl.signals.result.connect(self.print_output)
        worker_cl.signals.progress.connect(self.progress_fn)
        worker_cl.signals.progress.connect(self.prog_monte.setValue)
        worker_cl.signals.finished.connect(self.thread_complete)
        self.threadPool.start(worker_cl)

    def tr_view(self, val):
        if 'Epoch' in str(val):
            self.log_epoch = self.log_epoch + '\n' + str(val)
            self.state_train.setText(str(self.log_epoch))
            print('trova la parola epoca EMETTE:', self.log_epoch)
        else:
            show_b = self.log_epoch + '\n' + str(val)
            self.state_train.setText(str(show_b))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())