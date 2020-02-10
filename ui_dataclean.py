from PyQt5.QtWidgets import (QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QApplication, QLabel, QScrollArea, QMenu, QAction,
                             QFileDialog, QProgressBar, QListWidget, QLineEdit, QButtonGroup, QMessageBox)
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QFont, QIcon
from PyQt5.QtWidgets import QPushButton, QAction, QTabWidget, QRadioButton, QFrame, QCheckBox
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject, QRunnable, QThreadPool, Qt
import sys
import matplotlib
import traceback
import os
from multi_processing_analysis import StartAnalysis
from DropOut import ModelDropOut
from Kl import ModelKl
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import time
from Classification import Classification
from uncertenty_analysis import Th

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
        self.axes.set_xlim(0, 1)
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
        self.setWindowIcon(QIcon('icons/target.ico'))
        self.showMaximized()

    def createAct(self):
        self.aboutAct = QAction("&Tutorial", self,triggered=self.tutorial)
        self.exit = QAction("Exit", self, triggered=self.close)

    def tutorial(self):
        QMessageBox.information(self, "Bayesian datacleaner",
                                "The program is divided in tabs,\n"
                                "you should follow the tab sequence to ensure that all function will work in the"
                                " right way. \n \n "
                                "In the first tab 'Get tiles' you have to select a folder where you want to save all"
                                " data that will be created during the cleaning, after this you can select the train "
                                "and validation folders,"
                                " inside this the svs files should be organized in three class, AC,"
                                " AD, H. When you have done press Start and the process will run. \n"
                                "In Train tab you can select some parameters of the training model, and if needed "
                                "select a specific training and validation set, push start and wait the end of the "
                                "learning process."
                                "Press ok to continue")

    def createMenu(self):
        fileMenu = QMenu("&File", self)
        fileMenu.addAction(self.exit)

        about = QMenu("About", self)
        about.addAction(self.aboutAct)

        self.menuBar().addMenu(fileMenu)
        self.menuBar().addMenu(about)


class MyTableWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.train_path = "D:/test/train"
        self.val_path = "D:/test/val"
        self.new_path_model = "C:/Users/piero/Documents/GitHub/WSI_analysis/Model_1_85aug.h5"
        self.list_ale, self.list_epi, self.list_tot = [], [], []
        self.epoch = 100
        self.model = 'drop'
        self.batch_dim = 100
        self.monte = 5
        self.train_js, self.val_js = "train_js.txt", "test_js.txt"
        self.path_work = "D:/test"
        self.path_tiles_train, self.path_tiles_val, self.selected_th, self.path_save_clean = '', '', '', ''
        self.flag, self.aug = 0, 0
        self.layout = QVBoxLayout(self)

        self.threadPool = QThreadPool()

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()

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
        self.description_batch2 = QLabel('Default value: 100')

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

        self.description_optional = QLabel('Optional, select two folder where can retrive '
                                           'the training end validation tiles:')
        self.retrive_train = QPushButton('Train')
        self.retrive_train.clicked.connect(self.ret_train)
        self.retrive_test = QPushButton('Val')
        self.retrive_test.clicked.connect(self.ret_test)
        self.new_folder = QHBoxLayout(self)
        self.new_folder.addWidget(self.retrive_train)
        self.new_folder.addWidget(self.retrive_test)
        self.new_folder.addStretch(1)

        self.data_aug = QCheckBox('Data Agumentation')
        self.data_aug.stateChanged.connect(self.agumentation)
        # tab2
        self.tab2.layout.addLayout(self.set_model)
        self.tab2.layout.addLayout(self.set_epochs)
        self.tab2.layout.addLayout(self.set_batch_size)
        self.tab2.layout.addWidget(self.data_aug)
        self.tab2.layout.addWidget(QHLine())
        self.tab2.layout.addWidget(self.description_optional)
        self.tab2.layout.addLayout(self.new_folder)
        self.tab2.layout.addWidget(QHLine())
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
        self.start_classify_val.clicked.connect(self.cl_val)

        self.HMonte = QHBoxLayout(self)
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

        # Elements 4
        self.hist_ale = MplCanvas('Aleatoric uncertanty', self, width=5, height=4, dpi=100)
        self.hist_epi = MplCanvas('Epistemic uncertanty', self, width=5, height=4, dpi=100)
        self.hist_tot = MplCanvas('Total uncertanty', self, width=5, height=4, dpi=100)
        self.hist_removed = QPushButton('Show number of tiles that i will remove for class')
        self.hist_removed.hide()
        self.hist_removed.clicked.connect(self.show_class_removed)

        self.description_total_before = QLabel()
        self.description_total_before.hide()
        self.description_total_after = QLabel()
        self.description_total_after.hide()
        self.description_select_data = QLabel('Select which dataset analyze:')
        self.description_types = QLabel('Select one of the two modes, Auto the software will find the correct '
                                        'threshold to divide the images between certain and uncertain; in Manual '
                                        'you have to write the desired value in the text box.')
        self.description_folder = QLabel('Folder where the dataset cleaned will be created:')
        self.folder_cl_data = QPushButton('Select a emplty folder')
        self.folder_cl_data.clicked.connect(self.conclusion_folder)
        self.auto = QRadioButton('Auto')
        self.auto.setEnabled(False)
        self.auto.toggled.connect(self.update_auto)
        self.manual = QRadioButton('Manual')
        self.manual.setEnabled(False)
        self.manual.toggled.connect(self.update_man)
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.auto)
        self.mode_group.addButton(self.manual)
        self.prog_copy = QProgressBar()

        self.dataset_train = QRadioButton('Training set')
        self.dataset_train.toggled.connect(self.clean_train)
        self.dataset_val = QRadioButton('Validation set')
        self.dataset_val.toggled.connect(self.clean_val)
        self.dataset_group = QButtonGroup()
        self.dataset_group.addButton(self.dataset_train)
        self.dataset_group.addButton(self.dataset_val)

        self.manual_value = QLineEdit()
        self.manual_value.hide()
        self.start_clean = QPushButton('Start analysis with manual threshold')
        self.start_clean.clicked.connect(self.push_manual)
        self.start_clean.hide()
        self.create_new_dataset = QPushButton('Create new data set')
        self.create_new_dataset.clicked.connect(self.conclusion_cleaning)
        self.create_new_dataset.setEnabled(False)

        self.mode = QHBoxLayout()
        self.mode.addWidget(self.auto, alignment=Qt.AlignCenter)
        self.mode.addWidget(self.manual, alignment=Qt.AlignCenter)
        self.mode.addWidget(self.manual_value)
        self.mode.addWidget(self.start_clean)
        self.mode.addStretch(1)

        self.hist_v = QVBoxLayout()
        self.hist_v.addWidget(self.hist_ale)
        self.hist_v.addWidget(self.hist_epi)

        self.hist_o = QHBoxLayout()
        self.hist_o.addWidget(self.hist_tot)
        self.hist_o.addLayout(self.hist_v)

        self.folder_cl = QHBoxLayout()
        self.folder_cl.addWidget(self.description_folder)
        self.folder_cl.addWidget(self.folder_cl_data)

        self.number = QVBoxLayout()
        self.number.addWidget(self.description_total_before)
        self.number.addWidget(self.description_total_after)

        self.h_number = QHBoxLayout()
        self.h_number.addLayout(self.number)
        self.h_number.addWidget(self.hist_removed)

        self.h_select_dataset = QHBoxLayout()
        self.h_select_dataset.addWidget(self.description_select_data)
        self.h_select_dataset.addWidget(self.dataset_train)
        self.h_select_dataset.addWidget(self.dataset_val)

        self.otsu_th = QRadioButton('Otsu threshold')
        self.otsu_th.setEnabled(False)
        self.otsu_th.toggled.connect(self.sel_otsu)
        self.new_th = QRadioButton('New threshold')
        self.new_th.setEnabled(False)
        self.new_th.toggled.connect(self.sel_new)
        self.manul_th = QRadioButton('Manual threshold')
        self.manul_th.setEnabled(False)
        self.manul_th.toggled.connect(self.sel_manual)
        self.group_th = QButtonGroup()
        self.group_th.addButton(self.otsu_th)
        self.group_th.addButton(self.new_th)
        self.group_th.addButton(self.manul_th)

        self.h_selection_th = QHBoxLayout()
        self.h_selection_th.addWidget(self.otsu_th)
        self.h_selection_th.addWidget(self.new_th)
        self.h_selection_th.addWidget(self.manul_th)
        self.h_selection_th.addStretch(1)
        # tab 4
        self.tab4.layout.addLayout(self.h_select_dataset)
        self.tab4.layout.addWidget(QHLine())
        self.tab4.layout.addLayout(self.hist_o)
        self.tab4.layout.addLayout(self.h_number)
        self.tab4.layout.addWidget(QHLine())
        self.tab4.layout.addWidget(self.description_types)
        self.tab4.layout.addLayout(self.mode)
        self.tab4.layout.addWidget(QHLine())
        self.tab4.layout.addLayout(self.folder_cl)
        self.tab4.layout.addLayout(self.h_selection_th)
        self.tab4.layout.addWidget(self.create_new_dataset)
        self.tab4.layout.addWidget(self.prog_copy)
        self.tab4.layout.addStretch(1)

        self.tab4.setLayout(self.tab4.layout)

        # Add tabs to widget

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        start_time = time.asctime(time.localtime(time.time()))
        self.log_epoch = "Start  {}".format(start_time)

    def sel_otsu(self):
        self.selected_th = 'otsu'

    def sel_new(self):
        self.selected_th = 'new'

    def sel_manual(self):
        self.selected_th = 'manual'

    def agumentation(self, state):
        if Qt.Checked == state:
            self.aug = 1
            print(self.aug, 'aidsaidiaaaaaaaaaaaaaaaaaaa')
        else:
            self.aug = 0

    def show_class_removed(self):
        self.obj_clean.removed_class()

    def ret_test(self):
        fl = QFileDialog.getExistingDirectory(self, "Select Directory")
        if fl != '':
            self.path_tiles_val = fl
        else:
            pass

    def ret_train(self):
        fl = QFileDialog.getExistingDirectory(self, "Select Directory")
        if fl != '':
            self.path_tiles_train = fl
        else:
            pass

    def conclusion_cleaning(self):
        if not os.path.exists(os.path.join(self.path_save_clean, 'AC')):
            os.mkdir(os.path.join(self.path_save_clean, 'AC'))
            os.mkdir(os.path.join(self.path_save_clean, 'H'))
            os.mkdir(os.path.join(self.path_save_clean, 'AD'))

        work_copy = WorkerLong(self.obj_clean.clean_js, self.selected_th, self.path_save_clean)

        work_copy.signals.result.connect(self.print_output)
        work_copy.signals.progress.connect(self.progress_fn)
        work_copy.signals.progress.connect(self.prog_copy.setValue)
        work_copy.signals.finished.connect(self.thread_complete)
        self.threadPool.start(work_copy)

    def conclusion_folder(self):
        save_fl = QFileDialog.getExistingDirectory(self, "Select Directory")
        if save_fl != '':
            self.path_save_clean = save_fl
            self.create_new_dataset.setEnabled(True)
        else:
            pass

    def first_selection(self):
        fl = QFileDialog.getExistingDirectory(self, "Select Directory")
        if fl != '':
            self.path_work = fl
            self.path_tiles_train = os.path.join(fl, 'train')
            self.path_tiles_val = os.path.join(fl, 'val')
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

    def draw_hist(self, path_js, name):
        self.obj_clean = Th(path_js, name)
        print('sono in f')
        self.list_ale, self.list_epi, self.list_tot = self.obj_clean.create_list()

        self.hist_tot.axes.clear()
        self.hist_tot.axes.set_xlim(0, 1)
        self.hist_tot.axes.set_title('Total uncertainty')
        self.hist_tot.axes.hist(self.list_tot, 500, alpha=0.70, edgecolor='#003153')
        self.hist_tot.draw()
        self.hist_ale.axes.clear()
        self.hist_ale.axes.set_xlim(0, 1)
        self.hist_ale.axes.set_title('Aleatoric uncertainty')
        self.hist_ale.axes.hist(self.list_ale, 100, alpha=0.70, edgecolor='#003153')
        self.hist_ale.draw()
        self.hist_epi.axes.clear()
        self.hist_epi.axes.set_xlim(0, 1)
        self.hist_epi.axes.set_title('Epistemic uncertainty')
        self.hist_epi.axes.hist(self.list_epi, 500, alpha=0.70, edgecolor='#003153')
        self.hist_epi.draw()

    def unlock_an(self):
        self.auto.setEnabled(True)
        self.manual.setEnabled(True)

    def clean_train(self):
        self.draw_hist(self.train_js, 'train')
        self.description_total_before.setText('Total number of tiles before cleaning: {}'.format(len(self.list_tot)))
        self.description_total_before.show()
        self.description_total_after.hide()
        self.hist_removed.hide()
        self.unlock_an()

    def clean_val(self):
        self.draw_hist(self.val_js, 'val')
        self.description_total_before.setText('Total number of tiles before cleaning: {}'.format(len(self.list_tot)))
        self.description_total_before.show()
        self.description_total_after.hide()
        self.hist_removed.hide()
        self.unlock_an()

    def load_kl(self):
        self.model = 'kl'

    def load_drop(self):
        self.model = 'drop'

    def update_auto(self):
        self.obj_clean.otsu()
        self.newth, self.thfin, number_new_dataset1, number_new_dataset = self.obj_clean.th_managment()
        self.description_total_after.setText('Total number of tiles after cleaning: \n'
                                             'Otsu Threshold:   {:10}\n'
                                             'New Threshold:    {:10}'.format(number_new_dataset, number_new_dataset1))
        self.description_total_after.show()
        self.hist_removed.show()
        self.hist_tot.axes.axvline(x=self.newth, ls='--', color='k', label='New Threshold')
        self.hist_tot.axes.axvline(x=self.thfin, color='red', label='Otsu Threshold')
        self.hist_tot.axes.axvline(x=-3, ls='--', color='y', label='Manual Threshold')
        if self.flag == 0:
            self.hist_tot.axes.legend(prop={'size': 10})
            self.flag = 1
        else:
            pass
        self.hist_tot.draw()
        if self.auto.isChecked():
            self.manual_value.hide()
            self.start_clean.hide()
            self.th_update()

    def th_update(self):
        self.otsu_th.setEnabled(True)
        self.new_th.setEnabled(True)
        self.manul_th.setEnabled(True)

    def update_man(self):
        if self.manual.isChecked():
            self.manual_value.show()
            self.start_clean.show()

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

    def push_manual(self):
        tex = float(self.manual_value.text())
        self.th_update()
        if 1 > tex > 0.1:
            print('dentro id')
            self.selected_th = tex
            self.obj_clean.otsu()
            print('TEXT--------', tex)

            self.man, self.thfin, number_new_dataset1, number_new_dataset = self.obj_clean.th_managment(self.selected_th)
            self.description_total_after.setText('Total number of tiles after cleaning: \n'
                                                 'Otsu Threshold:        {:10}\n'
                                                 'Manual Threshold:      {:10}'.format(number_new_dataset, number_new_dataset1))
            self.description_total_after.show()
            self.hist_removed.show()
            self.hist_tot.axes.axvline(x=self.man, ls='--', color='y', label='Manual Threshold')
            self.hist_tot.draw()

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
        t_stamp = time.strftime("%Y_%m%d_%H%M%S")
        if self.model == 'drop':
            self.new_path_model = os.path.join(self.path_work, 'ModelDrop-' + t_stamp + '.h5')
            print(self.train_path)
            obj_model = ModelDropOut(n_model=self.new_path_model, epochs=self.epoch, path_train=self.path_tiles_train,
                                     path_val=self.path_tiles_val, b_dim=int(self.batch_dim), aug=self.aug)
        else:
            self.new_path_model = os.path.join(self.path_work, 'ModelKl-' + t_stamp + '.h5')
            obj_model = ModelKl(n_model=self.new_path_model, epochs=self.epoch, path_train=self.path_tiles_train,
                                path_val=self.path_tiles_val, b_dim=int(self.batch_dim), aug=self.aug)

        k = WorkerLong(obj_model.start_train)
        k.signals.result.connect(self.print_output)
        k.signals.progress.connect(self.progrestrain.setValue)
        k.signals.result1.connect(self.tr_view)
        k.signals.finished.connect(self.thread_train_complete)
        self.threadPool.start(k)

    def cl_train(self):
        self.start_an('train')
        self.train_js = os.path.join(self.path_work, 'train', 'dictionary_js.txt')

    def cl_val(self):
        self.start_an('val')
        self.val_js = os.path.join(self.path_work, 'val', 'dictionary_js.txt')

    def start_an(self, data):
        path = os.path.join(self.path_work, data)

        cls = Classification(path, ty='datacleaning')
        worker_cl = WorkerLong(cls.classify, 'datacleaning', self.monte, self.new_path_model)
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