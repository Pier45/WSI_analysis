from PyQt5.QtCore import *
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QFont, QIcon
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel, QPushButton,
    QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QToolBar,  QDialog, QHBoxLayout, QFrame,
    QSplitter, QStyleFactory)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
import os
import ctypes
import webbrowser
import time
import traceback
import sys
from multi_processing_analysis import StartAnalysis
from progress_bar import Actions
from Classification import Classification
from modifyset import Mod


class Worker(QRunnable):
    """Useful to run easy threads"""
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):

        self.fn(*self.args, **self.kwargs)


class WorkerSignals(QObject):

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


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


class ImageViewer(QMainWindow):
    def __init__(self):
        self.path_work, self.res_path = '', ''
        self.obj_an, self.fileName = '', ''
        self.levi, self.ny = 0, 0
        self.type_an = 'fast'
        self.model_name = 'Model_1_85aug.h5'
        self.monte_c = 5
        self.numx_start, self.numx_stop, self.list_proc, self.start_i, self.stop_i = [], [], [], [], []
        self.lev_sec = 1
        user32 = ctypes.windll.user32
        self.screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

        super(ImageViewer, self).__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0
        self.imageLabel = QLabel("Steps to start the analysis:\n \n"
                                 "1) File         ---> Select svs or select the yellow folder in the toolbar\n\n"
                                 "2) Analysis ---> Stat analysis or select the green arrow in the toolbar ")

        self.imageLabel.setFont(QFont("Akzidenz Grotesk", 15, QFont.Black))
        self.setCentralWidget(self.imageLabel)

        self.imageLabel.setBackgroundRole(QPalette.Dark)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)
        self.scrollArea.setAlignment(Qt.AlignCenter)

        self.threadPool = QThreadPool()

        self.toolbar = QToolBar("My main toolbar")
        self.addToolBar(self.toolbar)
        self.toolbar.setStyleSheet("QToolBar{spacing:15px;}")

        #self.button = QPushButton("Test")
        #self.scrollArea.setWidget(self.button)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Bayesian Analayzer")
        self.setWindowIcon(QIcon('icons/target.ico'))
        self.showMaximized()

    def open(self):
        """This method is the starting point, here is selected the svs files and is created the thumbnail that is
        immediately showed to the user"""

        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "*.svs")

        self.fileName = fileName

        if fileName:
            self.path_work = self.first_step(fileName)
            self.res_path = self.path_work + 'result/'

            image = QImage(self.path_work + 'thumbnail/th.png')
            self.imageLabel.setPixmap(QPixmap.fromImage(image))

            self.scaleFactor = 1
            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.startAnalysisAct.setEnabled(True)
            self.start_vis_deepAct.setEnabled(True)
            self.updateActions()
            self.thread_manager()

            try:
                image_size = [image.width(), image.height()]

                if image_size[0] > image_size[1]:
                    p = 0
                else:
                    p = 1

                print(image_size)
                print(self.screensize[0], self.screensize[1])

                if not self.fitToWindowAct.isChecked():
                    self.imageLabel.adjustSize()

                if self.screensize[p] < image_size[p]:
                    self.imageLabel.resize((self.screensize[p] / image_size[p] - 0.04) * self.imageLabel.pixmap().size())

            except():
                print('No Image')

            if image.isNull():
                QMessageBox.information(self, "Bayesian Analayzer",
                        "Cannot load %s." % self.fileName)
                return

    def view(self, name, fold):
        if fold == 'result':
            print(self.res_path + name + '.png')
            view_path = self.res_path + name + '.png'
        elif fold == 'th':
            view_path = self.path_work + 'thumbnail/th.png'
        else:
            view_path = self.res_path + 'uncertainty/' + name + '.png'
        print(view_path)
        image = QImage(view_path)
        self.imageLabel.setPixmap(QPixmap.fromImage(image))

        image_size = [image.width(), image.height()]

        if image_size[0] > image_size[1]:
            p = 0
        else:
            p = 1
        print(image_size)
        print(self.screensize[0], self.screensize[1])

        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()

        if self.screensize[p] < image_size[p]:
            self.imageLabel.resize((self.screensize[p] / image_size[p]- 0.04) * self.imageLabel.pixmap().size())

    def progress(self, title):
        """Show the progress bar"""

        self.pop = QDialog()
        self.ui = Actions()
        self.ui.initUI(self.pop, title)
        self.pop.show()

    def print_(self):
        """Print the image visualized"""

        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def first_step(self, filename):

        if self.slowAct.isChecked():
            obj_an = StartAnalysis(filename, lev_sec=self.lev_sec)
        else:
            obj_an = StartAnalysis(filename)

        ph = obj_an.get_thumb()
        self.numx_start, self.numx_stop, self.list_proc, self.start_i, self.stop_i, self.ny, self.levi = obj_an.tile_gen(state=0)
        print(self.numx_start, self.numx_stop, self.list_proc, self.start_i, self.stop_i, self.ny, self.levi)
        return ph

    def progress_fn(self, n):
        print("%d%% done" % n)

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def thread_cl_complete(self):
        self.pop.hide()
        print("THREAD COMPLETE!")
        self.view('Pred_class', 'result')
        print("THREAD COMPLETE!")

    def process_to_start(self, vet, progress_callback):
        n_start, n_stop, name_process, start, stop, ny, levi = vet
        print(n_start, n_stop, name_process, start, stop, ny, levi)

        if self.slowAct.isChecked():
            obj_k = StartAnalysis(self.fileName, lev_sec=self.lev_sec)
        else:
            obj_k = StartAnalysis(self.fileName)

        res = obj_k.tile_gen(state=1)
        f_manager = self.folder_manage(name_process)
        flag = False

        if start == 1:
            flag = True

        if not f_manager:
            create_fold = str(self.path_work) + str(name_process)
            os.mkdir(create_fold)

            for x in range(n_start, n_stop):
                for y in range(0, ny):
                    im = res.get_tile(levi, (x, y))
                    nome = create_fold + '/tile_' + str(start) + '_' + str(x) + '_' + str(y) + '.png'
                    im.save(nome, 'PNG')
                    start += 1
                    if flag:
                        progress_callback.emit(100*(start-1)/stop)
                        if (start-1) == stop:
                            time.sleep(1)
                            self.pop.hide()

            return 'End of First Analysis'
        else:
            self.pop.hide()
            return 'End of First Analysis, exit code 1, the folder already exist!'

    def folder_manage(self, name_process):
        """Test if the folder alredy exist, if true return 1 and the thread will stop"""

        fold = os.listdir(self.path_work)
        flag = 0
        for k in fold:
            if k == name_process:
                print('Folder alredy exist {}'.format(name_process))
                flag += 1
            else:
                pass

        if flag > 0:
            return True
        else:
            return False

    def thread_manager(self):
        """Here are created the threads that create in the specific folders the tile"""
        if os.listdir(self.path_work)[0] == self.list_proc[0]:
            pass
        else:
            self.progress(title='Tiles creation process')
            for lp in range(0, len(self.list_proc)):
                vet = [self.numx_start[lp], self.numx_stop[lp], self.list_proc[lp], self.start_i[lp], self.stop_i[lp], self.ny, self.levi]
                lp = WorkerLong(self.process_to_start, vet)
                if lp == 0:
                    lp.signals.result.connect(self.print_output)
                    lp.signals.progress.connect(self.progress_fn)
                    lp.signals.progress.connect(self.ui.onCountChanged)
                lp.signals.finished.connect(self.thread_complete)
                self.threadPool.start(lp)

    def zoomIn(self):
        self.scaleImage(1.2)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)

        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def fast(self):
        self.type_an = 'fast'
        self.slowAct.setChecked(False)

    def slow(self):
        self.type_an = 'slow'
        self.fastAct.setChecked(False)
        self.open()

    def deep_vis(self):
        """This method allows to view the svs image at maximum resolution using the browser and a javascript library;
        it's performed before starting the server a test on the paths in search of spaces, indeed the program does not
        works if in the path there are with spaces"""

        b = self.fileName
        if b.find(' ') != -1:
            print('there is a space in the path', b.find(' '))
            QMessageBox.critical(self, "About Image Viewer",
                              "There are same space in the path: \n\n {} \n\n"
                              "The deepzoom function need no space in the path.\n"
                              "Please, rename the folder without space and star again the program.".format(b))
        else:
            command = 'cmd /k python deepzoom/deepzoom_server.py {}'.format(b)
            worker = Worker(self.cmd_command, command)
            self.threadPool.start(worker)

            QMessageBox.information(self, "Bayesian Analayzer",
                                    "Now your browser will be opened ad a deep zoomable file\n"
                                    "with a lot of information will be showed. \n \n "
                                    "Press ok to continue")

            worker2 = Worker(self.open_b)
            self.threadPool.start(worker2)

    def open_b(self):
        """Open a new tab in the browser at a specific port"""

        url = "http://127.0.0.1:5000/"
        time.sleep(0.5)
        webbrowser.open_new_tab(url)

    def cmd_command(self, command):
        os.system(command)

    def start_an(self):

        if not os.path.exists(self.res_path):
            cls = Classification(self.path_work)
            self.progress(title='Analysis')
            worker_cl = WorkerLong(cls.classify, self.type_an, self.monte_c, self.model_name)
            worker_cl.signals.progress.connect(self.progress_fn)
            worker_cl.signals.progress.connect(self.ui.onCountChanged)
            worker_cl.signals.finished.connect(self.thread_cl_complete)
            self.threadPool.start(worker_cl)
        else:
            self.view('Pred_class', 'result')
            print('process an already done!!!')

        self.v_no_overlayAct.setEnabled(True)
        self.v_all_classAct.setEnabled(True)

        self.v_acAct.setEnabled(True)
        self.v_adAct.setEnabled(True)
        self.v_hAct.setEnabled(True)

        self.v_tot_uAct.setEnabled(True)
        self.v_a_uAct.setEnabled(True)
        self.v_e_uAct.setEnabled(True)

    def v_no_overlay(self):
        self.view('no_ov', 'th')

    def v_all_class(self):
        self.view('Pred_class', 'result')

    def v_ac(self):
        self.view('AC', 'result')

    def v_ad(self):
        self.view('AD', 'result')

    def v_h(self):
        self.view('H', 'result')

    def v_tot_u(self):
        self.view('tot', 'uncertainty')

    def v_a_u(self):
        self.view('ale', 'uncertainty')

    def v_e_u(self):
        self.view('epi', 'uncertainty')

    def select_model(self):
        QMessageBox.information(self, "Deepzoom Viewer",
                             "<p>The model must have tree class for output, AC, AD, H.</p>")
        self.model_name, _ = QFileDialog.getOpenFileName(self, "Open File", "*.h5")

    def five(self):
        self.tfiveAct.setChecked(False)
        self.fiftyAct.setChecked(False)
        self.monte_c = 5

    def tfive(self):
        self.fiveAct.setChecked(False)
        self.fiftyAct.setChecked(False)
        self.monte_c = 25

    def fifty(self):
        self.fiveAct.setChecked(False)
        self.tfiveAct.setChecked(False)
        self.monte_c = 50

    def about(self):

        QMessageBox.about(self, "About Image Viewer",
                          "<p>The <b>Bayesian Analayzer</b> is built for analayze "
                          "Svs file, that tipicaly are very heavy files (~1Gb) "
                          "whit a machine learning net. </p>"
                          "<p>In addition, the program allows to zoom in at maximum resolution "
                          "thanks to your browser.</p>")

    def info_deep(self):
        QMessageBox.about(self, "Deepzoom Viewer",
                          "<p>The <b>Deepzoom</b> function allows the user to zoom in "
                          "Svs file, at the maximum of the resolution.</p>"
                          "<p>The program will open a tab in your browser, where you can "
                          "zoom in to the image at maximum resolution.</p>"
                          "<p>In addition, at the right side, are also shows some info about the svs file "
                          "that the user is analyzing.</p>")

    def createActions(self):
        self.openAct = QAction(QIcon('icons/folder.png'), "Select svs", self, shortcut="Ctrl+O",
                triggered=self.open)

        self.printAct = QAction("&Print...", self, shortcut="Ctrl+P",
                enabled=False, triggered=self.print_)

        self.exitAct = QAction(QIcon('icons/exit.ico'), "E&xit", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.zoomInAct = QAction(QIcon('icons/zoomin.ico'), "Zoom &In (25%)", self, shortcut="Ctrl++",
                enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QAction(QIcon('icons/zoomout.ico'),"Zoom &Out (25%)", self, shortcut="Ctrl+-",
                enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S",
                enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False,
                checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)

        self.aboutAct = QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self, triggered=QApplication.instance().aboutQt)

        self.info_deepAct = QAction("&Info about deep viewer", self, triggered=self.info_deep)

        self.v_no_overlayAct = QAction("View whit no overlay", self, enabled=False, triggered=self.v_no_overlay)
        self.v_all_classAct = QAction("View all classes", self, enabled=False, triggered=self.v_all_class)

        self.v_acAct = QAction(QIcon('icons/AC.ico'), "View only AC", self, enabled=False, triggered=self.v_ac)
        self.v_adAct = QAction(QIcon('icons/AD.ico'), "View only AD", self, enabled=False, triggered=self.v_ad)
        self.v_hAct = QAction(QIcon('icons/H.ico'), "View only H", self, enabled=False, triggered=self.v_h)

        self.v_tot_uAct = QAction("View total uncertainty", self, enabled=False, triggered=self.v_tot_u)
        self.v_a_uAct = QAction("View only Aleatoric uncertainty", self, enabled=False, triggered=self.v_a_u)
        self.v_e_uAct = QAction("View only Epistemic uncertainty", self, enabled=False, triggered=self.v_e_u)

        self.start_vis_deepAct = QAction(QIcon('icons/binocul.ico'), "Go to deepzoom visualization", self, enabled=False,
                                         shortcut="Ctrl+S", triggered=self.deep_vis)

        self.startAnalysisAct = QAction(QIcon('icons/start.ico'), 'Start', self, triggered=self.start_an, enabled=False, shortcut="Ctrl+S")

        self.fastAct = QAction('Fast mode', enabled=True, checkable=True, checked=True,  triggered=self.fast)

        self.slowAct = QAction('Slow mode', enabled=True, checkable=True, triggered=self.slow)

        self.select_modAct = QAction('Change model', enabled=True, triggered=self.select_model)

        self.fiveAct = QAction('5', enabled=True, checkable=True,checked=True, triggered=self.five)
        self.tfiveAct = QAction('25', enabled=True, checkable=True, triggered=self.tfive)
        self.fiftyAct = QAction('50', enabled=True, checkable=True, triggered=self.fifty)


    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.Analyze = QMenu("&Analysis", self)
        self.Analyze.addAction(self.fastAct)
        self.Analyze.addAction(self.slowAct)
        self.Analyze.addSeparator()

        self.modi = QMenu("&Modify Settings", self)
        self.modi.addAction(self.select_modAct)

        self.monte_n = QMenu("&Select Monte Carlo Sample", self)
        self.monte_n.addAction(self.fiveAct)
        self.monte_n.addAction(self.tfiveAct)
        self.monte_n.addAction(self.fiftyAct)

        self.modi.addAction(self.monte_n.menuAction())

        self.Analyze.addAction(self.modi.menuAction())

        self.Analyze.addSeparator()
        self.Analyze.addAction(self.startAnalysisAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.v_no_overlayAct)
        self.viewMenu.addAction(self.v_all_classAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.v_acAct)
        self.viewMenu.addAction(self.v_adAct)
        self.viewMenu.addAction(self.v_hAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.v_tot_uAct)
        self.viewMenu.addAction(self.v_a_uAct)
        self.viewMenu.addAction(self.v_e_uAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.options = QMenu("&Options", self)
        self.options.addAction(self.start_vis_deepAct)
        self.options.addAction(self.info_deepAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        # TOOLBAR
        self.toolbar.addAction(self.openAct)
        self.toolbar.addAction(self.startAnalysisAct)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.zoomInAct)
        self.toolbar.addAction(self.zoomOutAct)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.start_vis_deepAct)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.v_acAct)
        self.toolbar.addAction(self.v_adAct)
        self.toolbar.addAction(self.v_hAct)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.exitAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.Analyze)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.options)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())

        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 4.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.2)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())