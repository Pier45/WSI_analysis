import json
import sys

import numpy as np
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from sklearn.metrics import confusion_matrix

from src.config import CLASS_NAMES


class QHLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class PerformanceTab(QWidget):

    def __init__(self, path='archive/dictionary_5_js.txt', parent=None):
        super(QWidget, self).__init__(parent)
        self.tableWidget = QTableWidget()
        self.tabSingle = QTableWidget()
        self.path = path
        self.vet_p = ['', '', '']
        self.initUI()

    def initUI(self):
        newfont = QFont("Helvetica", 15, QFont.Bold)
        self.title_test = QLabel("Confusion matrix EXAMPLE")
        self.title_paz = QLabel("List of patient, double click to see the patient confusion matrix.")
        self.title_single_paz = QLabel("Confusion matrix of a single patient")
        self.title_selection = QLabel("Select the data set to show the confusion matrix:")

        self.title_paz.setFont(newfont)
        self.title_test.setFont(newfont)
        self.title_single_paz.setFont(newfont)
        self.traincm = QRadioButton('Train')
        self.traincm.setEnabled(False)
        self.traincm.toggled.connect(self.cm_train)
        self.valcm = QRadioButton('Validation')
        self.valcm.setEnabled(False)
        self.valcm.toggled.connect(self.cm_val)
        self.testcm = QRadioButton('Test')
        self.testcm.setEnabled(False)
        self.testcm.toggled.connect(self.cm_test)
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.traincm)
        self.mode_group.addButton(self.valcm)
        self.mode_group.addButton(self.testcm)

        self.h_sel = QHBoxLayout()
        self.h_sel.addWidget(self.traincm)
        self.h_sel.addWidget(self.valcm)
        self.h_sel.addWidget(self.testcm)
        self.h_sel.addStretch(1)

        cm = self.get_data(self.path)

        self.createTable(cm)
        self.createTable_sigle(cm)

        self.v_list = QVBoxLayout()
        self.v_list.addWidget(self.title_paz)
        self.v_list.addWidget(self.list_pat)

        self.v_tab = QVBoxLayout()
        self.v_tab.addWidget(self.title_single_paz)
        self.v_tab.addWidget(self.tabSingle)

        self.h = QHBoxLayout()
        self.h.addLayout(self.v_list)
        self.h.addLayout(self.v_tab)

        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.title_selection)
        self.layout.addLayout(self.h_sel)
        self.layout.addWidget(QHLine())
        self.layout.addWidget(self.title_test)
        self.layout.addWidget(self.tableWidget)
        self.layout.addLayout(self.h)
        self.setLayout(self.layout)

        # Show widget
        self.show()

    def get_paths(self, train=None, val=None, test=None):
        if train is not None:
            self.vet_p[0] = train
        elif val is not None:
            self.vet_p[1] = val
        elif test is not None:
            self.vet_p[2] = test
        else:
            pass

    def cm_train(self):
        self.path = self.vet_p[0]
        cm = self.get_data(self.path)
        self.title_test.setText('Confusion matrix TRAIN')
        self.createTable(cm)
        self.createTable_sigle(cm)

    def cm_val(self):
        self.path = self.vet_p[1]
        cm = self.get_data(self.path)
        self.title_test.setText('Confusion matrix VAL')
        self.createTable(cm)
        self.createTable_sigle(cm)

    def cm_test(self):
        self.path = self.vet_p[2]
        cm = self.get_data(self.path)
        self.title_test.setText('Confusion matrix TEST')
        self.createTable(cm)
        self.createTable_sigle(cm)

    def w_list_paz(self, dizio):
        vet_paz = [dizio[m]['name'] for m in dizio]
        print('voci diz', len(vet_paz))
        vet_paz = list(set(vet_paz))

        self.list_pat = QListWidget()
        self.list_pat.addItems(vet_paz)
        self.list_pat.itemDoubleClicked.connect(self.sasa)

    def sasa(self, item):
        self.paz = item.text()
        print(item.text())
        cm = self.get_data_sigle_sub(self.path, self.paz)
        self.createTable_sigle(cm)

    def load_js(self, path):

        with open(path) as myfile:
            openf = myfile.read()
        dizio = json.loads(openf)

        return dizio

    def get_data(self, path):
        dizio = self.load_js(path)

        self.w_list_paz(dizio)
        pred_class = [dizio[m]['pred_class'] for m in dizio]
        true_class = [dizio[m]['true_class'] for m in dizio]

        cm = np.round(100*confusion_matrix(true_class, pred_class, labels=list(CLASS_NAMES))/len(true_class), 2)
        return cm

    def get_data_sigle_sub(self, path, pat):
        dizio = self.load_js(path)

        pred_class = [dizio[m]['pred_class'] for m in dizio if dizio[m]['name'] == pat]
        true_class = [dizio[m]['true_class'] for m in dizio if dizio[m]['name'] == pat]

        cm = np.round(100*confusion_matrix(true_class, pred_class, labels=list(CLASS_NAMES))/len(true_class), 2)
        return cm

    def createTable_sigle(self, cm):
        """Populate the single-patient confusion-matrix table."""
        self._fill_confusion_table(self.tabSingle, cm)

    def createTable(self, cm):
        """Populate the aggregate confusion-matrix table."""
        self._fill_confusion_table(self.tableWidget, cm)

    @staticmethod
    def _fill_confusion_table(table: QTableWidget, cm: np.ndarray) -> None:
        """
        Fill a 3x3 :class:`QTableWidget` with the confusion-matrix percentages
        and label its rows/columns with the canonical :data:`CLASS_NAMES`.

        Both :meth:`createTable` (aggregate) and :meth:`createTable_sigle`
        (single-patient) tables are 3x3 with identical formatting, so this
        helper takes the target widget as a parameter instead of duplicating
        the body.
        """
        table.setRowCount(3)
        table.setColumnCount(3)
        for r in range(3):
            for c in range(3):
                table.setItem(r, c, QTableWidgetItem(str(cm[r, c]) + '%'))
        labels = list(CLASS_NAMES)
        table.setHorizontalHeaderLabels(labels)
        table.setVerticalHeaderLabels(labels)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PerformanceTab()
    sys.exit(app.exec_())

