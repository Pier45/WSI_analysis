import sys
from PyQt5.QtWidgets import QApplication, QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QLabel, QListWidget, QHBoxLayout
from PyQt5.QtCore import pyqtSlot
import json
import numpy as np
from sklearn.metrics import confusion_matrix


class TestTab(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self.tableWidget = QTableWidget()
        self.tabSingle = QTableWidget()
        self.initUI()

    def initUI(self):
        self.title_test = QLabel("Confution matrix of the Test set")
        self.title_paz = QLabel("List of patient, doble click to see the patient confution matrix.")
        self.title_single_paz = QLabel("Confution matrix of a sigle patient")

        cm = self.get_data('C:/Users/piero/yy/train/dictionary_5_js.txt')

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
        self.layout.addWidget(self.title_test)
        self.layout.addWidget(self.tableWidget)
        self.layout.addLayout(self.h)
        self.setLayout(self.layout)

        # Show widget
        self.show()

    def w_list_paz(self, dizio):
        vet_paz = [dizio[m]['name'] for m in dizio]
        print(len(vet_paz))
        vet_paz = list(set(vet_paz))

        self.list_pat = QListWidget()
        self.list_pat.addItems(vet_paz)
        self.list_pat.itemDoubleClicked.connect(self.sasa)

    def fake_data(self):
        pred_class = ['AC', 'AC', 'AC', 'AD', 'AC', 'AD', 'H', 'H', 'AD', 'AC', 'AC', 'H', 'H', 'H', 'AC', 'AC', 'AC',
                      'AC', 'AD', 'AC', 'AD', 'H', 'H', 'AD', 'AC', 'AC', 'H', 'H', 'H', 'AC']
        true_class = ['AC', 'AC', 'AC', 'AD', 'AC', 'AC', 'H', 'H', 'H', 'AC', 'AC', 'AC', 'AD', 'AD', 'AD', 'AC', 'AC',
                      'AC', 'AD', 'AC', 'AD', 'H', 'H', 'AD', 'AC', 'AC', 'H', 'H', 'H', 'AC']

        cm = confusion_matrix(true_class, pred_class, labels=['AC', 'H', 'AD'])
        return cm

    def sasa(self, item):
        self.paz = item.text()
        print(item.text())
        cm = self.get_data_sigle_sub('C:/Users/piero/yy/train/dictionary_5_js.txt', self.paz)
        self.createTable_sigle(cm)

    def load_js(self, path):
        with open(path, 'r') as myfile:
            openf = myfile.read()
        dizio = json.loads(openf)
        return dizio

    def get_data(self, path):
        dizio = self.load_js(path)
        self.w_list_paz(dizio)
        pred_class = [dizio[m]['pred_class'] for m in dizio]
        true_class = [dizio[m]['true_class'] for m in dizio]

        cm = confusion_matrix(true_class, pred_class, labels=['AC', 'H', 'AD'])
        return cm

    def get_data_sigle_sub(self, path, pat):
        dizio = self.load_js(path)

        pred_class = [dizio[m]['pred_class'] for m in dizio if dizio[m]['name'] == pat]
        true_class = [dizio[m]['true_class'] for m in dizio if dizio[m]['name'] == pat]

        cm = confusion_matrix(true_class, pred_class, labels=['AC', 'H', 'AD'])
        return cm

    def createTable_sigle(self, cm):
        # Create table
        self.tabSingle.setRowCount(3)
        self.tabSingle.setColumnCount(3)
        self.tabSingle.setItem(0, 0, QTableWidgetItem(str(cm[0,0])))
        self.tabSingle.setItem(0, 1, QTableWidgetItem(str(cm[0, 1])))
        self.tabSingle.setItem(0, 2, QTableWidgetItem(str(cm[0,2])))

        self.tabSingle.setItem(1, 0, QTableWidgetItem(str(cm[1,0])))
        self.tabSingle.setItem(1, 1, QTableWidgetItem(str(cm[1,1])))
        self.tabSingle.setItem(1, 2, QTableWidgetItem(str(cm[1,2])))

        self.tabSingle.setItem(2, 0, QTableWidgetItem(str(cm[2,0])))
        self.tabSingle.setItem(2, 1, QTableWidgetItem(str(cm[2,1])))
        self.tabSingle.setItem(2, 2, QTableWidgetItem(str(cm[2,2])))
        a = ['AC', 'H', 'AD']
        self.tabSingle.setHorizontalHeaderLabels(a)
        self.tabSingle.setVerticalHeaderLabels(a)


    def createTable(self, cm):
        # Create table
        #cm = self.fake_data()
        self.tableWidget.setRowCount(3)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setItem(0, 0, QTableWidgetItem(str(cm[0,0])))
        self.tableWidget.setItem(0, 1, QTableWidgetItem(str(cm[0, 1])))
        self.tableWidget.setItem(0, 2, QTableWidgetItem(str(cm[0,2])))

        self.tableWidget.setItem(1, 0, QTableWidgetItem(str(cm[1,0])))
        self.tableWidget.setItem(1, 1, QTableWidgetItem(str(cm[1,1])))
        self.tableWidget.setItem(1, 2, QTableWidgetItem(str(cm[1,2])))

        self.tableWidget.setItem(2, 0, QTableWidgetItem(str(cm[2,0])))
        self.tableWidget.setItem(2, 1, QTableWidgetItem(str(cm[2,1])))
        self.tableWidget.setItem(2, 2, QTableWidgetItem(str(cm[2,2])))
        a = ['AC', 'H', 'AD']
        self.tableWidget.setHorizontalHeaderLabels(a)
        self.tableWidget.setVerticalHeaderLabels(a)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TestTab()
    sys.exit(app.exec_())