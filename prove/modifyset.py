import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QWidget,
                             QFileDialog, QHBoxLayout, QGridLayout, QFrame, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import QDir


class Mod:

    def ax(self, parent=None):
        #super().__init__(parent)
        parent.setGeometry(50, 70, 320, 250)
        parent.setWindowTitle("Modify settings")
        self.model_name = 'default'
        self.monte_carlo = 5
        parent.setAutoFillBackground(1)
        layout = QGridLayout()
        oriz = QHBoxLayout()

        combo = QComboBox()
        combo.addItem("5")
        combo.addItem("10")
        combo.addItem("15")
        combo.addItem("20")
        combo.addItem("50")
        combo.move(150, 20)

        self.qlabel1 = QLabel('Monte Carlo sample:')

        self.l_model = QLabel('Select another model:')

        self.button_sel = QPushButton('Select model .h5')
        self.button_sel.clicked.connect(self.newModel)
        self.button_close = QPushButton('Ok')
        self.button_close.clicked.connect(self.ok)
        self.button_close.setToolTip('Press ok to continue!')

        self.qlabel = QLabel('Default value: 5')

        combo.activated[str].connect(self.onChanged)

        oriz.addItem(QSpacerItem(50,10, QSizePolicy.Expanding))
        oriz.addWidget(self.button_close)

        layout.addWidget(self.qlabel1)
        layout.addWidget(combo)
        layout.addWidget(self.qlabel)
        layout.addWidget(QHLine())
        layout.addWidget(self.l_model)
        layout.addWidget(self.button_sel)
        layout.addWidget(QHLine())
        layout.addItem(oriz)

        parent.setLayout(layout)
        self.show()

    def onChanged(self, mc):
        self.qlabel.setText('Selected:  {}'.format(mc))
        self.qlabel.adjustSize()
        self.monte_carlo = mc

    def newModel(self, parent):
        self.model_name, _ = QFileDialog.getOpenFileName(parent, "Open File", QDir.currentPath())
        print(self.model_name)

    def ok(self, parent):
        parent.hide()
        return self.model_name, self.monte_carlo


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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Mod()
    sys.exit(app.exec_())