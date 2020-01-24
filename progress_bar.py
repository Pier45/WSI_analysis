from PyQt5.QtWidgets import (QApplication, QDialog, QProgressBar, QPushButton)
import sys


class Actions:

    def initUI(self, owindow, title):
        owindow.setWindowTitle(title)
        self.progress = QProgressBar(owindow)
        self.progress.setGeometry(0, 0, 300, 50)
        self.progress.setMaximum(100)

    def onCountChanged(self, value):
        self.progress.setValue(value)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Actions()
    sys.exit(app.exec_())