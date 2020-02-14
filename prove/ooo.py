from PyQt5.QtWidgets import (QWidget, QMainWindow, QHBoxLayout, QFrame, QSizePolicy,
                             QSplitter, QStyleFactory, QApplication, QLabel, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QFont, QIcon

import sys


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        image = QImage('C:/Users/piero/Test/31400_2/thumbnail/th.png')
        self.imageLabel = QLabel("Steps to start the analysis:\n \n"
                                 "1) File         ---> Select svs or select the yellow folder in the toolbar\n\n"
                                 "2) Analysis ---> Stat analysis or select the green arrow in the toolbar ")
        self.imageLabel.setFont(QFont("Akzidenz Grotesk", 15, QFont.Black))

        self.imageLabel2 = QLabel()
        self.imageLabel2.setPixmap(QPixmap.fromImage(image))
        self.imageLabel.setBackgroundRole(QPalette.Dark)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        hbox = QHBoxLayout()
        scroll = QScrollArea()
        scroll.setWidget(self.imageLabel)
        s1 = QScrollArea()
        scroll.setWidget(self.imageLabel2)
        hbox.addWidget(scroll)
        hbox.addWidget(s1)
        self.setLayout(hbox)


        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('QSplitter')
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())