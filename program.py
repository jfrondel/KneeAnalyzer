import sys
from PyQt5.QtWidgets import QApplication
from Components.qt_utils import main_window

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = main_window()
    sys.exit(app.exec_())