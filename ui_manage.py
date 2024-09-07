import os.path
from geocal_calibration import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidget, QMessageBox
import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import res_rc
from first_centroid import *
from exposure_fusion import *
from centroid import *
from grid_generation import *
from calibrate import *
from test import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.ui.pushButton_4.clicked.connect(self.go_detect)
        self.ui.pushButton_5.clicked.connect(self.go_calibrate)
        self.ui.pushButton_upload.clicked.connect(self.upload_img)
        self.ui.pushButton_change.clicked.connect(self.change_path)
        self.ui.pushButton_detect.clicked.connect(self.grid_detection)
        # self.ui.progressBar_detect.setVisible(False)
        self.ui.pushButton_7.clicked.connect(self.result_image)
        self.ui.pushButton_8.clicked.connect(self.result_coords)
        self.ui.pushButton_13.clicked.connect(self.choose_calibrate)
        self.ui.pushButton_9.clicked.connect(self.change_path2)
        self.ui.pushButton_calibrate.clicked.connect(self.calibration)
        self.ui.pushButton_12.clicked.connect(self.result_paras)
        self.ui.pushButton_11.clicked.connect(self.test_effect)
        self.ui.pushButton_14.clicked.connect(self.result_effect)

        self.ui.pushButton_7.setVisible(False)
        self.ui.pushButton_8.setVisible(False)
        self.ui.pushButton_12.setVisible(False)
        self.ui.pushButton_11.setVisible(False)
        self.ui.pushButton_14.setVisible(False)

        self.show()

    def go_calibrate(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def go_detect(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    def go_test(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def upload_img(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.bmp);;All Files (*)")

        if files:
            self.ui.listWidget_img.clear()
            self.ui.listWidget_img.addItems(files)
            # self.ui.listWidget_img.show()
            self.ui.lineEdit_output.setText(os.path.dirname(os.path.abspath(__file__)))

    def change_path(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.ui.lineEdit_output.setText(folder_path)

    def grid_detection(self):
        # self.ui.progressBar_detect.setVisible(True)
        images = []  # 将要处理的图像的路径
        for i in range(self.ui.listWidget_img.count()):
            item = self.ui.listWidget_img.item(i)
            images.append(item.text())

        first_centroid(images)

        output_folder = self.ui.lineEdit_output.text()
        if output_folder and images:
            exposure_fusion(output_folder)

            centroid(output_folder)

            grid_generation(output_folder)

            self.ui.pushButton_7.setVisible(True)
            self.ui.pushButton_8.setVisible(True)
        # self.ui.progressBar_detect.setVisible(False)

    def result_image(self):
        output_folder = self.ui.lineEdit_output.text()
        if output_folder:
            output_folder = os.path.join(output_folder, "output_image_with_coordinates.png")
            os.startfile(output_folder)

    def result_coords(self):
        output_folder = self.ui.lineEdit_output.text()
        if output_folder:
            output_folder = os.path.join(output_folder, "final_coordinates.txt")
            os.startfile(output_folder)

    def choose_calibrate(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select files", "", "Text Files (*.txt);;All Files (*)", options=options)

        if file_path:
            filename = "final_coordinates.txt"
            selected_filename = file_path.split('/')[-1]
            if selected_filename == filename:
                self.ui.lineEdit_2.setText(file_path)
            else:
                QMessageBox.warning(self, "Illegal file", f"Please choose file named {filename}")

    def change_path2(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.ui.lineEdit.setText(folder_path)

    def calibration(self):
        output_folder = self.ui.lineEdit.text()  # 输出结果的路径
        input_file = self.ui.lineEdit_2.text()  # 输入坐标的路径

        if output_folder and input_file:
            calibrate(output_folder, input_file)

            self.ui.pushButton_12.setVisible(True)
            self.ui.pushButton_11.setVisible(True)

    def result_paras(self):
        output_folder = self.ui.lineEdit.text()
        if output_folder:
            output_folder = os.path.join(output_folder, "calibration result.txt")
            os.startfile(output_folder)

    def test_effect(self):
        output_folder = self.ui.lineEdit.text()
        input_file = self.ui.lineEdit_2.text()

        if output_folder and input_file:
            test(output_folder, input_file)

            self.ui.pushButton_14.setVisible(True)

    def result_effect(self):
        output_folder = self.ui.lineEdit.text()
        if output_folder:
            output_folder = os.path.join(output_folder, "calibration_effect.bmp")
            os.startfile(output_folder)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    sys.exit(app.exec_())
