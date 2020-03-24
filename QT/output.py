# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1341, 949)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(890, 850, 401, 61))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(0, 30, 1341, 101))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(90, 200, 171, 51))
        self.label_2.setObjectName("label_2")
        self.listWidget_2 = QtWidgets.QListWidget(Dialog)
        self.listWidget_2.setGeometry(QtCore.QRect(260, 210, 256, 192))
        self.listWidget_2.setObjectName("listWidget_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(100, 480, 231, 33))
        self.label_3.setObjectName("label_3")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(320, 480, 201, 39))
        self.lineEdit.setObjectName("lineEdit")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(100, 580, 114, 33))
        self.label_4.setObjectName("label_4")
        self.radioButton = QtWidgets.QRadioButton(Dialog)
        self.radioButton.setGeometry(QtCore.QRect(90, 640, 197, 37))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(Dialog)
        self.radioButton_2.setGeometry(QtCore.QRect(90, 700, 197, 37))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(Dialog)
        self.radioButton_3.setGeometry(QtCore.QRect(90, 760, 197, 37))
        self.radioButton_3.setObjectName("radioButton_3")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Stock Predictor using Machine Learning"))
        self.label_2.setText(_translate("Dialog", "Chose Stock:"))
        self.label_3.setText(_translate("Dialog", "Principal Amount: "))
        self.label_4.setText(_translate("Dialog", "Strategy:"))
        self.radioButton.setText(_translate("Dialog", "Conservative"))
        self.radioButton_2.setText(_translate("Dialog", "Balanced"))
        self.radioButton_3.setText(_translate("Dialog", "Risky"))

