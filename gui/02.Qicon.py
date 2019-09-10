import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon

class MyApp(QWidget):
  
  def __init__(self):
    super().__init__()
    self.initUI()
    
  def initUI(self):
    self.setWindowTitle('My First Application')
    self.setWindowIcon(QIcon('icon.jpeg'))
    self.setGeometry(300,300,300,200)
    self.show()

if __name__ == '__main__':
  app = QApplication(sys.argv)
  # 모든 PyQt5 어플리케이션은 어플리케이션 객체를 생성합니다.
  ex = MyApp()
  sys.exit(app.exec_())