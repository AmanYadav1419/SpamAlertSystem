import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QTextEdit, QVBoxLayout

class SpamDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Spam Detector')

        # Create widgets
        self.text_edit = QTextEdit()
        self.classify_button = QPushButton('Classify')
        self.result_label = QLabel()
        self.accuracy_label = QLabel()
        self.precision_label = QLabel()

        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Enter message:'))
        layout.addWidget(self.text_edit)
        layout.addWidget(self.classify_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.accuracy_label)
        layout.addWidget(self.precision_label)
        self.setLayout(layout)

        # Connect button to function
        self.classify_button.clicked.connect(self.classify_message)

        self.show()

    def classify_message(self):
        # Load the data
        data = pd.read_csv('spam_data.csv')

        # Preprocess the data
        data['text'] = data['text'].apply(lambda x: x.lower())
        data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))

        # Vectorize the text
        vectorizer = CountVectorizer()
        X_vectorized = vectorizer.fit_transform(data['text'])
        y = data['label']

        # Train a Naive Bayes model
        model = MultinomialNB()
        model.fit(X_vectorized, y)

        # Classify the input message
        input_text = self.text_edit.toPlainText()
        input_text = input_text.lower()
        input_text = re.sub('[^a-zA-Z0-9\s]', '', input_text)
        input_vectorized = vectorizer.transform([input_text])
        result = model.predict(input_vectorized)

        # Display the result
        if result == 'spam':
            self.result_label.setText('SPAM')
        else:
            self.result_label.setText('NOT SPAM')

        # Calculate performance metrics
        accuracy = model.score(X_vectorized, y)
        self.accuracy_label.setText('Accuracy: {:.2f}'.format(accuracy))

        precision = precision_score(y, model.predict(X_vectorized), pos_label='spam')
        self.precision_label.setText('Precision: {:.2f}'.format(precision))

if __name__ == '__main__':
    app = QApplication([])
    spam_detector = SpamDetector()
    app.exec_()
