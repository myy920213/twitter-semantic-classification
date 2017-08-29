import re
import xlrd
import logging

import getopt, sys, logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


from nltk import *

############################## data preprocess start here###########################

class Data_preprocessing:



    def __init__(self, data_file_path):
        self.DATA = {}
        self.DATA_FILE_PATH = data_file_path
        self.readin_data = xlrd.open_workbook(self.DATA_FILE_PATH)

    def load_excel_data(self, sheet_name):
        sheet = self.readin_data.sheet_by_name(sheet_name)

        data_row = {}

        for i in range(sheet.nrows):
            try:
                tweet_text = sheet.row_values(i)[3]
                tweet_text = re.sub('<[^<]+?>', '', tweet_text) #remove html tags
                tweet_text = re.sub('[a-zA-z]+://[^\s]*', '', tweet_text)  # remove link
                tweet_text = re.sub('@\w+', '', tweet_text)  # remove @

                if tweet_text == '':
                    continue

                class_label = sheet.row_values(i)[4]

                if isinstance(class_label, str) and class_label not in ['0', '-1', '1']:
                    continue

                if int(class_label) == 2:
                    continue

                data_row[tweet_text.lower()] = int(class_label)

            except:

                pass
        #print data_row
        logging.info('data size:' + str(len(data_row.items())))

        self.DATA['text'] = data_row.keys()
        self.DATA['target'] = data_row.values()
        self.DATA['size'] = len(data_row.items())
        #print self.DATA

        return self.DATA


##################################classifier start here###########################


class TweetClassifier:


    def __init__(self):

        self.classifiers = []
        self.classifiers.append('MultinomialNB')
        self.classifiers.append('SVM_linear')
        self.classifiers.append('SVM_kernal')
        self.classifiers.append('Logic Regression')
        self.classifiers.append('SGD')
        self.classifiers.append('RandomForest')
        self.classifiers.append('VotingClassifier')
        self.classifiers.append('AdaBoostClassifier')
        #self.classifiers.append('BaggingClassifier')
        #self.classifiers.append('DecisionTreeClassifier')


    def get_classifier(self, name):

        vect = CountVectorizer(analyzer="word", strip_accents='unicode', ngram_range=(1, 2), tokenizer=self.my_tokenizer,
                               stop_words=None, lowercase=True)

        clf_params = [('vect', vect)]
        if name == 'MultinomialNB':
            clf_params.append(('clf', MultinomialNB()))
        elif name == 'BernoulliNB':
            clf_params.append(('clf', BernoulliNB()))
        elif name == 'SVM_linear':
            clf_params.append(('clf', LinearSVC(loss='hinge')))
        elif name == 'SVM_kernal':
            clf_params.append(('clf', SVC(kernel='rbf', gamma=0.7, C=1)))
        elif name == 'Logic Regression':
            clf_params.append(('clf', LogisticRegression(random_state=1)))
        elif name == 'SGD':
            clf_params.append(('clf', SGDClassifier(loss="hinge", penalty="l2")))
        elif name == 'RandomForest':
            clf_params.append(('clf', RandomForestClassifier()))
        elif name == 'DecisionTreeClassifier':
            clf_params.append(('clf', DecisionTreeClassifier()))
        elif name == 'VotingClassifier':
            clf1 = LogisticRegression(random_state=1)
            clf2 = MultinomialNB()
            clf3 = SVC(kernel='rbf', gamma=0.7, C=1, probability=True)
            clf_params.append(('clf', VotingClassifier(estimators=[('lg', clf1), ('mNB', clf2), ('svm', clf3)], voting='soft', weights=[1.3, 0.8, 2])))
        elif name == 'AdaBoostClassifier':
            clf_params.append(('clf',AdaBoostClassifier(n_estimators=100)))
        elif name == 'BaggingClassifier':
            clf_params.append(('clf',BaggingClassifier(KNeighborsClassifier(),
                            max_samples=0.5, max_features=0.5)))
        text_clf = Pipeline(clf_params)

        return text_clf


    def crossvalidate(self, data, Nfold):

        text = data['text']
        target = data['target']


        assert(len(text) == len(target))

        classifier_report = {}

        for name in self.classifiers:

            kf = StratifiedKFold(n_splits=Nfold, shuffle=True,
                                 random_state=None)
            avg_accuracy = 0.0

            positive = [0.0, 0.0, 0.0]    #precision, recall, fscore
            negative = [0.0, 0.0, 0.0]

            for train_index, test_index in kf.split(text,target):
                # print train_index, test_index

                train_data = [text[i] for i in train_index]
                test_data = [text[i] for i in test_index]

                train_target = [target[i] for i in train_index]
                test_target = [target[i] for i in test_index]

                classifier = self.get_classifier(name)

                text_fit = classifier.fit(train_data, train_target)
                predicted = text_fit.predict(test_data)

                avg_accuracy += accuracy_score(test_target, predicted)


                confusionMatrix = confusion_matrix(test_target, predicted, labels=[-1, 0, 1])

                l = 0.000001
                positive[0] += (float(confusionMatrix[2][2]) / (confusionMatrix[2][2] + confusionMatrix[1][2] + confusionMatrix[0][2] + l))
                positive[1] += (float(confusionMatrix[2][2]) / (confusionMatrix[2][0] + confusionMatrix[2][1] + confusionMatrix[2][2] + l))
                positive[2] = (2 * positive[0] * positive[1]) / (positive[0] + positive[1] + l)

                negative[0] += (float(confusionMatrix[0][0]) / (confusionMatrix[0][0] + confusionMatrix[1][0] + confusionMatrix[2][0]) + l)
                negative[1] += (float(confusionMatrix[0][0]) / (confusionMatrix[0][0] + confusionMatrix[0][1] + confusionMatrix[0][2]) + l)
                negative[2] = (2 * negative[0] * negative[1]) / (negative[0] + negative[1] + l)


            report = {
                'precision' :   ((positive[0] / Nfold), (negative[0] / Nfold)),
                'recall'    :   ((positive[1] / Nfold), (negative[1] / Nfold)),
                'fscore'    :   ((positive[2] / Nfold), (negative[2] / Nfold)),
                'accuracy'  :   (avg_accuracy / Nfold)
            }

            classifier_report[name] = report

        return classifier_report

    def train_test(self, train_data, test_data):

        classifier_report = {}
        train_text = train_data['text']
        train_target = train_data['target']

        negative = [i for i, v in enumerate(train_target) if v == -1]
        positive = [i for i, v in enumerate(train_target) if v == 1]
        if float(len(negative)) / float(len(positive)) > 2:
            for i in positive:
                train_text.append(train_text[i])
                train_target.append(train_target[i])


        for name in self.classifiers:
            accuracy = 0.0

            positive = [0.0, 0.0, 0.0]    #precision, recall, fscore
            negative = [0.0, 0.0, 0.0]

            classifier = self.get_classifier(name)

            text_fit = classifier.fit(train_data['text'], train_data['target'])
            predicted = text_fit.predict(test_data['text'])

            accuracy += accuracy_score(test_data['target'], predicted)

            confusionMatrix = confusion_matrix(test_data['target'], predicted, labels=[-1, 0, 1])

            l = 0.000001
            positive[0] += (float(confusionMatrix[2][2]) / (confusionMatrix[2][2] + confusionMatrix[1][2] + confusionMatrix[0][2] + l))
            positive[1] += (float(confusionMatrix[2][2]) / (confusionMatrix[2][0] + confusionMatrix[2][1] + confusionMatrix[2][2] + l))
            positive[2] = (2 * positive[0] * positive[1]) / (positive[0] + positive[1] + l)

            negative[0] += (float(confusionMatrix[0][0]) / (confusionMatrix[0][0] + confusionMatrix[1][0] + confusionMatrix[2][0]) + l)
            negative[1] += (float(confusionMatrix[0][0]) / (confusionMatrix[0][0] + confusionMatrix[0][1] + confusionMatrix[0][2]) + l)
            negative[2] = (2 * negative[0] * negative[1]) / (negative[0] + negative[1] + l)

            report = {
                'precision' :   (positive[0], negative[0]),
                'recall'    :   (positive[1], negative[1]),
                'fscore'    :   (positive[2], negative[2]),
                'accuracy'  :   accuracy
            }

            classifier_report[name] = report

        return classifier_report

    def my_tokenizer(self, text):
        #stemmer = PorterStemmer()
        words = [ re.match('^[a-zA-Z\'-]+', w).group() for w in text.split(" ") if re.match('^[a-zA-Z\'-]+', w) != None]
        #process_words = [stemmer.stem(w) for w in words]
        return words


##################################print and main function###############################

def print_report(report):

        if not report:
            print ("None")
            return

        print ('%-18s %10s %10s %10s %10s' % ('CLASSIFIER', 'PRECISION', 'RECALL', 'FSCORE', 'ACCURACY'))

        for cls, rep in report.iteritems():
            print ('')
            print ("%-18s %10.2f %10.2f %10.2f %10.2f" % (cls, (100 * report[cls]['precision'][0]), (100 * report[cls]['recall'][0]), (100 * report[cls]['fscore'][0]), (100 * report[cls]['accuracy'])))
            print ("%-18s %10.2f %10.2f %10.2f" % ('', (100 * report[cls]['precision'][1]), (100 * report[cls]['recall'][1]), (100 * report[cls]['fscore'][1])))




def main():

    data_path = "training-Obama-Romney-tweets.xlsx"
    #test_data_path = ''
    test_data_path = 'testing-Obama-Romney-tweets.xlsx'


    processed_train = Data_preprocessing(data_path)
    classification = TweetClassifier()

    if test_data_path != '':

        processed_test = Data_preprocessing(test_data_path)

        print '\n****** OBAMA ******\n'
        data = processed_train.load_excel_data('Obama')
        data_test = processed_test.load_excel_data('Obama')
        report = classification.train_test(data, data_test)
        print_report(report)

        print '\n****** ROMNEY ******\n'
        data = processed_train.load_excel_data('Romney')
        data_test = processed_test.load_excel_data('Romney')
        report = classification.train_test(data, data_test)
        print_report(report)

    else:
        print '\n****** OBAMA ******\n'
        data = processed_train.load_excel_data('Obama')
        report = classification.crossvalidate(data, 10)
        print_report(report)


        print '\n****** ROMNEY ******\n'
        data = processed_train.load_excel_data('Romney')
        report = classification.crossvalidate(data, 10)
        print_report(report)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()

