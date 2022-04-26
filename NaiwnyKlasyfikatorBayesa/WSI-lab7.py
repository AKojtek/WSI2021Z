import csv
import math
import random

def load_csv(file_name):
    dataset = []
    with open(file_name, 'r') as file:
        csv_file = csv.reader(file, delimiter=',')
        for row in csv_file:
            if not row:
                continue
            for i in range(len(row)-1):
                row[i] = float(row[i])
            dataset.append(row)
    return dataset

def divide_dataset(dataset, part):
    learn = []
    test = []
    train = dataset.copy()
    no = int(len(train)*part)
    for i in range(no):
        number = random.randrange(len(train))
        test.append(train[number])
        del train[number]

    learn = [x[:-1] for x in test]
    return train, learn, test

class NaiveBayesClassifier:
    def __init__(self, dataset):

        # do wyrzucenia
        self.dataset = dataset

        # Dictionary of divided rows by its classes
        self.classes = self.create_class(dataset)
        # Dictionray of mean, standard deviation of each argument of each class
        self.summarized = self.summarize_by_class()

    # Functions to calculate mean of given set and its standard deviation
    def mean_of_set(self, numbers):
        return sum(numbers)/float(len(numbers))

    def st_deviation(self, numbers):
        avg = self.mean_of_set(numbers)
        variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
        return math.sqrt(variance)

    # This method creates dictionary which separetes data from its class
    def create_class(self, dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in separated):
                separated[class_value] = []
            separated[class_value].append(vector[:-1])
        return separated

    # This method counts mean, standard deviation based on given dataset
    def summarize_dataset(self, dataset):
        zipped = zip(*dataset)
        zipped = list(zipped)
        summaries = []

        for column in zipped:
            data = []
            data.append(self.mean_of_set(column))
            data.append(self.st_deviation(column))
            data.append(len(column))
            summaries.append(data)

        return summaries

    def summarize_by_class(self):
        summaries = {}
        for item in self.classes:
            summaries[item] = self.summarize_dataset(self.classes[item])
        return summaries

    # Method to calculate probability
    def probability(self, x, mean, st_dev):
        exponent = math.exp(-((x-mean)**2 / (2 * st_dev**2 )))
        return (1 / (math.sqrt(2 * math.pi) * st_dev)) * exponent

    def calculate_class_probabilities(self, row):
        total_rows = sum([self.summarized[label][0][2] for label in self.summarized])
        probabilities = {}
        for class_value, class_summaries in self.summarized.items():
            probabilities[class_value] = self.summarized[class_value][0][2]/float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, count = class_summaries[i]
                probabilities[class_value] *= self.probability(row[i], mean, stdev)
        return probabilities

    def predict(self, row):
        probabilities = self.calculate_class_probabilities(row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    def predict_rows(self, dataset):
        for row in dataset:
            row.append(self.predict(row))


def main():
    dataset = load_csv('iris.csv')

    errors = 0
    elements = 0

    parts = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8]

    for part in parts:
        for _ in range(1000):
            train, learn, test = divide_dataset(dataset, part)

            classifier = NaiveBayesClassifier(train)
            classifier.predict_rows(learn)

            mistakes = 0
            for i in range(len(learn)):
                if learn[i] != test[i]:
                    # print(f'ROW {i}; For row {learn[i][:-1]} program predicted {learn[i][-1]} where correct answer is {test[i][-1]}')
                    mistakes += 1

            # print(f'Number of incorrect predictions: {mistakes}')
            # print(f'Program was succesful in {(len(learn)-mistakes)/len(learn)*100}% of cases')

            errors += mistakes
            elements += len(learn)

        print(f'Program was successful in {(elements-errors)/elements*100}% of cases when testing on {part} part of whole set')




if __name__ == "__main__":
    main()