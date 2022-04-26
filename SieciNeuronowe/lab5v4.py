import math
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt


def easy_f(x):
    return 10 * math.sin(x/10)


def f(x):
    return x**2*math.sin(x) + 100*math.sin(x)*math.cos(x)


# Activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_intergal(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Layer:
    def __init__(self, size, prev_layer_size):
        self.size = size
        self.inp_size = prev_layer_size
        self.weights = np.random.rand(self.inp_size, self.size) - 0.5
        self.biases = np.random.rand(1, self.size) - 0.5

    def set_funs(self, act_fun, act_int_fun):
        self.act_fun = act_fun
        self.act_int_fun = act_int_fun

    def forward(self, prev_layer_output):
        self.input = prev_layer_output
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.act_fun(self.output)

    def backward(self, errors, learning_rate):
        self.errors = self.act_int_fun(self.output) * errors

        input_errors = np.dot(self.errors, self.weights.T)
        weights_errors = np.dot(self.input.T, self.errors)

        self.weights += learning_rate * weights_errors
        self.biases += learning_rate * self.errors

        return input_errors


class Network:
    def __init__(self, middle_layer_sizes):
        self.layer_sizes = [1] + middle_layer_sizes + [1]
        self.layers = []
        for i, size in enumerate(self.layer_sizes[1:]):
            self.layers.append(Layer(size, self.layer_sizes[i]))

    def set_funs(self, act_fun, act_int_fun):
        self.act_fun = act_fun
        self.act_int_fun = act_int_fun
        for layer in self.layers:
            layer.set_funs(act_fun, act_int_fun)

    def train(self, epochs, learning_rate, training_vals, proper_res):
        with tqdm(total=epochs * len(training_vals)) as progress:
            for i in range(epochs):
                err = 0
                for j, x in enumerate(training_vals):
                    out = x
                    for layer in self.layers:
                        out = layer.forward(out)
                    err += abs(proper_res[j] - out[0])

                    errors = 2 * (proper_res[j] - out[0])
                    for layer in reversed(self.layers):
                        errors = layer.backward(errors, learning_rate)
                    progress.update(1)

    def predict(self, x):
        out = [x]
        for layer in self.layers:
            out = layer.forward(out)
        return out[0]


def experiments():
    # Prepare training set and test set
    xs = []
    ys = []
    tests = []
    res = []
    scale = 10
    for i in range(-40 * scale, 40 * scale + 1, 1):
        xs.append(i/scale)
        ys.append(f(i/scale))
        tests.append(i/scale)
        tests.append((i/scale + 1/(2*scale)))
        res.append(f(i/scale))
        res.append(f(i/scale + 1/(2*scale)))
    xa = np.asarray(xs)
    ya = np.asarray(ys)
    testa = np.asarray(tests)
    xa = xa.reshape(len(xa), 1)
    ya = ya.reshape(len(ya), 1)
    testa = testa.reshape(len(testa), 1)
    scale_x = MinMaxScaler()
    xa = scale_x.fit_transform(xa)
    testa = scale_x.fit_transform(testa)
    scale_y = MinMaxScaler()
    ya = scale_y.fit_transform(ya)

    neuron_combs = [[100], [1000], [2000], [5000], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100]]

    for comb in neuron_combs:
        np.random.seed(24513)

        netw = Network(comb)
        netw.set_funs(sigmoid, sigmoid_intergal)
        netw.train(10000, 0.1, xa, ya)

        predicts = []
        for q in testa:
            predicts.append(netw.predict(q))

        predicta = np.asarray(predicts)
        predicta = scale_y.inverse_transform(predicta)
        predicta = predicta.reshape(1, len(predicta))
        predicts = predicta.tolist()[0]

        plt.clf()
        plt.plot(tests, res, '-', label="Expected")
        plt.plot(tests, predicts, '-', label="Predicted")
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Neurons in discrete layers: ' + str(comb))
        plt.savefig('wykres' + str(comb) + '.png')
        # plt.show()


if __name__ == "__main__":
    experiments()
