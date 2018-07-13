import numpy as np

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weight1 = np.random.rand(self.input.shape[1], 4)
        self.weight2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weight1))
        self.output = sigmoid(np.dot(self.layer1, self.weight2))

    def back_prop(self):
        d_weight2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weight1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weight2.T) * sigmoid_derivative(self.layer1)))

        self.weight1 += d_weight1
        self.weight2 += d_weight2


if __name__ == '__main__':
    
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(X, y)

    for i in range(1500):
        nn.feedforward()
        nn.back_prop()

    print(nn.output)