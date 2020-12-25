import numpy as np
class NN():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((5, 1)) - 1

    def training(self, inputs_training, outputs_training, iterate):
        for iteration in range(iterate):
            output = self.decide(inputs_training)
            error = outputs_training - output
            adjust = np.dot(inputs_training.T, error * self.give_sigmoid_derivative(output))
            self.synaptic_weights =self.synaptic_weights + adjust
        print("training results of ", iterate, "th iteration : ")
        print(output)

    def give_sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def give_sigmoid_derivative(self, x):
        return x*(1-x)
    def random_synaptic_weights(self):
        print("Random  synaptic weights: ")
        s = neural_network.synaptic_weights
        return s
    def decide(self, inputs):
        inputs = inputs.astype(float)
        output = self.give_sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

if __name__ == "__main__":
    neural_network = NN()
    print(neural_network.random_synaptic_weights())
    inputs_training = np.array([
                                [0, 1, 0, 0, 0],
                                [1, 0, 1, 1, 1],
                                [1, 0, 1, 1, 1],
                                [0, 1, 1, 0, 0],
                                [1, 1, 1, 1, 1]
                                ])
    outputs_training = np.array([
                                 [0],
                                 [1],
                                 [1],
                                 [0],
                                 [1]
                                 ])
    neural_network.training(inputs_training, outputs_training, 90000)
    print("Adjusted synaptic weights after training :   ")
    print(neural_network.synaptic_weights)
    a = str(input("input data 1: "))
    b = str(input("input data 2: "))
    c = str(input("input data 3: "))
    d = str(input("input data 4: "))
    e = str(input("input data 5: "))
    print("Predicted result : ")
    print(neural_network.decide(np.array([a,b,c,d,e])))
