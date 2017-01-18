import random
import math

random.seed(10)

def rand(a, b):
    return random.uniform(a, b)

def sigmoid(value):
    #return 1/(1+numpy.exp(-value))
    return math.tanh(value)

def derSigmoid(value):
    # return value - value ** 2
    return 1 - value ** 2
class Network:
    def __init__(self, size = [3,2,3] ,inputs = [1.0, 0.25, -0.5] ,expected_output = [-0.756,-0.5,1.0]):

        """"
        This function initialize the layouts with the expected inputs / outputs, and generate
        random numbers for the different weights.
        """""

        # Store the data sets
        self.inputs = inputs
        self.expected_output = expected_output

        # Initialize size
        self.size_input = size[0] + 1 # +1 for the bias node
        self.size_hidden = size[1]
        self.size_output = size[2]

        # Create nodes ; we initialize it to 1 because of the initial state of the network
        self.iNodes = [1.0] * self.size_input
        self.hNodes = [1.0] * self.size_hidden
        self.oNodes = [1.0] * self.size_output

        # Create Weights
        # Here we will create a Matrix containing the weight, from top to bottom

        temp_mat = []
        for i in range(self.size_input):
            temp_mat.append([0.0] * self.size_hidden)
        self.weights_input_to_hidden = temp_mat
        temp_mat = []
        for i in range(self.size_hidden):
            temp_mat.append([0.0] * self.size_output)
        self.weights_hidden_to_output = temp_mat

        for i in range(self.size_input):
            for j in range(self.size_hidden):
                self.weights_input_to_hidden[i][j] = rand(-1.0, 1.0)
        for i in range(self.size_hidden):
            for j in range(self.size_output):
                self.weights_hidden_to_output[i][j] = rand(-1.0, 1.0)

        # Weight Change Matrix
        # Used in backPropagation

        temp_mat = []
        for i in range(self.size_hidden):
            temp_mat.append([0.0] * self.size_output)
        self.lChangeOut = temp_mat

        temp_mat = []
        for i in range(self.size_input):
            temp_mat.append([0.0] * self.size_hidden)
        self.lChangeIn = temp_mat

    def feedForward(self, input):
        if len(input) != self.size_input - 1:
            raise ValueError("Input and Input_Layer aren't the same size")

        # Input Layer Calculation
        for i in range(self.size_input - 1):
            self.iNodes[i] = input[i]

        # Hidden Layer Calculation
        for hidden_node in range(self.size_hidden):
            nodeValue = 0
            for input_node in range(self.size_input):
                nodeValue += self.iNodes[input_node] * self.weights_input_to_hidden[input_node][hidden_node]
            self.hNodes[hidden_node] = sigmoid(nodeValue)

        # Output Layer Calculation
        for output_node in range(self.size_output):
            nodeValue = 0
            for hidden_node in range(self.size_hidden):
                nodeValue += self.hNodes[hidden_node] * self.weights_hidden_to_output[hidden_node][output_node]
            self.oNodes[output_node] = sigmoid(nodeValue)

        return self.oNodes[:]

    def backPropagation(self, N = 2.0, M = 1.0):
        if len(self.expected_output) != self.size_output:
            raise ValueError("Expected output and Ouput Layer aren't the same size")

        # We create the difference lists
        oDiff = [0.0] * self.size_output
        hDiff = [0.0] * self.size_hidden

        # We first calculate the error for the output layer
        for i in range(self.size_output):
            error = self.expected_output[i] - self.oNodes[i]
            oDiff[i] = derSigmoid(self.oNodes[i]) * error

        # We then calculate the error the the hidden layer
        for i in range(self.size_hidden):
            error = 0.0
            for j in range(self.size_output):
                error += oDiff[j] * self.weights_hidden_to_output[i][j]
            hDiff[i] = derSigmoid(self.hNodes[i]) * error

        # Update of the weights between hidden and output
        for i in range(self.size_hidden):
            for j in range(self.size_output):
                change = oDiff[j] * self.hNodes[i]
                self.weights_hidden_to_output[i][j] += N * change + M * self.lChangeOut[i][j]
                self.lChangeOut[i][j] = change

        # Update the weights between input and hidden
        for i in range(self.size_input):
            for j in range(self.size_hidden):
                change = hDiff[j] * self.iNodes[i]
                self.weights_input_to_hidden[i][j] += N * change + M * self.lChangeIn[i][j]
                self.lChangeIn[i][j] = change


        # We calculate the error
        error = 0
        for i in range(len(self.expected_output)):
            error += 0.5 * (self.expected_output[i] - self.oNodes[i]) ** 2
        return error

    def launch(self, threshold_eqm = 0.00002, inputs_group = [], N = 4.0, M = 1.0):
        #N : learning rate ; the higher the slower
        #M : momentum factor ; don't really understand this one, just leaving it there

        error = 100
        i=0
        while (error > threshold_eqm):
        #for i in range(1000):
            error = 0.0
            self.feedForward(self.inputs)
            error += self.backPropagation(N, M)
            if i % 100 == 0:
                print('error %-.7f' % error)

    def display(self):
        print("Input : ", end="")
        print("[ ", end="")
        for i in range(self.size_input - 1):
            print(str(self.iNodes[i]) + " ", end="")
        print("]")
        print("Expected Output : [ ", end="")
        for i in range(self.size_output):
            print(str(self.expected_output[i]) + " ", end="")
        print("]")
        print("Final Output : [ ", end="")
        for i in range(self.size_output):
            print(str(self.oNodes[i]) + " ", end="")
        print("]")

if __name__ == "__main__":
    print("Beginning of the program")
    new_network = Network()

    new_network.launch()

    new_network.display()