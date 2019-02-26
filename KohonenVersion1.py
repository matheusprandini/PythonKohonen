import numpy as np
np.warnings.filterwarnings('ignore')

##### CLASS KOHONEN VERSION 1 ####

#### KOHONEN WITH R(radius) = 0 ####

class Kohonen:

    def __init__(self, number_input_neurons, number_output_neurons, learning_rate=0.4):
		
		# Initialize Weights and Learning rate
        self.weights = self.initialize_weights(number_input_neurons, number_output_neurons)
        self.learning_rate = learning_rate
    
    def initialize_weights(self, number_input_neurons, number_output_neurons):
        # Weights initialization -> extremely small random values
        W = np.random.randn(number_output_neurons, number_input_neurons)*0.01

        return W
		
    def compute_winner_neuron(self, input_data):
	    # Compute the euclidean distance between the input_data and the weights vector
        diff = np.subtract(input_data, self.weights)
        power_diff = np.power(diff, 2)
        euclidean_distance = np.sum(power_diff, axis=1, keepdims=True)
		
		# Gets the winner neuron with the least euclidean distance
        winner_neuron = np.argmin(euclidean_distance)

        return winner_neuron
		
    def update_weights_winner_neuron(self, data, winner_neuron):
		# Update the weights of the winner neuron
        self.weights[winner_neuron] += np.multiply(self.learning_rate, (np.subtract(data, self.weights[winner_neuron])))
		
    def training_phase(self, input_data, number_iterations=10000):
        for i in range(number_iterations):
            for data in input_data:
                winner_neuron = self.compute_winner_neuron(data)
                self.update_weights_winner_neuron(data, winner_neuron)
				
    def test_phase(self, input_data):
        print('Final weights: ', self.weights, '\n')
        for data in input_data:
            winner_neuron = self.compute_winner_neuron(data)
            print('Input: ', data, ' - Output: ', winner_neuron)			