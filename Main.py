from KohonenVersion1 import Kohonen
import numpy as np

input_data = np.array([[1,1,0,0,0],[1,0,0,0,0],[0,0,0,0,1],[0,0,0,1,1],[0,0,1,0,0]])

kohonen = Kohonen(number_input_neurons=input_data.shape[1], number_output_neurons=3)

kohonen.training_phase(input_data)
kohonen.test_phase(input_data)