import numpy as np

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return np.array([float(line.strip()) for line in data])

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def verify_softmax(calculated, expected, tolerance=1e-6):
    return np.allclose(calculated, expected, atol=tolerance)

data = read_data('data.txt')
calculated_softmax = softmax(data)
expected_softmax = read_data('expected_softmax.txt')
is_correct = verify_softmax(calculated_softmax, expected_softmax)

if(is_correct == False):
    for i in range(len(calculated_softmax)):
        if(calculated_softmax[i] != expected_softmax[i]):
            print('i = {}, calculated_softmax = {}, expected_softmax = {}'.format(i, calculated_softmax[i], expected_softmax[i]))

print("Calculated Softmax:", calculated_softmax)
print("Expected Softmax:", expected_softmax)
print("Softmax calculation is correct:", is_correct)
