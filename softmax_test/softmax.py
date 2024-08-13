import numpy as np

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return np.array([float(line.strip()) for line in data])

def softmax(x):
    max = 0
    pre_max = max
    ans = []
    for i in range(0,(1024*768),768):
        # if(i<2048):
        #     print(i)
        #     print(x[i:i+16])
        a = np.exp(x[i:i+768] - np.max(x[i:i+768]))/np.sum(np.exp(x[i:i+768] - np.max(x[i:i+768])))
        # a = np.exp(x[i:i+768])/np.sum(np.exp(x[i:i+768]))
        for n in a:
            ans.append(n)
    return ans

def verify_softmax(calculated, expected, tolerance=1e-6):
    return np.allclose(calculated, expected, atol=tolerance)

data = read_data('random_floats.txt')
# data = read_data('random_floats_16bit_fixed_point.txt')
calculated_softmax = softmax(data)
# np.savetxt('expected_softmax.txt', calculated_softmax)
np.savetxt('expected_softmax_safe.txt', calculated_softmax)
# np.savetxt('expected_softmax_16bit_fixed_point.txt', calculated_softmax)
# expected_softmax = read_data('expected_softmax.txt')
# is_correct = verify_softmax(calculated_softmax, expected_softmax)

# if(is_correct == False):
#     for i in range(len(calculated_softmax)):
#         if(calculated_softmax[i] != expected_softmax[i]):
#             print('i = {}, calculated_softmax = {}, expected_softmax = {}'.format(i, calculated_softmax[i], expected_softmax[i]))

# print("Calculated Softmax:", calculated_softmax)
# print("Expected Softmax:", expected_softmax)
# print("Softmax calculation is correct:", is_correct)
