import numpy as np
import h5py
from PIL import Image

#* save and load h5 file
def store_parameters(parameters, filename):

    with h5py.File(filename, 'w') as hdf:
        for key, value in parameters.items():
            hdf.create_dataset(key, data = value)

def load_parameters(filename):
    parameters = {}

    with h5py.File(filename, 'r') as hdf:
        for key in hdf.keys():
            parameters[key] = hdf[key][:]

#* the defination of activation function
def relu(Z):
    
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)

    cache = Z
    return A

def sigmoid(Z):
    
    A = 1/(1 + np.exp(-Z))
    assert(A.shape == Z.shape)

    cache = Z
    return A

#* dZ = dA* g'(Z)
def back_relu(dA, Z):
    
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0 ] = 0

    return dZ

def back_sigmoid(dA, Z):

    s = 1/(1 + np.exp(- Z))
    dZ = dA * s * (1 - s)

    return dZ

#* cost function
def cost_fun(A, Y):
    m = Y.shape[1]

    cost = -(1/m) * ( np.dot(Y, np.log(A.T)) + np.dot( 1 - Y , np.log( 1 - A ).T ) )
    cost = np.squeeze(cost)
    return cost


#* initialize the parameters: W and b

def initialize_parameter(sizes_of_layers):
    np.random.seed(1)
    parameters = {}
    #* only concern about the layer 2 to layer L, layer 1 is the input layer
    for i in range(1, len(sizes_of_layers)):
        parameters["W" + str(i)] = np.random.randn(sizes_of_layers[i], sizes_of_layers[i - 1]) / np.sqrt(sizes_of_layers[i - 1])
        parameters["b" + str(i)] = np.zeros((sizes_of_layers[i], 1))

    return parameters



#* forward
def forward_porpogate(X, parameters, activation):

    caches = []

    activation_dict = {"relu": relu,
                       "sigmoid": sigmoid}
    
    activation_fun = activation_dict[activation]

    L = len(parameters) // 2 # the number of layers, input layer doesn't contain

    A = X
    for i in range(1, L):
        A_prev = A
        W = parameters["W" + str(i)]
        b = parameters["b" + str(i)]
        Z = np.dot(W, A_prev) + b
        A = activation_fun(Z)
        caches.append((A, W, b, Z, A_prev))
        
        '''
        print ("A_prev" + str(A_prev.shape))
        print ("W" + str(W.shape))
        print ("b" + str(b.shape))
        print ("Z" + str(Z.shape))
        print ("A" + str(A.shape))
        '''


    WL = parameters["W" + str(L)]
    bL = parameters["b" + str(L)]
    ZL = np.dot(WL, A) + bL
    AL = sigmoid(ZL)
    
    '''
    print ("A" + str(A.shape))
    print ("WL" + str(WL.shape))
    print ("bL" + str(bL.shape))
    print ("ZL" + str(ZL.shape))
    print ("AL" + str(AL.shape))
    '''

    caches.append((AL, WL, bL, ZL, A))

    return caches


#*backward
def back_propogate(Y, caches, activation):

    activation_dict={"relu": back_relu,
                     "sigmoid": back_sigmoid}
    
    activation_fun = activation_dict[activation]


    #* caches - AL WL bL ZL A_prev
    grads = {}

    m = Y.shape[1]

    L = len(caches) # the number of layers, input layer doesn't contain
    
    AL, WL, bL, ZL, AL_prev = caches[L - 1]

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dZL = back_sigmoid(dAL, ZL)
    dWL = (1/m) * np.dot(dZL, AL_prev.T)
    dbL = (1/m) * np.sum(dZL, axis = 1, keepdims = True)
    dAL_next = np.dot(WL.T, dZL)

    grads["dW" + str(L)] = dWL
    grads["dA" + str(L)] = dAL_next #! it denots dA^l-1 !!!
    grads["db" + str(L)] = dbL

    for i in reversed(range(L - 1 )):
        A, W, b, Z, A_prev = caches[i]

        dA_current = grads["dA" + str(i + 2)]
        dZ = activation_fun(dA_current, Z)
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
        dA_next = np.dot(W.T, dZ)

        grads["dW" + str(i + 1)] = dW
        grads["dA" + str(i + 1)] = dA_next #! it denots dA^l-1 !!!
        grads["db" + str(i + 1)] = db

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for i in range(1 , L + 1):
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * grads["dW" + str(i)]
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * grads["db" + str(i)]

    return parameters

#* the training process
def L_layer_model_training(X, Y, sizes_of_layers, activation, learning_rate, iteration_num, print_cost):

    #* check whether the size of the first and last layer are set correctly.
    assert sizes_of_layers[0] == X.shape[0], "the size of the first layer does not match the input layer."
    assert sizes_of_layers[-1] == 1, "the output is a value not array!"

    parameters = initialize_parameter(sizes_of_layers)

    cost_list = []

    for i in range(0, iteration_num ):
        caches = forward_porpogate(X, parameters, activation)
        grads = back_propogate(Y, caches, activation)
        parameters = update_parameters(parameters, grads, learning_rate)

        AL, WL, bL, ZL, A = caches[-1]
        cost = cost_fun(AL, Y)

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            cost_list.append(cost)

    store_parameters(parameters, "parameters.h5")

    return parameters

def predict_picture(filename, num_px, num_py, activation):
    classes = [b'non-cat' , b'cat']

    img = Image.open(filename)
    img_resized = img.resize((num_px, num_py))
    img_rgb = img_resized.convert('RGB')
    img_rgb = np.array(img_rgb)

    plt.imshow(img_rgb)

    img_input = img_rgb.reshape((num_px * num_py * 3,1))

    parameters = load_parameters("parameters.h5")
    
    caches = forward_porpogate(img_input, parameters, activation)

    AL, WL, bL, ZL, A = caches[-1]

    y_output = AL

    if y_output < 0.5:
        y_final = 0
    else:
        y_final = 1

    print("y = " + str(y_final) + ", you predicted that it is a \"" + classes[y_final].decode("utf-8") +  "\" picture.")