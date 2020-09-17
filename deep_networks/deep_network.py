import numpy as np
import pickle
import h5py


class InputDimenitionsError(Exception):
    pass

class DeepNetworkModel:
    def __init__(self, layers, learning_rate=0.0075, number_of_iterations=2000, show_cost=False):
        if layers:
            self._parameters = self._init_parameters(layers)
            self.layers = layers
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.print_cost = show_cost
        
    @property
    def parameters(self):
        return self._parameters
    
    @staticmethod
    def _sigmoid_forward(z):
        cache = z
        return (1/(1+np.exp(-z)), cache)
    
    @staticmethod
    def _relu_forward(z):
        cache = z
        return (np.maximum(0, z), cache)
    
    @staticmethod
    def _sigmoid_backward(dA, cache):
        z = cache
        s = 1/(1+np.exp(-z))
        dZ = dA * s * (1-s)
        return dZ
    
    @staticmethod
    def _relu_backward(dA, cache):
        z = cache
        dZ = np.array(dA, copy=True) 
        dZ[z <= 0] = 0
        return dZ
    
    def _compute_cost(self, AL, y):
        m = y.shape[1]
        cost = np.sum(y*np.log(AL)+(1-y)*np.log(1-AL))/-m
        cost = np.squeeze(cost)
        return cost
    
    def _init_parameters(self, layers):
        np.random.seed(1)
        parameters = {}
        L = len(layers)           
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1])
            parameters['b' + str(l)] = np.zeros((layers[l], 1))
        return parameters
    
    def _linear_forward(self, a, w, b):
        z = w.dot(a) + b
        cache = (a, w, b)
        return z, cache
    
    def _linear_activation_forward(self, A_prev, w, b, activation):
        if activation == "sigmoid":
            z, linear_cache = self._linear_forward(A_prev, w, b)
            a, activation_cache = DeepNetworkModel._sigmoid_forward(z)
    
        elif activation == "relu":
            z, linear_cache = self._linear_forward(A_prev, w, b)
            a, activation_cache = DeepNetworkModel._relu_forward(z)
    
        cache = (linear_cache, activation_cache)
        return a, cache
    
    def _L_model_forward(self, x):
        caches = []
        a = x
        L = len(self._parameters) // 2                
        for l in range(1, L):
            A_prev = a 
            a, cache = self._linear_activation_forward(A_prev, self._parameters['W' + str(l)], self._parameters['b' + str(l)], activation = "relu")
            caches.append(cache)
        AL, cache = self._linear_activation_forward(a, self._parameters['W' + str(L)], self._parameters['b' + str(L)], activation = "sigmoid")
        caches.append(cache)
        return AL, caches
    
    def _linear_backward(self, dZ, cache):
        A_prev, w, b = cache
        m = A_prev.shape[1]
        dW = 1./m * np.dot(dZ,A_prev.T)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(w.T,dZ) 
        return dA_prev, dW, db
    
    def _linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = DeepNetworkModel._relu_backward(dA, activation_cache)
            dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        elif activation == "sigmoid":
            dZ = DeepNetworkModel._sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def _L_model_backward(self, AL, y, caches):
        grads = {}
        L = len(caches) 
        m = AL.shape[1]
        y = y.reshape(AL.shape) 
        dAL = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self._linear_activation_backward(dAL, current_cache, activation = "sigmoid")
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        return grads


    def _update_params(self, grads):
        L = len(self._parameters) // 2 
        for l in range(L):
            self._parameters["W" + str(l+1)] = self._parameters["W" + str(l+1)] - self.learning_rate * grads["dW" + str(l+1)]
            self._parameters["b" + str(l+1)] = self._parameters["b" + str(l+1)] - self.learning_rate * grads["db" + str(l+1)]
        return self._parameters

    def predict(self, x, y):
        m = x.shape[1]
        n = len(self._parameters) // 2 
        p = np.zeros((1,m))
        probas, caches = self._L_model_forward(x)
        print('probas - ', probas)
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        print("Accuracy: "  + str(np.sum((p == y)/m)))
        return p

    
    def predict_image(self, filename):
        try:
            from PIL import Image
            image = Image.open(filename).resize((64,64))
            image = np.array(image).reshape((64*64*3,1))
            image = image /255.
            print(self._L_model_forward(image))
        except ImportError:
            print('Please, install PIL')


    def fit(self, x, y, show_cost=False):

        if (x.shape[0] != self.layers[0]):
            raise InputDimenitionsError("The number of input parameters must be the same as a size of the first layer")
        if (x.shape[1] != y.shape[1]):
            raise InputDimenitionsError('X.shape[1] must be the same as Y.shape[1]')

        self.show_cost= show_cost
        np.random.seed(1)
        costs = []          
        for i in range(0, self.number_of_iterations):
            AL, caches = self._L_model_forward(x)
            cost = self._compute_cost(AL, y)
            grads = self._L_model_backward(AL, y, caches)
            self._update_params(grads)
            
            if self.show_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                costs.append(cost)

        self.costs = costs
        print('Fitting is ready')
        try:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per hundreds)')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.show()
        except NameError:
            print('Please, install package "matplotlib" to see learning graph!')

    def serialize(self, filename : str):
        dirname = 'models/'
        if not filename.endswith('.data'):
            filename = filename + '.data'
        with open(dirname+filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def get_model_from_file(filename: str):
        dirname = 'models/'
        if not filename.endswith('.data'):
            filename =  filename + '.data'
        with open(dirname+filename, 'rb') as file:
            return pickle.load(file)
    


