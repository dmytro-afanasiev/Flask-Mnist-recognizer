import numpy as np
import json

def load_and_prepare_mnist():
    try:
        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]**2) /255.0
        
        ones = np.zeros((len(y_train), 10))

        for i, l in enumerate(y_train):
            ones[i][l] = 1
        y_train = ones

        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2) / 255.0

        ones = np.zeros((len(y_test), 10))

        for i, l in enumerate(y_test):
            ones[i][l] = 1
        y_test = ones
        return (x_train, y_train, x_test, y_test)

    except ImportError:
        print('Please, install tensorflow and keras')

class SimpleClassificationModel:
    """
    Create a simple classification model with one hidden layer
    """
    def __init__(self, hidden_layer=50, iterations=400, learning_rate=0.005, dropout=False):
        self.hidden_layer = hidden_layer
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.dropout = dropout
    
    def _generate_parameters(self, input_layer: int, output_layer: int):
        self.input_layer = input_layer
        self.output_layer = output_layer
        parameters = {}
        parameters['W1'] = 0.2 *np.random.random(size=(input_layer, self.hidden_layer)) - 0.1
        parameters['W2'] = 0.2 *np.random.random(size=(self.hidden_layer, output_layer)) - 0.1
        return parameters

    @staticmethod
    def _relu(x):
        return (x>0) * x

    @staticmethod
    def _relu_derivative(x):
        return x>=0
    
    
    def fit(self, train_x, train_y):
        """
        x.shape = (number_of_examples, number_of_params)
        y.shape = (number_or_examples, classes [0,1,0,0, ... 0])
        """
        m = train_x.shape[0]
        assert train_x.shape[0] == train_y.shape[0], "number of examples must be the same"
        self._parameters = self._generate_parameters(train_x.shape[1], train_y.shape[1])

        relu = lambda x: (x>0) * x
        reluD = lambda x: x>=0

        for i in range(self.iterations):
            error = 0
            for j in range(m):
                layer_0 = train_x[j].reshape(1, train_x[j].shape[0])
                layer_1 = relu(np.dot(layer_0, self._parameters['W1']))
                if self.dropout:
                    dropout_mask = np.random.randint(2, size=layer_1.shape)
                    layer_1 *=dropout_mask *2
                layer_2 = np.dot(layer_1, self._parameters['W2'])
                error += np.sum((train_y[j] - layer_2)**2)

                layer_2_delta = layer_2 - train_y[j]
                layer_1_delta = layer_2_delta.dot(self._parameters['W2'].T) * reluD(layer_1)
                if self.dropout:
                    layer_1_delta *= dropout_mask
                self._parameters['W2'] -= 0.005 * layer_1.T.dot(layer_2_delta)
                self._parameters['W1'] -= 0.005 * layer_0.T.dot(layer_1_delta)
            print(f'I: {i}, error: {error/float(len(train_x))}')



    def print_accuracy(self, test_x, test_y):
        assert self.input_layer == test_x.shape[1], "dimentions of test data must be the same as a dimentions of train data"
        assert self.output_layer == test_y.shape[1], "dimentions of test data must be the same as a dimentions of train data"
        
        correct = 0
        for i in range(len(test_x)):
            n = np.argmax(np.dot(SimpleClassificationModel._relu(np.dot(test_x[i], self._parameters['W1'])), self._parameters['W2']))
            if test_y[i][n] == 1:
                correct+=1

        print('Accuracy:', correct/len(test_x)*100, "%")
        
    def predict_proba(self, x):
        assert x.shape[1] == self.input_layer
        res = np.dot(SimpleClassificationModel._relu(np.dot(x, self._parameters['W1'])), self._parameters['W2'])
        print(f"class: {np.argmax(res)},\nAll answer: {res}")
        return np.argmax(res)


    def serialize_to_json(self, path : str):
        par = {
            'W1': self._parameters['W1'].tolist(),
            'W2': self._parameters['W2'].tolist()
        }
        data = {
            'hidden_layer': self.hidden_layer,
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'dropout': self.dropout,
            '_parameters': par
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load_model_from_json(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f);
        model = cls()
        for key in data.keys():
            setattr(model, key, data[key])
        model._parameters = {
            'W1': np.array(data['_parameters']["W1"]),
            'W2': np.array(data['_parameters']['W2'])
        }
        model.input_layer = model._parameters['W1'].shape[0]
        model.output_layer = model._parameters['W2'].shape[1]
        return model

    def get_prepare_simple(self, l: list):
        try:
            res = np.array(l).reshape(1, self.input_layer) / 255
            return res
        except Exception:
            print("your data is not valid");
    

