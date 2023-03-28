import numpy as np


def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test

 



class Neural_network(): 

    ##Lager en klasse som inneholder alle funksjonene som skal brukes i det nevrale nettverket
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.input_lag = [0.0, 0.0]
        self.hidden_layer = [0.0, 0.0]
        self.ouput_lag = [0.0]

        self.error = 0
        self.lærings_rate = 0.1
        self.bias = 1

        self.vekt_input_hidden = [np.random.random(), np.random.random(), np.random.random(), np.random.random()]
        self.vekt_hidden_output = [np.random.random(), np.random.random()]

    ##Sigmoid funksjonen
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))



    ##Funksjon for trene på treningsdataen, går over alle dataene og kjører gjennom det nevrale nettverket
    def train(self):
        for x, y in zip(self.X_train, self.y_train):
            # Sender input fremover i det nevrale nettverket
            self.input_lag = x
            self.hidden_layer[0] = self.sigmoid(np.dot(self.input_lag, self.vekt_input_hidden[:2]) + self.bias)
            self.hidden_layer[1] = self.sigmoid(np.dot(self.input_lag, self.vekt_input_hidden[2:]) + self.bias)
            self.ouput_lag[0] = np.dot(self.hidden_layer, self.vekt_hidden_output)


            ## Sender tilbake og oppdaterer vektene til nodene
            self.error = y - self.ouput_lag[0]
            self.vekt_hidden_output[0] += self.lærings_rate * self.error * self.hidden_layer[0]
            self.vekt_hidden_output[1] += self.lærings_rate * self.error * self.hidden_layer[1]
            self.vekt_input_hidden[0] += self.lærings_rate * self.error * self.vekt_hidden_output[0] * self.hidden_layer[0] * (1 - self.hidden_layer[0]) * self.input_lag[0]
            self.vekt_input_hidden[1] += self.lærings_rate * self.error * self.vekt_hidden_output[0] * self.hidden_layer[0] * (1 - self.hidden_layer[0]) * self.input_lag[1]
            self.vekt_input_hidden[2] += self.lærings_rate * self.error * self.vekt_hidden_output[1] * self.hidden_layer[1] * (1 - self.hidden_layer[1]) * self.input_lag[0]
            self.vekt_input_hidden[3] += self.lærings_rate * self.error * self.vekt_hidden_output[1] * self.hidden_layer[1] * (1 - self.hidden_layer[1]) * self.input_lag[1]



    def predict(self):
        ## Gjør det samme som tidligere bare med test data
        for (x,y) in zip(self.X_test, self.y_test):
            self.input_lag = x
            self.hidden_layer[0] = self.sigmoid(self.input_lag[0] * self.vekt_input_hidden[0] + self.input_lag[1] * self.vekt_input_hidden[1])
            self.hidden_layer[1] = self.sigmoid(self.input_lag[0] * self.vekt_input_hidden[2] + self.input_lag[1] * self.vekt_input_hidden[3])
            self.ouput_lag[0] = self.hidden_layer[0] * self.vekt_hidden_output[0] + self.hidden_layer[1] * self.vekt_hidden_output[1]
            self.error = y - self.ouput_lag[0]
        print("Test error: ", self.error)
    
    def mse(self):
        self.error = 0
        for (x,y) in zip(self.X_test, self.y_test):
            self.input_lag = x
            self.hidden_layer[0] = self.sigmoid(self.input_lag[0] * self.vekt_input_hidden[0] + self.input_lag[1] * self.vekt_input_hidden[1])
            self.hidden_layer[1] = self.sigmoid(self.input_lag[0] * self.vekt_input_hidden[2] + self.input_lag[1] * self.vekt_input_hidden[3])
            self.ouput_lag[0] = self.hidden_layer[0] * self.vekt_hidden_output[0] + self.hidden_layer[1] * self.vekt_hidden_output[1]
            self.error += (y - self.ouput_lag[0])**2
        self.error = self.error / len(self.X_test)
        print("MSE: ", self.error)
    
    ##Funksjon som kjører alle funksjonene
    def run(self):
        self.train()
        self.mse()
        self.predict()
    
    

if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    nn = Neural_network(X_train, y_train, X_test, y_test)
    nn.run()
