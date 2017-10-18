import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # Advance the window forward one element at a time until the left index
    # of the window meets the max of the series
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):

    # Initilize the model object
    rnn_model = Sequential()

    # Add an LSTM layer with 5 hidden units. Adjust input to the specified
    # window size
    rnn_model.add(LSTM(units = 5, input_shape = (window_size,1)))

    # Add a final fully connected layer to connect to labels
    # FIXME keep the specified activation function? Let's see how things go.
    rnn_model.add(Dense(1, activation = "tanh"))

    # Return a completed model
    return rnn_model

### TODO: return the text input with only ascii lowercase and the punctuation
# given below included.
def cleaned_text(text):

    from string import ascii_lowercase

    text = text.lower()
    punctuation = ['!', ',', '.', ':', ';', '?']
    remove_stuff = [x for x in text if x not in list(ascii_lowercase) + punctuation + list(" ")]

    # Iterate across items to replace, and apply a vectorized substitution
    # within the text
    for bad_element in list(set(remove_stuff)):
        text = text.replace(bad_element, "")

    return text


### TODO: fill out the function below that transforms the input text and
# window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    # FIXME I really thought range took keyword arguments...
    #for step_index in range(0, stop = len(text) - window_size, step = step_size):

    # Iterate our window through elements of the text in index values
    # equal to step_size
    for step_index in range(0, len(text) - window_size, step_size):
        inputs.append(text[step_index : step_index + window_size])
        outputs.append(text[step_index + window_size])

    return inputs,outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy
# loss
def build_part2_RNN(window_size, num_chars):

    # Initilize the model object and add the required layers
    bigger_rnn = Sequential()
    bigger_rnn.add(LSTM(units = 200, input_shape = (window_size, num_chars)))
    bigger_rnn.add(Dense(num_chars, activation = "softmax"))
    
    return bigger_rnn
