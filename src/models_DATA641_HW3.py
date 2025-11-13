# models.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# used to set between the three chosen optimizers
def get_optimizer(name, clip=False, clipnorm_val=5):
    name = name.lower()
    if name == 'adam':
        return Adam(clipnorm=clipnorm_val) if clip else Adam()
    elif name == 'sgd':
        return SGD(clipnorm=clipnorm_val) if clip else SGD()
    elif name == 'rmsprop':
        return RMSprop(clipnorm=clipnorm_val) if clip else RMSprop()
    else:
        raise ValueError("optimizer_name must be one of ['adam', 'sgd', 'rmsprop']")


def build_model(model_type='rnn', activation='tanh', seq_len=100):
    model_type = model_type.lower()
    # depending on model type input, build the Sequential model
    if model_type == 'rnn':
        model = Sequential([
            Embedding(input_dim=10000, output_dim=100, input_length=seq_len),
            SimpleRNN(64, activation=activation, return_sequences=True),
            Dropout(0.3),
            SimpleRNN(64, activation=activation),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'lstm':
        model = Sequential([
            Embedding(input_dim=10000, output_dim=100, input_length=seq_len),
            LSTM(64, activation=activation, return_sequences=True),
            Dropout(0.3),
            LSTM(64, activation=activation),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'bilstm':
        model = Sequential([
            Embedding(input_dim=10000, output_dim=100, input_length=seq_len),
            Bidirectional(LSTM(64, activation=activation, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64, activation=activation)),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
    else:
        raise ValueError("model_type must be one of ['rnn', 'lstm', 'bilstm']")

    return model