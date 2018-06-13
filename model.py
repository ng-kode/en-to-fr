from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
from preprocess import Data

# preprocess data
NUM_SAMPLES = 10000
data = Data(NUM_SAMPLES)

encoder_input_data = data.encoder_input_data
decoder_input_data = data.decoder_input_data
decoder_target_data = data.decoder_target_data
target_token_index = data.target_token_index
reverse_target_token_index = data.target_token_index

# hyper-parameters
ENCODER_NUM_TOKENS = data.ENCODER_NUM_TOKENS
LATENT_DIM = 256
DECODER_NUN_TOKENS = data.DECODER_NUN_TOKENS
BATCH_SIZE = 64
EPOCHS = 100
DECODER_MAXLEN = data.DECODER_MAXLEN

# encoder part
encoder_inputs = Input(shape=(None, ENCODER_NUM_TOKENS))
encoder_lstm = LSTM(LATENT_DIM, activation='tanh', return_state=True)
_, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# decoder part
decoder_inputs = Input(shape=(None, DECODER_NUN_TOKENS))
decoder_lstm = LSTM(LATENT_DIM, activation='tanh', return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(DECODER_NUN_TOKENS, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# bridge the seq2seqModel
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# get summary
model.summary()

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# Train the model
checkpointer = ModelCheckpoint(filepath='model.{epoch:02d}-{val_acc:.2f}.h5', verbose=1)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.2,
            callbacks=[checkpointer])