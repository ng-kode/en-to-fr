from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
encoder_input_data = None
decoder_input_data = None
decoder_target_data = None
target_token_index = {}
reverse_target_token_index = {}

# hyper-parameters
ENCODER_NUM_TOKENS = 10000
LATENT_DIM = 300
DECODER_NUN_TOKENS = 8000
BATCH_SIZE = 32
EPOCHS = 10
DECODER_MAXLEN = 10

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

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# Train the model
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.2)

# inference part
# encoder part
encoder_model = Model(encoder_inputs, encoder_states)

# decoder part
decoder_input_h = Input(shape=(None, LATENT_DIM))
decoder_input_c = Input(shape=(None, LATENT_DIM))
decoder_states_inputs = [decoder_input_h, decoder_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states_outputs = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states_outputs
)

def decoder_sequence(input_seq):
    states_values = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, DECODER_NUN_TOKENS))
    target_seq[0, 0, target_token_index['\t']] = 1

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        ouputs, h, c = decoder_model.predict([target_seq] + states_values)
        
        last_char_idx = np.argmax(ouputs[0, -1, :])
        last_char = reverse_target_token_index[last_char_idx]
        decoded_sentence += last_char

        if last_char == '\n' or len(decoded_sentence) > DECODER_MAXLEN:
            stop_condition = True

        target_seq = np.zeros((1, 1, DECODER_NUN_TOKENS))
        target_seq[0, 0, last_char_idx] = 1

        states_values = [h, c]

    return decoded_sentence