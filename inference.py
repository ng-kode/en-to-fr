from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
from preprocess import Data

LATENT_DIM = 256
model = load_model('model.100-0.23.h5')

# inference part
# encoder part
encoder_inputs = model.get_layer('encoder_inputs').input
_, encoder_h, encoder_c = model.get_layer('encoder_lstm').output
encoder_states = [encoder_h, encoder_c]
encoder_model = Model(encoder_inputs, encoder_states)

# decoder part
decoder_input_h = Input(shape=(LATENT_DIM,), name='decoder_input_h')
decoder_input_c = Input(shape=(LATENT_DIM,), name='decoder_input_c')
decoder_states_inputs = [decoder_input_h, decoder_input_c]
decoder_inputs = model.get_layer('decoder_inputs').input
decoder_lstm = model.get_layer('decoder_lstm')
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states_outputs = [state_h, state_c]
decoder_dense = model.get_layer('decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states_outputs
)

NUM_SAMPLES = 10000
data = Data(NUM_SAMPLES)
DECODER_NUM_TOKENS = data.DECODER_NUN_TOKENS
target_token_index = data.target_token_index
reverse_target_token_index = data.reverse_target_token_index
DECODER_MAXLEN = data.DECODER_MAXLEN

def decode_sequence(input_seq):
    states_values = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, DECODER_NUM_TOKENS))
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

        target_seq = np.zeros((1, 1, DECODER_NUM_TOKENS))
        target_seq[0, 0, last_char_idx] = 1

        states_values = [h, c]

    return decoded_sentence

for idx in range(100):
    input_seq = data.encoder_input_data[idx:idx+1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence: ', data.input_texts[idx])
    print('Decoded sentence: ', decoded_sentence)

