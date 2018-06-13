import numpy as np

class Data():
    def __init__(self, NUM_SAMPLES, random=False):
        self.NUM_SAMPLES = NUM_SAMPLES
        self.input_texts = []
        self.target_texts = []
        self.ENCODER_NUM_TOKENS = None
        self.DECODER_NUN_TOKENS = None
        self.ENCODER_MAXLEN = None
        self.DECODER_MAXLEN = None
        
        self.input_token_index = {}
        self.reverse_input_token_index = {}
        self.target_token_index = {}
        self.reverse_target_token_index = {}

        self.encoder_input_data = None
        self.decoder_input_data = None
        self.decoder_target_data = None
    
        # Prepare maxlen, num_tokens, token_index
        lines = None
        with open('fra.txt', 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')
        
        input_chars = set()
        target_chars = set()
        for line in lines[: min(NUM_SAMPLES, len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in input_chars:
                    input_chars.add(char)
            for char in target_text:
                if char not in target_chars:
                    target_chars.add(char)

        input_chars = sorted(list(input_chars))
        target_chars = sorted(list(target_chars))

        self.ENCODER_NUM_TOKENS = len(input_chars)
        self.DECODER_NUN_TOKENS = len(target_chars)

        self.ENCODER_MAXLEN = max(len(seq) for seq in self.input_texts)
        self.DECODER_MAXLEN = max(len(seq) for seq in self.target_texts)

        print('NUM_SAMPLES: {} \n'.format(self.NUM_SAMPLES))
        print('ENCODER_NUM_TOKENS: {}, ENCODER_MAXLEN: {} \n'.format(self.ENCODER_NUM_TOKENS, self.ENCODER_MAXLEN))
        print('DECODER_NUN_TOKENS: {}, DECODER_MAXLEN: {} \n'.format(self.DECODER_NUN_TOKENS, self.DECODER_MAXLEN))

        self.input_token_index = dict((c, i) for i, c in enumerate(input_chars))
        self.reverse_input_token_index = dict((v, k) for k, v in self.input_token_index.items())
        self.target_token_index = dict((c, i) for i, c in enumerate(target_chars))
        self.reverse_target_token_index = dict((v, k) for k, v in self.target_token_index.items())

        del input_chars, target_chars

        # Prepare encoder input, decoder input, decoder target
        self.encoder_input_data = np.zeros((len(self.input_texts), self.ENCODER_MAXLEN, self.ENCODER_NUM_TOKENS), dtype='float32')
        self.decoder_input_data = np.zeros((len(self.target_texts), self.DECODER_MAXLEN, self.DECODER_NUN_TOKENS), dtype='float32')
        self.decoder_target_data = np.zeros((len(self.target_texts), self.DECODER_MAXLEN, self.DECODER_NUN_TOKENS), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.
                # decoder target is one timestep ahead
                # i.e. input's curr step = target's prev step
                if t > 0:
                    self.decoder_target_data[i, t-1, self.target_token_index[char]] = 1.

            