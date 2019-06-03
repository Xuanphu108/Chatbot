# Importing the libraries
import tensorflow as tf
from tensorflow.python.layers.core import Dense


class InputGenerate(object):
    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    # Creating placeholders for the inputs and the targets
    def ModelInputs(self):
        inputs = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='target')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return inputs, targets, lr, keep_prob

    # Preprocessing the targets
    def PreprocessTargets(self, targets, word2int):  # batch_size
        '''
        :param targets:
        :param word2int:
        :return: preprocess_targets matrix [batch_size, lengths_of_sequences_in_batch]
        tf.fill: will add SOS in start of the sequences in the batch
        tf.strided_slice: will remove the final word of the sequences in the batch
        '''
        left_side = tf.fill([self.batch_size, 1], word2int['<SOS>'])
        right_side = tf.strided_slice(targets, [0, 0], [self.batch_size, -1], [1, 1])
        preprocess_targets = tf.concat([left_side, right_side], 1)
        return preprocess_targets


class Seq2Seq(object):
    def __init__(self, batch_size=64, rnn_size=512, num_layers=3,
                 encoding_embedding_size=512, decoding_embedding_size=512):
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.encoding_embedding_size = encoding_embedding_size
        self.decoding_embedding_size = decoding_embedding_size
        #self.LSTMCell = tf.contrib.rnn.BasicLSTMCell(rnn_size)


    # Creating the Encoder RNN Layer
    def EncoderRNN(self, rnn_inputs, sequence_length_input, keep_prob):
        '''
        :param rnn_inputs: shape (batch_size, sequence_length, number_features = length_embedded_vector)
        :param rnn_size: the number of units in LSTM cell
        :param num_layer: the number of layers in LSTm cell
        :param keep_prob: dropout
        :param sequence_length: An int32/int 64 vector, containing the actual lengths for each of the sequences in the batch
        :return: the final encoder state
        '''
        # with tf.variable_scope("encoder"):
        def lstm_func_encoder(rnn_size):
            lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            # lstm = self.LSTMCell  # lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
            return lstm_dropout
        encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_func_encoder(self.rnn_size) for _ in range(self.num_layers)])
        # encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*self.num_layers)
        encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                                        cell_bw=encoder_cell,
                                                                        sequence_length=sequence_length_input,
                                                                        inputs=rnn_inputs,
                                                                        dtype=tf.float32)
        encoder_output = tf.concat(encoder_output, 2)

        return encoder_output, encoder_state

    # Decoding the training set
    def DecodeTrainingSet(self, decoder_embedded_input, sequence_length_output, max_length, attn_cell, initial_state, output_layer, decoding_scope):
        '''
        :param decoder_embedded_input:
        :param sequence_length_output:
        :param max_length:
        :param attn_cell:
        :param initial_state:
        :param output_layer:
        :param decoding_scope:
        :return:
        '''

        '''
        attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])

        attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                        attention_option='bahdanau',
                                                                                                                                        num_units=decoder_cell.output_size)
        training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                                attention_keys,
                                                                                attention_values,
                                                                                attention_score_function,
                                                                                attention_construct_function,
                                                                                name='attn_dec_train')
        decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                training_decoder_function,
                                                                                                                decoder_embedded_input,
                                                                                                                sequence_length,
                                                                                                                scope=decoding_scope)
        '''
        # train_output = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)
        # output_embed = tf.contrib.layers.embed_sequence(train_output, vocab_size=num_words, embed_dim=embed_dim, scope='embed', reuse=True)

        train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedded_input, sequence_length_output)  ################

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=attn_cell,  # out_cell
                                                  helper=train_helper,
                                                  initial_state=initial_state,  # out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
                                                  output_layer=output_layer)

        decoder_final_output, decoder_final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
                                                                                                              impute_finished=True, maximum_iterations=max_length, scope=decoding_scope)
        decoder_output_dropout = decoder_final_output  # tf.nn.dropout(decoder_final_output, keep_prob)

        return decoder_output_dropout  # output_function(decoder_output_dropout)


    # Decoding the test/ validation set
    def DecodeTestSet(self, decoder_embeddings_matrix, sos_id, eos_id, max_length, attn_cell, initial_state, output_layer, decoding_scope):
        '''
        :param decoder_embeddings_matrix:
        :param sos_id:
        :param eos_id:
        :param max_length:
        :param attn_cell:
        :param initial_state:
        :param output_layer:
        :param decoding_scope:
        :return:
        '''

        '''
        attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
        attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                        attention_option='bahdanau',
                                                                                                                                        num_units=decoder_cell.output_size)
        test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                                encoder_state[0],
                                                                                attention_keys,
                                                                                attention_values,
                                                                                attention_score_function,
                                                                                attention_construct_function,
                                                                                decoder_embeddings_matrix,
                                                                                sos_id,
                                                                                eos_id,
                                                                                maximum_length,
                                                                                num_words,
                                                                                name='attn_dec_inf')
        test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                    test_decoder_function,
                                                                                                                    scope=decoding_scope)
        '''
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings_matrix, tf.fill([self.batch_size], sos_id), eos_id) ################

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=attn_cell,
                                                  helper=pred_helper,
                                                  initial_state=initial_state,
                                                  output_layer=output_layer)

        decoder_final_output, decoder_final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
                                                                                                              impute_finished=True, maximum_iterations=max_length, scope=decoding_scope)

        decoder_output_dropout = decoder_final_output  # tf.nn.dropout(decoder_final_output, keep_prob)

        return decoder_output_dropout


    # Creating the Decoder RNN
    def DecoderRNN(self, encoder_output, decoder_embedded_input, decoder_embeddings_matrix, encoder_state,
                   num_words, sequence_length_input, sequence_length_output, max_length, word2int, keep_prob):
        '''
        :param encoder_output:
        :param decoder_embedded_input:
        :param decoder_embeddings_matrix:
        :param encoder_state:
        :param num_words:
        :param sequence_length_input:
        :param sequence_length_output:
        :param max_length:
        :param word2int:
        :param keep_prob:
        :return:
        '''

        def lstm_func_decoder(rnn_size):
            lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)

            # lstm = self.LSTMCell  # lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)

            lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
            return lstm_dropout

        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_func_decoder(self.rnn_size) for _ in range(self.num_layers)])
        # decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)

        '''
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                        num_words,
                                                                        None,
                                                                        scope=decoding_scope,
                                                                        weights_initializers=weights,
                                                                        biases_initializers=biases
                                                                        )
        '''
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=decoder_cell.output_size,  # self.rnn_size
                                                                   memory=encoder_output,
                                                                   memory_sequence_length=sequence_length_input)

        attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=decoder_cell.output_size)  # self.rnn_size or self.rnn_size/2

        attn_zero = attn_cell.zero_state(self.batch_size, tf.float32)

        attn_zero = attn_zero.clone(cell_state=encoder_state[0])

        ''' Errors here!
        initial_state = tf.contrib.seq2seq.AttentionWrapperState(cell_state=encoder_state[0],
                                                                 attention=attn_zero,
                                                                 time=0,
                                                                 alignments=None,
                                                                 alignment_history=())
        '''
        # out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, num_words)  #################
        output_layer = Dense(num_words, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        with tf.variable_scope("decoding") as decoding_scope:
            training_predictions = self.DecodeTrainingSet(decoder_embedded_input,
                                                          sequence_length_output,
                                                          max_length,
                                                          # output_function,
                                                          attn_cell,
                                                          attn_zero,  # initial_state,
                                                          output_layer,
                                                          decoding_scope)

        with tf.variable_scope("decoding", reuse=True) as decoding_scope:
        # decoding_scope.reuse_variables()
            test_predictions = self.DecodeTestSet(decoder_embeddings_matrix,
                                                  word2int['<SOS>'],
                                                  word2int['<EOS>'],
                                                  max_length,
                                                  # output_function
                                                  attn_cell,
                                                  attn_zero,  # initial_state,
                                                  output_layer,
                                                  decoding_scope)
        return training_predictions, test_predictions


    # Building the seq2seq model
    def Seq2SeqModel(self, inputs, targets, keep_prob, sequence_length_input, sequence_length_output, max_length,
                     answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, questionswords2int):
        '''
        :param inputs:
        :param targets:
        :param keep_prob:
        :param sequence_length_input:
        :param sequence_length_output:
        :param max_length:
        :param answers_num_words:
        :param questions_num_words:
        :param encoder_embedding_size:
        :param decoder_embedding_size:
        :param questionswords2int:
        :return:
        '''

        encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                                  answers_num_words,
                                                                  encoder_embedding_size,
                                                                  initializer=tf.random_uniform_initializer(0, 1))

        encoder_output, encoder_state = self.EncoderRNN(encoder_embedded_input, sequence_length_input, keep_prob)

        preprocessed_targets = InputGenerate(self.batch_size).PreprocessTargets(targets, questionswords2int)

        decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words, decoder_embedding_size], 0, 1))

        decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)

        training_predictions, test_predictions = self.DecoderRNN(encoder_output,
                                                                 decoder_embedded_input,
                                                                 decoder_embeddings_matrix,
                                                                 encoder_state,
                                                                 questions_num_words,
                                                                 sequence_length_input,
                                                                 sequence_length_output,
                                                                 max_length,
                                                                 questionswords2int,
                                                                 keep_prob)
        return training_predictions, test_predictions









