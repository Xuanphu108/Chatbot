# Importing the libraries
import tensorflow as tf
import numpy as np
from datetime import datetime
import time
from seq2seq import InputGenerate, Seq2Seq
from data_preprocessing import TextFile, Vocab

# Setting the Hyperparameters
epochs = 100
batch_size = 1  # 64
rnn_size = 32  # 512
num_layers = 1
encoding_embedding_size = 32  # 512
decoding_embedding_size = 32  # 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5


# Defining a session
tf.reset_default_graph()
#session = tf.InteractiveSession()
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# root_logdir = "tf_logs"
# logdir = "{}/run-{}/".format(root_logdir, now)


# Loading the model inputs
inputs, targets, lr, keep_prob = InputGenerate(batch_size).ModelInputs()  # lr, keep_prob

# Setting the sequence length
sequence_length_input = tf.placeholder(tf.int32, [batch_size], name='sequence_length_input')

sequence_length_output = tf.placeholder(tf.int32, [batch_size], name='sequence_length_output')

# max_length = 25
max_length = tf.placeholder(tf.int32, (), name='max_length')

# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)

txt = TextFile('movie_lines.txt', 'movie_conversations.txt')
lines, conversations = txt.OpenFile()
id2line, conversations_ids = txt.MapLine2Ids(lines, conversations)
questions, answers = txt.SeparateAns2Ques(id2line, conversations_ids)
clean_questions, clean_answers = txt.CleanSeq(questions, answers)
vocab = Vocab(20)
word2count = vocab.Words2Occur(clean_questions, clean_answers)
questionswords2int, answerswords2int = vocab.Words2IndexDic(word2count)
questionswords2int, answerswords2int = vocab.AddTokens()
answersints2word = vocab.InverseAnswerWords2IntDic()
clean_answers = vocab.AddEos(clean_answers)
questions_into_int, answers_into_int = vocab.Words2Int(clean_questions, clean_answers)
sorted_clean_questions, sorted_clean_answers = vocab.SortedSequences(questions_into_int, answers_into_int)

# Getting the training and test prediction
Seq2Seq = Seq2Seq(batch_size, rnn_size, num_layers, encoding_embedding_size, decoding_embedding_size)

training_predictions, test_predictions = Seq2Seq.Seq2SeqModel(tf.reverse(inputs, [-1]),
                                                              targets,
                                                              keep_prob,
                                                              sequence_length_input,
                                                              sequence_length_output,
                                                              max_length,
                                                              len(answerswords2int),
                                                              len(questionswords2int),
                                                              encoding_embedding_size,
                                                              decoding_embedding_size,
                                                              questionswords2int)

# Setting up the Loss Error, the Optimizer and Gradient Clipping
#output_maxlen = tf.minimum(tf.shape(training_predictions.rnn_output)[1], max_length)
#out_data_slice = tf.slice(targets, [0, 0], [-1, output_maxlen])
#out_logits_slice = tf.slice(training_predictions.rnn_output, [0, 0, 0], [-1, output_maxlen, -1])
with tf.name_scope("optimization"):
    '''
    length_mask = tf.sequence_mask(sequence_length_output, maxlen=output_maxlen, dtype=tf.float32)
    loss_error = tf.contrib.seq2seq.sequence_loss(out_logits_slice,  # training_predictions
                                                    out_data_slice,  # targets
                                                    weights=length_mask)  # tf.ones([input_shape[0], max_length])
    '''
    start_tokens = tf.zeros([batch_size], dtype=tf.int32)
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), targets], 1)
    tf.identity(training_predictions.sample_id[0])
    weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))
    loss_error = tf.contrib.seq2seq.sequence_loss(
            training_predictions.rnn_output, targets, weights=weights)
    tf.summary.scalar("loss_error", loss_error)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)


# Padding the sequences with the <PAD> token
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']]*(max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index*batch_size
        questions_in_batch = questions[start_index:start_index + batch_size]
        answers_in_batch = answers[start_index:start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch


# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions)*0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"


log_path = '/logs/plot_training' + '/train_{}'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S"))  # ./logs/plot_1
session.run(tf.global_variables_initializer())
loss_error_writer = tf.summary.merge_all()
writer_training = tf.summary.FileWriter(log_path, graph_def=session.graph_def)
# writer_validation = tf.summary.FileWriter(" ", graph_def=session.graph_def)

for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length_input: np.ones((batch_size), dtype=int)*padded_questions_in_batch.shape[1],
                                                                                               sequence_length_output: np.ones((batch_size), dtype=int)*padded_answers_in_batch.shape[1],
                                                                                               max_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
                                                                                               
        total_training_loss_error += batch_training_loss_error

        total_training_loss_error_writer = session.run(loss_error_writer, {inputs: padded_questions_in_batch,
                                                                            targets: padded_answers_in_batch,
                                                                            lr: learning_rate,
                                                                            sequence_length_input: np.ones((batch_size), dtype=int) *padded_questions_in_batch.shape[1],
                                                                            sequence_length_output: np.ones((batch_size), dtype=int) *padded_answers_in_batch.shape[1],
                                                                            max_length: padded_answers_in_batch.shape[1],
                                                                            keep_prob: keep_probability})
        writer_training.add_summary(total_training_loss_error_writer, epoch*len(training_questions) + batch_index)


        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:f} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error =0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error,
                                                          {inputs: padded_questions_in_batch,
                                                           targets: padded_answers_in_batch,
                                                           lr: learning_rate,
                                                           sequence_length_input: np.ones((batch_size), dtype=int)*padded_questions_in_batch.shape[1],
                                                           sequence_length_output: np.ones((batch_size), dtype=int)*padded_answers_in_batch.shape[1],
                                                           max_length: padded_answers_in_batch.shape[1],
                                                           keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")






