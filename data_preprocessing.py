# Importing the libraries
import re

# Adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

class TextFile(object):
    def __init__(self, file_name_1, file_name_2):
        self.file_name_1 = file_name_1
        self.file_name_2 = file_name_2

    def OpenFile(self):
        # Importing the dataset
        lines = open(self.file_name_1, encoding='utf-8', errors='ignore').read().split('\n') # 'movie_lines.txt'
        conversations = open(self.file_name_2, encoding='utf-8', errors='ignore').read().split('\n') # 'movie_conversations.txt'
        return lines, conversations

    # Creating a dictionary that maps each line and its id
    def MapLine2Ids(self, lines, conversations):
        # lines, conversations = text_file.open_file(self)
        id2line = {}
        for line in lines:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]

        # Creating a list of all of the conversations
        conversations_ids = []
        for conversation in conversations[:-1]:
            _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", ""). replace(" ", "")
            conversations_ids.append(_conversation.split(','))
        return id2line, conversations_ids

    # Getting separately the questions and the answer
    def SeparateAns2Ques(self, id2line, conversations_ids):
        questions = []
        answers = []
        # id2line, conversations_ids = self.map_line2ids()
        for conversation in conversations_ids:
            for i in range(len(conversation) - 1):
                questions.append(id2line[conversation[i]])
                answers.append(id2line[conversation[i + 1]])
        return questions, answers

    # Doing a first cleaning of the texts
    @staticmethod
    def CleanText(text):
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"haven't", "have not", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"isn't", "is not", text)
        text = re.sub(r"aren't", "are not", text)
        text = re.sub(r"don't", "do not", text)
        text = re.sub(r"doesn't", "does not", text)
        text = re.sub(r"[-()\"#/@;:<>{}+=|.!?,]", "", text)
        return text

    # Cleaning the sequences
    def CleanSeq(self, questions, answers):
        # questions, answers = text_file.separate_ans2ques(self)
        # Cleaning the questions
        clean_questions = []
        for question in questions:
            clean_questions.append(TextFile.CleanText(question))
        # Cleaning the answers
        clean_answers = []
        for answer in answers:
            clean_answers.append(TextFile.CleanText(answer))
        return clean_questions, clean_answers


class Vocab(object):
    def __init__(self, threshold=20):
        self.threshold = threshold
        self.word2count = {}
        self.questionswords2int = {}
        self.answerswords2int ={}
        self.questions_into_int = []
        self.answers_into_int = []
        self.sorted_clean_questions = []
        self.sorted_clean_answers = []

    # Creating a dictionary that maps each word to its number of occurences
    def Words2Occur(self, clean_questions, clean_answers):
        # word2count = {}
        for question in clean_questions:
            for word in question.split():
                if word not in self.word2count:
                    self.word2count[word] = 1
                else:
                    self.word2count[word] += 1

        for answer in clean_answers:
            for word in answer.split():
                if word not in self.word2count:
                    self.word2count[word] = 1
                else:
                    self.word2count[word] += 1
        return self.word2count


    # Creating two dictionaries that map the questions words and the answers words to a unique integer
    def Words2IndexDic(self, word2count):
        # threshold = 20
        # questionswords2int = {}
        word_number = 0
        for word, count in word2count.items():
            if count >= self.threshold:
                self.questionswords2int[word] = word_number
                word_number += 1

        # answerswords2int = {}
        word_number = 0
        for word, count in word2count.items():
            if count >= self.threshold:
                self.answerswords2int[word] = word_number
                word_number += 1
        return self.questionswords2int, self.answerswords2int

    def AddTokens(self):
        for token in tokens:
            self.questionswords2int[token] = len(self.questionswords2int)

        for token in tokens:
            self.answerswords2int[token] = len(self.answerswords2int)
        return self.questionswords2int, self.answerswords2int


    # Creating the inverse dictionary of the answerwords2int dictionary
    def InverseAnswerWords2IntDic(self):
        self.answersints2word = {w_i: w for w, w_i in self.answerswords2int.items()}
        return self.answersints2word

    # Adding the End of String token to the end of every answer
    def AddEos(self, clean_answers):
        for i in range(len(clean_answers)):
            clean_answers[i] += ' <EOS>'
        return clean_answers


    # Translating all the questions and the answers into integers
    # and Replacing all the words that were filtered out by <OUT>
    def Words2Int(self, clean_questions, clean_answers):
        # questions_into_int = []
        for question in clean_questions:
            ints = []
            for word in question.split():
                if word not in self.questionswords2int:
                    ints.append(self.questionswords2int['<OUT>'])
                else:
                    ints.append(self.questionswords2int[word])
            self.questions_into_int.append(ints)

        # answers_into_int = []
        for answer in clean_answers:
            ints = []
            for word in answer.split():
                if word not in self.answerswords2int:
                    ints.append(self.answerswords2int['<OUT>'])
                else:
                    ints.append(self.answerswords2int[word])
            self.answers_into_int.append(ints)
        return self.questions_into_int, self.answers_into_int


    # Sorting questions and answers by the length of questions
    def SortedSequences(self, questions_into_int, answers_into_int):
        # sorted_clean_questions = []
        # sorted_clean_answers = []
        for length in range(1, 25 + 1):
            for i in enumerate(questions_into_int):
                if len(i[1]) == length:
                    self.sorted_clean_questions.append(questions_into_int[i[0]])
                    self.sorted_clean_answers.append(answers_into_int[i[0]])
        return self.sorted_clean_questions, self.sorted_clean_answers

if __name__ == '__main__':
    txt = TextFile('movie_lines.txt', 'movie_conversations.txt')
    lines, conversations = txt.OpenFile()
    id2line, conversations_ids = txt.MapLine2Ids(lines, conversations)
    questions, answers = txt.SeparateAns2Ques(id2line, conversations_ids)
    clean_questions, clean_answers = txt.CleanSeq(questions, answers)

    vocab = Vocab(20)
    word2count = vocab.Words2Occur(clean_questions, clean_answers)
    questionswords2int, answerswords2int = vocab.Words2IndexDic(word2count)
    questionswords2int, answerswords2int = vocab.AddTokens()
    print(answerswords2int)
    answersints2word = vocab.InverseAnswerWords2IntDic()
    print(answersints2word)
    clean_answers = vocab.AddEos(clean_answers)
    questions_into_int, answers_into_int = vocab.Words2Int(clean_questions, clean_answers)
    sorted_clean_questions, sorted_clean_answers = vocab.SortedSequences(questions_into_int, answers_into_int)












