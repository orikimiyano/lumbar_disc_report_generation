from numpy import array
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
import help_func as func
import tensorflow as tf
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
import sys
import os

# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)
NameOfModel = 'cn'


class Logger(object):
    def __init__(self, filename=str(NameOfModel) + ".log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('log/history_' + str(NameOfModel) + '.txt')


def create_sequences(tokenizer, max_length, dsc_list, photos, vocab_size):
    X1, X2, y = [], [], []
    for line in dsc_list:
        seq = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photos)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


EMBEDDING_DIM = 128
lstm_layers = 3
dropout_rate = 0.22
learning_rate = 0.001


def define_model(vocab_size, max_length):
    # 图像特征
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(dropout_rate)(inputs1)
    fe2 = Dense(EMBEDDING_DIM, activation='relu')(fe1)
    fe3 = RepeatVector(max_length)(fe2)

    # embedding
    inputs2 = Input(shape=(max_length,))
    emb2 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(inputs2)
    emb2 = BatchNormalization()(emb2)
    emb2 = GRU(EMBEDDING_DIM, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)(emb2)
    emb2 = GRU(EMBEDDING_DIM, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)(emb2)

    # merge inputs
    merged = concatenate([fe3, emb2])
    # language model (decoder)
    GRU1 = BatchNormalization()(merged)
    GRU1 = GRU(EMBEDDING_DIM, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)(GRU1)
    decoder1 = Dense(EMBEDDING_DIM, activation='relu')(GRU1)
    flatten1 = Lambda(lambda x: x, output_shape=lambda s: s)(decoder1)
    flatten1 = Flatten()(flatten1)
    outputs = Dense(vocab_size, activation='softmax')(flatten1)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adadelta(),
        metrics=['accuracy'])
    model.summary()
    return model
F

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    while True:  # 只要调用该函数，就源源不断生成数据，所以这是一个死循环
        for img_id, dsc_list in descriptions.items():
            photo = photos[img_id][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, dsc_list, photo, vocab_size)
            yield [[in_img, in_seq], out_word]


filename = 'trainImages.txt'
train = func.load_set(filename)
print('Namelist of train data：%d' % len(train))
train_descriptions = func.load_descriptions('descriptions_' + str(NameOfModel) + '.txt', train)
print('Descriptions of train data：%d' % len(train_descriptions))
train_features = func.load_photo_features('features.pkl', train)
print('Photo features of train data：%d' % len(train_features))
tokenizer = func.create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size：%d' % vocab_size)
max_length = func.max_length(train_descriptions)
print('Description Length：%d' % max_length)

model = define_model(vocab_size, max_length)
epochs = 20
steps = len(train_descriptions)

generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
model_checkpoint = ModelCheckpoint('saved_models/' + str(NameOfModel) + '.hdf5', monitor='loss', verbose=1,
                                   save_best_only=True)
history = model.fit_generator(generator,
                              steps_per_epoch=1000,
                              epochs=20,
                              verbose=1,
                              callbacks=[model_checkpoint])
