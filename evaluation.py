import numpy as np
from numpy import argmax
from pickle import load, dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model, Sequential
from nltk.translate.bleu_score import corpus_bleu
import nltk
import help_func as func
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from rouge import Rouge

NameOfModel = 'cn'

if NameOfModel == 'en':
    normal = 'normal'
    local_num = 0
    inter_num = 4
elif NameOfModel == 'cn':
    normal = '常'


def word_for_id(integer, tokenizer):
    for word, word_id in tokenizer.word_index.items():
        if word_id == integer:
            return word
    return None


def generate_dsc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        input_seq = tokenizer.texts_to_sequences([in_text])[0]
        input_seq = pad_sequences([input_seq], maxlen=max_length)
        output_seq = model.predict([photo, input_seq], verbose=0)
        output_int = argmax(output_seq)
        output_word = word_for_id(output_int, tokenizer)
        if output_word == None:
            break
        in_text = in_text + ' ' + output_word
        if output_word == 'endseq':
            break
    return in_text


def rouge_score(pre_text, org_text):
    rouge = Rouge()
    return rouge.get_scores(pre_text, org_text)


def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    y_tag, y_pdc = [], []
    for img_id, dsc_list in descriptions.items():
        yhat = generate_dsc(model, tokenizer, photos[img_id], max_length)
        references = [d.split() for d in dsc_list]
        y_tag.append(references)
        y_pdc.append(yhat.split())
    print('BLEU-1: %f' % corpus_bleu(y_tag, y_pdc, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(y_tag, y_pdc, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(y_tag, y_pdc, weights=(0.34, 0.33, 0.33, 0)))
    print('BLEU-4: %f' % corpus_bleu(y_tag, y_pdc, weights=(0.25, 0.25, 0.25, 0.25)))

    hypothesis = np.array(y_pdc)
    reference = np.array(y_tag)
    num = hypothesis.shape
    merteor_score_all = 0.
    rouge_score_all = 0.

    acc_l1l2 = 0.
    acc_l2l3 = 0.
    acc_l3l4 = 0.
    acc_l4l5 = 0.
    acc_l5s1 = 0.

    acc_l1l2_p = 0.
    acc_l2l3_p = 0.
    acc_l3l4_p = 0.
    acc_l4l5_p = 0.
    acc_l5s1_p = 0.

    num_l1l2_p = 0.
    num_l2l3_p = 0.
    num_l3l4_p = 0.
    num_l4l5_p = 0.
    num_l5s1_p = 0.

    for count in range(num[0]):
        refer_rem = list(reference[count][0])
        refer_rem.remove('startseq')
        refer_rem.remove('endseq')
        hypo_rem = list(hypothesis[count])
        hypo_rem.remove('startseq')
        hypo_rem.remove('endseq')
        merteor_score_single = nltk.translate.meteor_score.single_meteor_score(refer_rem, hypo_rem)
        merteor_score_all = merteor_score_all + merteor_score_single

        print('判断语句'+str(count))
        print(hypo_rem)
        print('参考语句'+str(count))
        print(refer_rem)
        hypo_str = ' '.join(list(hypo_rem))
        refer_str = ' '.join(list(refer_rem))
        rouge_score_str = rouge_score(hypo_str, refer_str)
        # print(rouge_score_str)
        str_rouge = str(rouge_score_str[0])
        strlist_1 = str_rouge.split(',')
        strlist_2 = strlist_1[3].split(':')
        rouge_score_single = float(strlist_2[-1])
        rouge_score_all = rouge_score_all + rouge_score_single

        # 准确率统计
        if NameOfModel == 'cn':
            if refer_rem[0][-1] != normal:
                num_l1l2_p += 1
            if refer_rem[1][-1] != normal:
                num_l2l3_p += 1
            if refer_rem[2][-1] != normal:
                num_l3l4_p += 1
            if refer_rem[3][-1] != normal:
                num_l4l5_p += 1
            if refer_rem[4][-1] != normal:
                num_l5s1_p += 1

            if hypo_rem[0][-1] == refer_rem[0][-1]:
                acc_l1l2 += 1
                if refer_rem[0][-1] != normal:
                    acc_l1l2_p += 1
            if hypo_rem[1][-1] == refer_rem[1][-1]:
                acc_l2l3 += 1
                if refer_rem[1][-1] != normal:
                    acc_l2l3_p += 1
            if hypo_rem[2][-1] == refer_rem[2][-1]:
                acc_l3l4 += 1
                if refer_rem[2][-1] != normal:
                    acc_l3l4_p += 1
            if hypo_rem[3][-1] == refer_rem[3][-1]:
                acc_l4l5 += 1
                if refer_rem[3][-1] != normal:
                    acc_l4l5_p += 1
            if hypo_rem[4][-1] == refer_rem[4][-1]:
                acc_l5s1 += 1
                if refer_rem[4][-1] != normal:
                    acc_l5s1_p += 1

    acc_l1l2 = acc_l1l2 / num[0]
    acc_l2l3 = acc_l2l3 / num[0]
    acc_l3l4 = acc_l3l4 / num[0]
    acc_l4l5 = acc_l4l5 / num[0]
    acc_l5s1 = acc_l5s1 / num[0]

    acc_all = (acc_l1l2 + acc_l2l3 + acc_l3l4 + acc_l4l5 + acc_l5s1) / 5

    acc_l1l2_p = acc_l1l2_p / num_l1l2_p
    acc_l2l3_p = acc_l2l3_p / num_l2l3_p
    acc_l3l4_p = acc_l3l4_p / num_l3l4_p
    acc_l4l5_p = acc_l4l5_p / num_l4l5_p
    acc_l5s1_p = acc_l5s1_p / num_l5s1_p

    acc_all_p = (acc_l1l2_p + acc_l2l3_p + acc_l3l4_p + acc_l4l5_p + acc_l5s1_p) / 5

    final_merteor_score = merteor_score_all / num[0]
    final_rouge_score = rouge_score_all / num[0]
    print('METEOR: %f' % final_merteor_score)

    print('ROUGE-2: %f' % final_rouge_score)
    # print('CIDEr: %f' % cider(y_pdc, y_tag))
    # print('SPICE: %f' % spice(y_pdc, y_tag))
    print('Abnormal Classification Rate:' + str(acc_all))
    print('Classification Accuracy Rates of Abnormal Discs:' + str(acc_all_p))

    return None


def extract_features(filename):
    model = Sequential()
    for layer in VGG16().layers[:-1]:
        model.add(layer)
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature = model.predict(img, verbose=0)
    return feature


filename = 'trainImages.txt'
train = func.load_set(filename)
print('Namelist of train data：%d' % len(train))
train_descriptions = func.load_descriptions('descriptions_' + str(NameOfModel) + '.txt', train)
print('Descriptions of train data：%d' % len(train_descriptions))
tokenizer = func.create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size：%d' % vocab_size)
max_length = func.max_length(train_descriptions)
print('Description Length：%d' % max_length)
dump(tokenizer, open('tokenizer.pkl', 'wb'))

filename = 'testImages.txt'
test = func.load_set(filename)
print('Namelist of test data：%d' % len(test))
test_descriptions = func.load_descriptions('descriptions_' + str(NameOfModel) + '.txt', test)
print('Descriptions of test data：%d' % len(test_descriptions))
test_features = func.load_photo_features('features.pkl', test)
print('Photo features of test data：%d' % len(test_features))

filename = 'saved_models/' + str(NameOfModel) + '/model_' + str(NameOfModel) + '_49.h5'

model = load_model(filename)
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

photo = extract_features('Patient_268.jpg')
description = generate_dsc(model, tokenizer, photo, max_length)
print(description[9:-7])
