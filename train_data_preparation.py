import string
import help_func as func

def clean_dsc(dsc):
    dsc = [word.lower() for word in dsc]
    dsc = [word for word in dsc if len(word)>1]
    dsc = ' '.join(dsc)
    # for i in (string.punctuation + string.digits):
    #     dsc = dsc.replace(i,'')
    for i in string.punctuation:
        dsc = dsc.replace(i,'')
    return dsc

# 保存txt文件
def save_descriptions(txt, filename):
    txt = txt.split('\n')
    descriptions = []
    for line in txt:
        line = line.split()
        if len(line)<2:
            continue
        img_id, img_dsc = line[0][:-6], line[1:]
        img_dsc = clean_dsc(img_dsc)
        descriptions.append(img_id + ' ' + img_dsc)
    data = '\n'.join(descriptions)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    return print('Descriptions saved!')

filename = 'G://MRI_imagecaptain//source_code//spine_for_IC_label//spine_label_cn.txt'
txt = func.load_doc(filename)
save_descriptions(txt, 'descriptions_cn.txt')