

def build_dict():
    f2_lines = open('OpenSubData/movie_25000', 'r').readlines()
    word2id = {}
    id2word = {}
    for idx, line in enumerate(f2_lines):
        id2word[idx+1] = line.strip()
        word2id[line.strip()] = idx+1

    id2word[0] = '<pad>'
    word2id['<pad>'] = 0
    n = len(id2word)
    id2word[n] = '<s>'
    word2id['<s>'] = n
    n += 1
    id2word[n] = '</s>'
    word2id['</s>'] = n
    return word2id, id2word


def convert_to_text(f_name):
    word2id, id2word = build_dict()
    # f = open('OpenSubData/' + f_name + '.csv', 'r').readlines()
    import pandas as pd
    df = pd.read_csv('OpenSubData/' + f_name + '.csv')
    f_write = open('OpenSubData/' + f_name + '.text', 'w')
    # text = []
    for bar in df['target']:
        # foo, bar = line.split('|')
        # foo = [id2word[int(x)] for x in foo.split()]
        bar = ' '.join([id2word[int(x)] for x in bar.split()])
        # text.append((foo, bar))
        f_write.write(bar + '\n')
    #return text


def convert_to_csv():
    split_ratio = 0.9
    f = open('OpenSubData/s_given_t_dialogue_length2_3.txt', 'r').readlines()
    s_train = []
    t_train = []
    s_test = []
    t_test = []
    num_line = len(f)
    for idx, line in enumerate(f):
        foo, bar = line.split('|')
        if idx < num_line * split_ratio:
            s_train.append(foo.strip())
            t_train.append(bar.strip())
        else:
            s_test.append(foo.strip())
            t_test.append(bar.strip())
    import pandas as pd
    df = pd.DataFrame({'source': s_train,
                       'target': t_train})
    df.to_csv('OpenSubData/data_2_train.csv')
    df = pd.DataFrame({'source': s_test,
                       'target': t_test})
    df.to_csv('OpenSubData/data_2_test.csv')

    f = open('OpenSubData/s_given_t_dialogue_length2_6.txt', 'r').readlines()
    s_train = []
    t_train = []
    s_test = []
    t_test = []
    num_line = len(f)
    for idx, line in enumerate(f):
        foo, bar = line.split('|')
        if idx < num_line * split_ratio:
            s_train.append(foo.strip())
            t_train.append(bar.strip())
        else:
            s_test.append(foo.strip())
            t_test.append(bar.strip())
    import pandas as pd
    df = pd.DataFrame({'source': s_train,
                       'target': t_train})
    df.to_csv('OpenSubData/data_6_train.csv')
    df = pd.DataFrame({'source': s_test,
                       'target': t_test})
    df.to_csv('OpenSubData/data_6_test.csv')


if __name__ == '__main__':
    import os

    if not os.path.exists('OpenSubData'):
        os.makedirs('OpenSubData')

    import urllib.request, tarfile
    url = 'https://nlp.stanford.edu/data/OpenSubData.tar'
    urllib.request.urlretrieve(url, 'OpenSubData/OpenSubData.tar')
    url2 = 'https://raw.githubusercontent.com/jiweil/Neural-Dialogue-Generation/master/data/movie_25000'
    urllib.request.urlretrieve(url2, 'OpenSubData/movie_25000')

    tar = tarfile.open('OpenSubData/OpenSubData.tar')
    tar.extractall()
    tar.close()
    convert_to_csv()
    convert_to_text('data_2_train')
    convert_to_text('data_2_test')
    convert_to_text('data_6_train')
    convert_to_text('data_6_test')
    import pickle
    word2id, id2word = build_dict()
    with open('OpenSubData/word_dict.pkl', 'bw') as f:
        pickle.dump((word2id, id2word), f)


