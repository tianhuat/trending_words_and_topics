import yaml
import logging
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict, Counter
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from sklearn.decomposition import LatentDirichletAllocation
from scipy import sparse
from datetime import datetime
from textblob import TextBlob
from dateutil.parser import parse
import math
import re
level = logging.INFO
format = '  %(message)s'
handlers = [logging.FileHandler('info.log'), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)


target = 'STORY_DATE_TIME'
combine = 'COMBINE'


def load_data(input):
    return pd.read_csv(input, parse_dates=[target])


def clear_sentence(sentence):
    sent = sentence
    if not isinstance(sent, str):
        sent = ''
    # normalize
    sent = sent.lower()
    # tokenize/lemmatize
    word_blob = TextBlob(sent)
    tokens = [w.lemmatize() for w in word_blob.words]
    return " ".join(tokens)


def preprocess(df):
    res = {target: [], combine: []}
    story_index = defaultdict(list)
    for index, row in df.iterrows():
        # remove test message and delete entries
        if row['PRODUCTS'] != 'TEST' and row['EVENT_TYPE'] != 'DELETE':
            story_index[row["UNIQUE_STORY_INDEX"]].append((row["EVENT_TYPE"], row, index))

    token_cnt = 0
    token_set = set()

    # To reduce noise + improve efficiency, we use only event = "STORY_TAKE_OVERWRITE" and "ALERT" entry
    for key, values in story_index.items():
        story_take = None
        is_story_take = False
        alert = None
        for (event, row, index) in values:
            if event == "STORY_TAKE_OVERWRITE":
                story_take = row
                is_story_take = True
            elif event == "ALERT":
                alert = row
        
        if is_story_take:
            res[target].append(story_take[target])
            tokens=clear_sentence(story_take["HEADLINE_ALERT_TEXT"]) + " " + clear_sentence(story_take["TAKE_TEXT"])
        else:
            res[target].append(alert[target])
            tokens=clear_sentence(alert["HEADLINE_ALERT_TEXT"])
        
        res[combine].append(tokens)
        token_set.update(tokens.split(" "))
        token_cnt+=len(tokens)

    logging.info("Total sentences after preprocessing: " + str(len(story_index)))
    logging.info("Total tokens after preprocessing: " + str(token_cnt))
    logging.info("Total unique tokens after preprocessing: " + str(len(token_set)))

    df = pd.DataFrame(res)
    return df


def generate_lda(train_data, num_of_topics, num_of_word_per_topic, lda_iteration):
    # display the top K words in each topic
    def print_top_words(model, feature_names, n_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            topics.append(message)

        with open(r"topic.txt", "w", encoding="utf-8") as f:
            f.write('\n'.join(topics))

    dictionary = Dictionary(train_data)

    # remove the high frequency words appear in more than half documents
    dictionary.filter_extremes(no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in train_data]
    lda = LatentDirichletAllocation(n_components=num_of_topics, n_jobs=-1, max_iter=lda_iteration,
                                    learning_method='online')
    M = sparse.lil_matrix((len(corpus), len(dictionary)), dtype=int)
    for i, row in enumerate(corpus):
        for col in row:
            M[i, col[0]] = col[1]
    lda.fit(M)
    temp = dictionary[0]  # this is just to lead the dictionary
    id2word = dictionary.id2token
    print_top_words(lda, id2word, num_of_word_per_topic)

def is_date(token):
    try:
        parse(token)
        return True
    except:
        return False


def filter_stop_word(sentence_list):
    word_cnt = defaultdict(int)
    new_sent_list = []
    for sentence in sentence_list:
        if isinstance(sentence, str):
            split_sentence = sentence.split(" ")
            new_sent_list.append(split_sentence)
            for word in split_sentence:
                word_cnt[word] += 1

    rev = sorted(word_cnt.items(), key=lambda x: x[1], reverse=True)
    stopWords1 = stopwords.words('english')
    stopWords2 = [i[0] for i in rev[:100]]
    stopWords3 = [i[0] for i in rev if i[1] < 5]
    stopWords_all = set(list(stopWords1) + list(stopWords2) + list(stopWords3))

    token_cnt=0
    new_sent_list_1 = []
    token_set=set()
    for sentence in new_sent_list:
        tokens=[word for word in sentence 
                                if word not in stopWords_all 
                                and len(word) > 2 
                                and bool(re.search('[a-zA-Z]', word)) 
                                and not bool(re.search(r'[\d/]', word)) 
                                and not is_date(word) and 
                                word not in ['nil'] 
                                and 'thomsonreuters.' not in word 
                                and 'reuters.' not in word]
        new_sent_list_1.append(tokens)
        token_cnt+=len(tokens)
        token_set.update(tokens)

    logging.info("Total tokens after stop word filtering: " + str(token_cnt))
    logging.info("Total unique tokens after stop word filtering: " + str(len(token_set)))

    return new_sent_list_1


def filter_background_word(sentence_list, time_list):
    # filter word that has flat occurence using method as discuss here
    # J.Weng and B. - S.Lee, “Event detection in twitter,” in Proc.Int.Conf.Weblogs Soc.Media, 2011, pp. 401–408.
    
    date_words_map = OrderedDict()

    for date_str, _words in zip(time_list, sentence_list):
        dt_obj = datetime.strptime(str(date_str), "%Y-%m-%d %H:%M:%S")
        i = dt_obj.date()
        if dt_obj.date() in date_words_map:
            date_words_map[dt_obj.date()].append(_words)
        else:
            date_words_map[dt_obj.date()] = [_words]

    result = defaultdict(dict)
    accumulated_time_cnt = 0
    accumulated_time_cnt_by_word = defaultdict(int)
    for date, word_list in date_words_map.items():
        current_time_cnt = len(word_list)  #
        accumulated_time_cnt += current_time_cnt
        for words in word_list:
            for w in set(words):  # count once for each word
                accumulated_time_cnt_by_word[w] += 1

        all_words_in_current_time = [w for words in word_list for w in
                                     words]
        current_time_word_tot_cnt = len(all_words_in_current_time)
        counter = Counter(
            all_words_in_current_time)
        for w, current_time_word_cnt in counter.items():
            # how frequent a word occur in current time frame
            current_time_ratio = current_time_word_cnt / current_time_word_tot_cnt
            # inverse of how frequent a word occur in all time frame
            accumulated_time_ratio = accumulated_time_cnt / accumulated_time_cnt_by_word[
                w]
            result[w][date] = current_time_ratio * math.log(accumulated_time_ratio)

    # Use word entropy for filtering
    wordsignal = result  
    entropy = dict()
    for word in wordsignal:  
        row = wordsignal[word]  
        values = list(row.values())  
        values = np.array(values)
        values = values / sum(values)
        E = -np.sum(np.multiply(values, np.log(values)))
        entropy[word] = E  

    tokens = list(entropy.items())
    sortedtemp = sorted(tokens, key=lambda x: x[1], reverse=True)
    stopwords = set([i[0] for i in sortedtemp[:2000]])

    token_cnt=0
    token_set=set()

    # remove the background words
    filtered_date_words_map = OrderedDict()
    for date1 in date_words_map:
        token_list = []
        for words in date_words_map[date1]:
            words = [w for w in words if w not in stopwords]
            if words:
                token_list.append(words)
                token_cnt += len(words)
                token_set.update(words)
        filtered_date_words_map[date1] = token_list
        

    train_data = [i for d in filtered_date_words_map.values() for i in d]

    logging.info("Total tokens after background word filtering: " + str(token_cnt))
    logging.info("Total unique tokens after background word filtering: " + str(len(token_set)))

    return train_data


if __name__ == '__main__':
    config=yaml.load(open("config/config.yaml"))
    num_top_words = config["num_top_words"]
    num_of_topics = config["num_of_topics"]
    num_of_word_per_topic = config["num_of_word_per_topic"]
    time_period = config["time_period"]
    lda_iteration = config["lda_iteration"]

    df = load_data('input/translated_file.csv')
    logging.info('File Loaded')
    logging.info('Starting preprocessing...')
    df = preprocess(df)
    logging.info('Preprocess done')

    logging.info('Starting filter stop words...')
    sentence_list = filter_stop_word(df[combine].tolist())
    logging.info('Filter stop words done')
    
    logging.info('Starting filter background words...')
    sentence_list = filter_background_word(sentence_list, df[target].tolist())
    logging.info('Filter background words done')

    logging.info('Starting generating LDA...')
    generate_lda(sentence_list, num_of_topics, num_of_word_per_topic, lda_iteration)
    logging.info('Generating LDA done, please find the trending words and topics in topic.txt under current directory')
    
    logging.info('Completed!')