'''
Authors: Jingkai Wang, Yidi Wang, Qixiang Jiang
Feb 27 2023
DS3500
Prof. Rachlin
'''

from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import os
import parsers as par
import plotly.graph_objects as go



class TextParser:

    def __init__(self, filename=None):
        # manage data about the different texts that
        # we register with the framework
        self.data: dict = {} # our data extracted from text files
        self.filename = filename

        # word_count: {text1: wordcount1, text2: wordcount2, ….}
        # word_length: {text1: wordlength1, text2:wordlength2, …}
        # sentiment: {text1, sentiment1, text2:sentiment2, …}

        self.stop_words = set(stopwords.words('english'))
        # edited by Jingkai


    @staticmethod
    def _default_parser(filename):
        '''
                Pre-process the file including removing unnecessary whitespace, stop words, punctuation,
            and capitalization.Then store intermediate results: statistics such as word count,
            average word length and sentiment.

            @param:
                filename (str): the name of the file we need to use
            @return:
                a dcitionary: {word_count: X, word_length: Y, sentiment_score: Z}
        '''
        text = open(filename, 'r').read()
        clean_text = par.remove_punctuation(text)
        tokens = par.tokenize(clean_text)
        no_stopword_tokens = par.remove_stop_words(tokens)
        final_tokens = par.capitalize(no_stopword_tokens)

        wc = len(final_tokens)
        avg_length = par.get_avg_word_length(final_tokens)
        sent_score = par.calc_sentiment_score(final_tokens)

        data = {
            'word_count': wc,
            'word_length': avg_length,
            'sentiment_score': sent_score,
            'word_list': final_tokens
        }

        total_length = 0
        with open(filename) as file:
            while line := file.readline():
                words = line.split()
                for word in words:
                    if word in no_stopword_tokens:
                        total_length += len(word)
                        data['word_count'] += 1
        data['word_length'] = total_length / data['word_count'] if data['word_count'] else 0
        counter = Counter(data['word_list'])

        data['counter'] = counter
        return data

    def _save_results(self, label, results):
        """ Integrate parsing results into internal state
        label: unique label for a text file that we parsed
        results: the data extracted from the file as a dictionary attribute-->raw data
        """
        for k, v in results.items():
            self.data[k][label] = v

    def load_text(self, filename, label='', parser=None):
        if parser is None:
            parser = self._default_parser()
        data: dict = parser(filename)

        # format
        # data = {
        #     "word_count": 10,
        #     "word_length": 10,
        # }

        # update to global dict
        if label:
            key = label
        else:
            basename = os.path.basename(filename)
            key = ''.join(basename.split('.')[:-1])
        self.data[key] = data

    # creating sankey didagram with reusable code
    def wordcount_sankey(self, word_list=None, k=5):
        """
           Creating sankey diagram by using the wordlist and control the max numer of
           word in different text file
           :param word_list: the words taht want ot be sed
           :param k: the maxixum number of words that should be shown in the diagram, defalt as 5.
           :return: a sankey diagram
        """

        if word_list is None:
            word_list = []
            for label, data in self.data.items():
                most_common = self.get_most_common_words(data['word_list'], k)
                word_list.extend([pair[0] for pair in most_common])

        labels = list(self.data.keys()) + word_list
        sources = []
        targets = []
        values = []

        for text in list(self.data.keys()):
            for word in word_list:
                word_cnt = self.data[text]['word_list'].count(word)
                sources.append(labels.index(text))
                targets.append(labels.index(word))
                values.append(word_cnt)

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=labels,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            ))])

        fig.update_layout(title_text='Sankey Diagram of President speeches', font_size=10)
        fig.show()

    def get_most_common_words(self, word_list, k):
        counter = Counter(word_list)
        most_occur = counter.most_common(k)
        return most_occur

    # lecture code for mapping
    def _code_mapping(df, src, targ):
        # Get distinct labels
        labels = sorted(list(set(list(df[src]) + list(df[targ]))))

        # Get integer codes
        codes = list(range(len(labels)))

        # Create label to code mapping
        lc_map = dict(zip(labels, codes))

        # Substitute names for codes in dataframe
        df = df.replace({src: lc_map, targ: lc_map})

        return df, labels

    # make sankey function for the class
    def make_sankey(self, df, src, targ, vals=None, **kwargs):
        """ Create a sankey diagram linking src values to
        target values with thickness vals """

        if vals:
            values = df[vals]
        else:
            values = [1] * len(df)

        df, labels = self._code_mapping(df, src, targ)
        link = {'source': df[src], 'target': df[targ], 'value': values}
        pad = kwargs.get('pad', 50)

        node = {'label': labels, 'pad': pad}
        sk = go.Sankey(link=link, node=node)
        fig = go.Figure(sk)
        fig.show()


    def top_k_word(self, k: int = 5) -> list:
        words_union = set()

        for key, data in self.data.items():
            counter = data['counter']
            num = 0
            for w in counter.most_common():
                if w[0] not in self.stop_words:
                    words_union.add(w[0])
                    num += 1
                    if num == k:
                        break
        return list(words_union)

    # the second visualization
    # created by Jingkai Wang
    def word_cloud(self, *args, **kwargs):
        ''' Create a visuliazation of the word cloud '''
        # Create the wordcloud object
        num_article = len(self.data)
        plt.figure(figsize=(4 * num_article, 4))

        idx = 1
        for key, val in self.data.items():
            word_counts = self.data[key]['counter']
            my_cloud = WordCloud(
                background_color='white',  # set the background color
                width=1000, height=800,
                max_font_size=150,  # set the max size
                min_font_size=12,  # set the min size
                margin=0,
                colormap='winter',
                stopwords=self.stop_words,
                random_state=idx  # set a random numer in order to create different color
            ).generate_from_frequencies(word_counts)
            plt.subplot(1, num_article, idx)
            plt.imshow(my_cloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.title(key)
            idx += 1
        plt.show()

    # 3rd viz: create the bar chart with negative value
    # create by Yidi Wang
    def visualize_bar(self, data):

        name = list(data.keys())
        scores = [v['sentiment_score'] for k, v in data.items()]

        pos_scores = [score if score > 0 else 0 for score in scores]
        neg_scores = [score if score < 0 else 0 for score in scores]

        # set the graph size and label size
        fig = plt.figure(figsize=(20, 32))
        ax = fig.add_subplot(111)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        ax.bar(name, pos_scores, color='green')
        ax.bar(name, neg_scores, color='red')
        ax.set_ylim(min(scores) * 1.2, max(scores) * 1.2)
        ax.set_xlabel('Speech', fontsize=30)
        ax.set_ylabel('Sentiment', fontsize=30)
        ax.set_title('The Bar chart with negative value', fontsize=30)
        plt.xticks(rotation=45)

        # show the bar chart with negative value
        plt.show()




