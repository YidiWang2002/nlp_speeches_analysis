'''
Authors: Jingkai Wang, Yidi Wang, Qixiang Jiang
Feb 27 2023
DS3500
Prof. Rachlin

This is the main file of the hw3
'''

from Text_viz_reusable import TextParser


def main():
    tp = TextParser()
    tp.load_text(
        "Biden_thanksgiving.txt",
        label="Biden_thanksgiving",
        parser=TextParser._default_parser,
    )
    tp.load_text(
        "Biden_illagal_immagrants.txt",
        label="Biden_illagal_immagrants",
        parser=TextParser._default_parser,
    )
    tp.load_text(
        "Obama_thanksgiving.txt",
        label="Obama_thanksgiving",
        parser=TextParser._default_parser,
    )
    tp.load_text(
        "Obama_illegal_immagrants.txt",
        label="Obama_illegal_immagrants",
        parser=TextParser._default_parser,
    )
    tp.load_text(
        "Trump_thanksgiving.txt",
        label="Trump_thanksgiving",
        parser=TextParser._default_parser,
    )
    tp.load_text(
        "Trump_illegal_immagrants.txt",
        label="dTrump_illegal_immagrants",
        parser=TextParser._default_parser,
    )

    # Generate the sankey diagram; we can specify our own word list or not
    tp.wordcount_sankey()
    # tp.wordcount_sankey(word_list=["AND", "GOING", "PEOPLE", "WE", "DAY"])

    # Generate the word cloud
    tp.word_cloud(tp.data)

    # Generate the bar chart
    tp.visualize_bar(tp.data)


if __name__ == "__main__":
    main()
