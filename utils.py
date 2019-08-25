import csv
import os
import xml.etree.ElementTree as eTree
import tensorflow.keras.preprocessing.text as kpt
from tqdm import tqdm


def generate_tweets_csv(input_xml_path, input_label_path, tweet_path):
    """
    Generate Tweets CSV file
    :param input_xml_path: Input File path
    :param input_label_path: Labels file path
    :param tweet_path: output file to save the tweets
    :return:
    """
    with open(tweet_path, 'w', encoding='utf-8') as file:
        labels = ['text', 'gender']
        writer = csv.DictWriter(file, fieldnames=labels)
        writer.writeheader()
        for file_name in tqdm(os.listdir(input_xml_path)):
            if not file_name.endswith('.xml'):
                continue

            # Get the label of the file first
            file_id = os.path.splitext(file_name)[0]
            label_file = open(input_label_path)
            for row in label_file:
                if file_id in row:
                    # Use array pos 2 if you want to get ger but make sure to replace trailing \n
                    gender = row.split(':::')[2].rstrip()

            # Read the xml
            xml_file = os.path.join(input_xml_path, file_name)
            tree = eTree.parse(xml_file)
            root = tree.getroot()
            for i in range(len(root[0])):
                writer.writerow({'text': str(root[0][i].text), 'gender': str(gender)})
    file.close()


def convert_text_to_indices(text, tokenized_dictionary):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    word_indices = []
    for word in kpt.text_to_word_sequence(text):
        if word in tokenized_dictionary:
            word_indices.append(tokenized_dictionary[word])
    return word_indices
