import docx2txt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]
    tokens = [[word.lower() for word in sentence if word not in string.punctuation] for sentence in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [[word for word in sentence if word not in stop_words] for sentence in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in tokens]
    return tokens, sentences

def calculate_similarity(text1, text2):
    tokens1, sentences1 = preprocess_text(text1)
    tokens2, sentences2 = preprocess_text(text2)
    
    # Flatten the list of tokens into a single list
    flat_tokens1 = [word for sentence in tokens1 for word in sentence]
    flat_tokens2 = [word for sentence in tokens2 for word in sentence]
    
    intersection = len(set(flat_tokens1) & set(flat_tokens2))
    union = len(set(flat_tokens1) | set(flat_tokens2))
    similarity_ratio = intersection / union * 100
    
    # Calculate similarity at sentence level
    similar_sentences = []
    for sentence1 in sentences1:
        for sentence2 in sentences2:
            if calculate_sentence_similarity(sentence1, sentence2) >= 50:  # Assume sentences are similar if similarity >= 50%
                similar_sentences.append((sentence1, sentence2))
                break  # Move to the next sentence in text1
    return similarity_ratio, similar_sentences

def calculate_sentence_similarity(sentence1, sentence2):
    # Convert sentences to tokens
    tokens1 = word_tokenize(sentence1)
    tokens2 = word_tokenize(sentence2)
    
    # Preprocess tokens
    tokens1 = [word.lower() for word in tokens1 if word not in string.punctuation and word.isalnum()]
    tokens2 = [word.lower() for word in tokens2 if word not in string.punctuation and word.isalnum()]
    
    # Calculate Jaccard similarity at word level
    intersection = len(set(tokens1) & set(tokens2))
    union = len(set(tokens1) | set(tokens2))
    similarity_ratio = intersection / union * 100
    return similarity_ratio

def plot_similarity(percentage):
    labels = ['Similarity', 'Difference']
    sizes = [percentage, 100 - percentage]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0) 
    plt.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode)
    plt.axis('equal')  
    plt.title('Plagiarism Checker Result')
    plt.show()

def load_docx(file_path):
    try:
        with open(file_path, "rb") as file:
            text = docx2txt.process(file)
        return text
    except Exception as e:
        print("Error:", e)
        return None

def main():
    file1_path = input("Enter file path 1: ").strip('"')
    file2_path = input("Enter file path 2: ").strip('"')
    text1 = load_docx(file1_path)
    text2 = load_docx(file2_path)
    if text1 and text2:
        similarity_percentage, similar_sentences = calculate_similarity(text1, text2)
        print("Similarity Percentage:", similarity_percentage)
        print("Similar Sentences:")
        for pair in similar_sentences:
            print("Text 1:", pair[0])
            print("Text 2:", pair[1])
            print()
        plot_similarity(similarity_percentage)
    else:
        print("Failed to load one or both of the DOCX files.")

if __name__ == "__main__":
    main()
