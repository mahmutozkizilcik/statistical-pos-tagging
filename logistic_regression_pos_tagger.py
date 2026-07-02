import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

def extract_features(sentence, index):
    """
    extracts various features from the word to train the logistic regression model
    
    """
    word = sentence[index]
    features = {}
    
    features['word'] = word.lower()
    features['is_cap'] = word[0].isupper()
    features['is_numeric'] = word.isdigit()
    
    # getting prefixes and suffixes
    if len(word) >= 2:
        features['prefix-2'] = word[:2]
        features['suffix-2'] = word[-2:]
    else:
        features['prefix-2'] = word
        features['suffix-2'] = word
        
    if len(word) >= 3:
        features['prefix-3'] = word[:3]
        features['suffix-3'] = word[-3:]
    else:
        features['prefix-3'] = word
        features['suffix-3'] = word
        
    # check surrounding words for context
    if index == 0:
        features['prev_word'] = 'START'
    else:
        features['prev_word'] = sentence[index-1].lower()
        
    if index == len(sentence) - 1:
        features['next_word'] = 'END'
    else:
        features['next_word'] = sentence[index+1].lower()
        
    return features


def main():
    tagged_sents = list(brown.tagged_sents(tagset='universal'))
    
    print(f"Total sentences in dataset: {len(tagged_sents)}")
    print(f"\nFirst example sentence:\n{tagged_sents[0]}")
    
    universal_tags = sorted(set(tag for sent in tagged_sents for word, tag in sent))
    print(f"\nUniversal Tags: {universal_tags}")

    train_sents, test_sents = train_test_split(tagged_sents, test_size=0.2, random_state=42)
    
    print(f"\nWorking with Subset -> Train sentences: {len(train_sents)}, Test sentences: {len(test_sents)}")


    X_train = []
    y_train = []

    
    for sent in train_sents:
        words = [w for w, t in sent]

        for i, (word, tag) in enumerate(sent):
            X_train.append(extract_features(words,i))
            y_train.append(tag)

    model = Pipeline([
        ('vectorizer', DictVectorizer(sparse=True)),
        ('classifier', LogisticRegression(max_iter=1000, n_jobs=-1))
    ])

    print('Model training started')
    model.fit(X_train, y_train)


    X_test = []
    y_test = []
    for sent in test_sents:
        words = [w for w,t in sent]
        for i, (word, tag) in enumerate(sent):
            X_test.append(extract_features(words, i))
            y_test.append(tag)
    
    print("\n=== LOGISTIC REGRESSION EVALUATION ===")
    y_pred = []
    last_5_samples = []

    print("Starting decoding for test sentences...")

    for i, sent in enumerate(test_sents):
        words = [w for w, t in sent]
        true_tags = [t for w, t in sent]
        
        # Extract features for the current sentence
        sent_features = [extract_features(words, i) for i in range(len(words))]
        pred_tags = model.predict(sent_features)
        pred_tags_list = [str(t) for t in pred_tags]
        
        y_pred.extend(pred_tags_list)
        
        if i >= len(test_sents) - 5:
            last_5_samples.append((words, true_tags, pred_tags_list))
        
        if (i + 1) % 2500 == 0:
            print(f"Processed {i+1}/{len(test_sents)} sentences...")

    print("\n=== FINAL RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n=== SAMPLE PREDICTIONS (LAST 5 SENTENCES) ===")
    for words, true, pred in last_5_samples:
        print("-" * 50)
        print(f"SENTENCE  : {' '.join(words)}")
        print(f"ORIGINAL  : {true}")
        print(f"PREDICTED : {pred}")
    

if __name__ == "__main__":
    main()
