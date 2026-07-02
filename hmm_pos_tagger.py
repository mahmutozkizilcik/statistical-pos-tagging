import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from collections import defaultdict, Counter

nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

class HMMTagger:
    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.emissions = defaultdict(Counter)
        self.tags = set()
        self.vocab = set()
        self.tag_counts = Counter()

    def train(self, tagged_sents):
        for sent in tagged_sents:
            prev_tag = '<START>'
            for word, tag in sent:
                self.transitions[prev_tag][tag] += 1
                self.emissions[tag][word] += 1
                self.tag_counts[tag] += 1
                self.tags.add(tag)
                self.vocab.add(word)
                prev_tag = tag


    def get_emission_prob(self, word, tag):
        """
        Calculates emission probability of a word given a tag.
        """
        emission_prob =0
        if word in self.vocab:
            emission_prob = (self.emissions[tag][word] + 1e-10) / self.tag_counts[tag] # Laplace smoothing for unknown/unseen emissions
        else:
            emission_prob = 1 /( self.tag_counts[tag] + len(self.vocab)) # Laplace smoothing for unknown vocabulary terms
        return np.log(emission_prob)

    def get_transition_prob(self, prev_tag, curr_tag):
        """
        Calculates transition probability from prev_tag to curr_tag.
        """
        transition_prob = 0
        transition_between_two = self.transitions[prev_tag][curr_tag] #laplace smoothing
        total_transition_from_prev = sum(self.transitions[prev_tag].values()) 

        num_tags = len(self.tags)

        transition_prob = (transition_between_two + 1)/(total_transition_from_prev + num_tags)


        return np.log(transition_prob)

    def viterbi(self, sentence):
        """
        Computes the viterbi algorithm for the given sentence.
        returns the predicted tags.
        """
        all_tags = sorted(list(self.tags))
        num_tags = len(all_tags)
        sen_len = len(sentence)
        
        # init matrices
        v_table =   np.full((num_tags,sen_len), -np.inf)
        back_ptr =np.zeros((num_tags,sen_len), dtype=int)
        
        # first word importance
        first_w = sentence[0]
        for i, t in enumerate(all_tags):
            v_table[i, 0] = self.get_transition_prob('<START>',t)+ self.get_emission_prob(first_w, t)
            
        # looping thru the rest of the sentence
        for j in range(1,sen_len):
            curr_w = sentence[j]
            for i, c_tag in enumerate(all_tags):
                temp_probs =[]
                for k, p_tag in enumerate(all_tags):
                    prob = v_table[k, j-1] + self.get_transition_prob(p_tag, c_tag)+ self.get_emission_prob(curr_w, c_tag)
                    temp_probs.append(prob)
                
                v_table[i, j] = np.max(temp_probs)
                back_ptr[i, j] = np.argmax(temp_probs)
                
        # backtracking part
        last_col = v_table[:, sen_len-1]
        best_end = np.argmax(last_col)
        path_idx =[best_end]
        
        for j in range(sen_len-1, 0, -1):
            prev = back_ptr[path_idx[-1], j]
            path_idx.append(prev)
        path_idx.reverse() # reverse to get correct order
        


        result_tags =[]
        for idx in path_idx:
            result_tags.append(all_tags[idx])
            
        return result_tags


def main():

    tagged_sents = list(brown.tagged_sents(tagset='universal'))
    
    print(f"Total sentences in dataset: {len(tagged_sents)}")
    print(f"\nFirst example sentence:\n{tagged_sents[0]}")
    
    universal_tags = sorted(set(tag for sent in tagged_sents for word, tag in sent))
    print(f"\nUniversal Tags: {universal_tags}")

    train_sents, test_sents = train_test_split(tagged_sents, test_size=0.2, random_state=42)
    
    print(f"\nWorking with Subset -> Train sentences: {len(train_sents)}, Test sentences: {len(test_sents)}")
    
    
    tagger = HMMTagger()
    tagger.train(train_sents)

    transitions_summary = {k: dict(sorted(v.items())[:5]) for k, v in sorted(tagger.transitions.items())[:5]}
    emissions_summary = {k: dict(sorted(v.items())[:5]) for k, v in sorted(tagger.emissions.items())[:5]}

    print(f"\nTagger Transitions (first 5 sorted): {transitions_summary}")
    print(f"\nTagger Emissions (first 5 sorted): {emissions_summary}")
    print(f"\nTagger Tags (sorted): {sorted(list(tagger.tags))}")
    print(f"\nVocabulary size: {len(tagger.vocab)}")
    print(f"\nTagger Vocab (first 5 sorted): {sorted(list(tagger.vocab))[:5]}")
    print(f"\nTagger Tag Counts (sorted items): {dict(sorted(tagger.tag_counts.items()))}")

    print("\n=== VITERBI DECODING & EVALUATION ===")
    y_true = []
    y_pred = []
    last_5_samples = []

    print("Starting decoding for test sentences...")
    for i, sent in enumerate(test_sents):
        words = [w for w, t in sent]
        true_tags = [t for w, t in sent]

        pred_tags = tagger.viterbi(words)
        y_true.extend(true_tags)
        y_pred.extend(pred_tags)
        
        if i >= len(test_sents) - 5:
            last_5_samples.append((words, true_tags, pred_tags))
        
        if (i + 1) % 2500 == 0:
            print(f"Processed {i+1}/{len(test_sents)} sentences...")

    print("\n=== FINAL RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("=== SAMPLE PREDICTIONS (LAST 5 SENTENCES) ===")
    for words, true, pred in last_5_samples:
        print("-" * 50)
        print(f"SENTENCE  : {' '.join(words)}")
        print(f"ORIGINAL  : {true}")
        print(f"PREDICTED : {pred}")
    

if __name__ == "__main__":
    main()
