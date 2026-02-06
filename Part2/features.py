from collections import ChainMap
from typing import Callable, Dict, Set

import pandas as pd


class FeatureMap:
    name: str

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        pass

    @classmethod
    def prefix_with_name(self, d: Dict) -> Dict[str, float]:
        """just a handy shared util function"""
        return {f"{self.name}/{k}": v for k, v in d.items()}


class BagOfWords(FeatureMap):
    name = "bow"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        # split, lowercase, remove stopwords, create binary features
        words = set(word.lower() for word in text.split() if word.lower() not in self.STOP_WORDS)
        features = {word: 1.0 for word in words}
        return self.prefix_with_name(features)


class SentenceLength(FeatureMap):
    name = "len"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        if len(text.split()) < 10:
            k = "short"
            v = 1.0
        else:
            k = "long"
            v = 5.0
        ret = {k: v}
        return self.prefix_with_name(ret)


class NegationFeatures(FeatureMap):
    name = "negation"
    
    # negation words that flip sentiment
    NEGATION_WORDS = {
        "not", "no", "never", "none", "nothing", "nobody", "nowhere",
        "neither", "nor", "cannot", "can't", "cant", "won't", "wont",
        "don't", "dont", "didn't", "didnt", "doesn't", "doesnt",
        "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt",
        "weren't", "werent", "shouldn't", "shouldnt", "wouldn't",
        "wouldnt", "couldn't", "couldnt", "haven't", "havent",
        "hasn't", "hasnt", "hadn't", "hadnt"
    }
    
    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """Detects negation words that flip sentiment meaning"""
        # clean and tokenize
        tokens = text.lower().split()
        
        # count the amount of negation words
        negation_count = sum(1 for token in tokens if token.strip(",.!?;:\"'()[]{}") in self.NEGATION_WORDS)
        
        features = {}
        
        # cap at 3 and scale to 0-1 range  
        features["negation_count"] = float(min(negation_count, 3)) / 3.0
        
        # binary presence feature
        if negation_count > 0:
            features["has_negation"] = 1.0
        
        return self.prefix_with_name(features)


class PunctuationFeatures(FeatureMap):
    name = "punct"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """Extracts punctuation features that indicate sentiment intensity"""
        # Count specific punctuation marks
        exclamation_count = text.count('!')
        question_count = text.count('?')
        ellipsis_count = text.count('...')
        
        # Get word count for ratios
        word_count = len(text.split())
        
        features = {}
        
        # Count features
        features["exclamation_count"] = float(exclamation_count)
        features["question_count"] = float(question_count)
        features["ellipsis_count"] = float(ellipsis_count)
        
        # Binary presence features
        if exclamation_count > 0:
            features["has_exclamation"] = 1.0
        if question_count > 0:
            features["has_question"] = 1.0
        if ellipsis_count > 0:
            features["has_ellipsis"] = 1.0
        
        # Ratio features (normalized by text length)
        if word_count > 0:
            features["exclamation_ratio"] = float(exclamation_count) / word_count
            features["question_ratio"] = float(question_count) / word_count
        
        return self.prefix_with_name(features)

class SentimentLexicon(FeatureMap):
    name = "sentiment"
    
    # sentiment lexicons  
    POSITIVE_WORDS = {
        # strong 
        "excellent", "amazing", "wonderful", "fantastic", "brilliant", "perfect",
        "outstanding", "superb", "terrific", "awesome", "incredible", "magnificent",
        "spectacular", "phenomenal", "exceptional", "marvelous", "sensational",
        "fabulous", "extraordinary", "stunning", "breathtaking", "impressive",
        "remarkable", "masterpiece", "genius", "flawless", "sublime",
        # general 
        "good", "great", "nice", "fine", "best", "better", "superior", "top",
        "beautiful", "lovely", "charming", "delightful", "pleasant", "attractive",
        "gorgeous", "pretty", "elegant", "graceful", "wonderful",
        # emotions 
        "love", "loved", "loves", "loving", "enjoy", "enjoyed", "enjoys", "enjoying",
        "like", "liked", "likes", "liking", "adore", "adored", "adores", "adoring",
        "happy", "joy", "joyful", "cheerful", "pleased", "glad", "delighted",
        "excited", "thrilled", "enthusiastic", "passionate", "inspired",
        # entertainment 
        "entertaining", "fun", "funny", "hilarious", "amusing", "humorous",
        "witty", "clever", "smart", "interesting", "engaging", "compelling",
        "captivating", "gripping", "thrilling", "exciting", "powerful",
        "moving", "touching", "emotional", "heartwarming", "uplifting",
        # quality 
        "quality", "well", "strong", "solid", "effective", "successful",
        "worthy", "valuable", "recommended", "must-see", "worth",
        "refreshing", "original", "creative", "innovative", "unique",
        # satisfaction
        "satisfying", "satisfied", "pleased", "content", "fulfilled"
    }
    
    NEGATIVE_WORDS = {
        # strong 
        "terrible", "awful", "horrible", "dreadful", "atrocious", "abysmal",
        "appalling", "deplorable", "disastrous", "catastrophic", "horrendous",
        "horrid", "hideous", "ghastly", "monstrous", "nightmare",
        # general 
        "bad", "poor", "worst", "worse", "inferior", "low", "weak",
        "disappointing", "disappointed", "disappoints", "disappointment",
        "unfortunate", "inadequate", "unsatisfactory", "subpar",
        # emotions 
        "hate", "hated", "hates", "hating", "dislike", "disliked", "dislikes",
        "despise", "despised", "despises", "detest", "loathe",
        "sad", "depressing", "depressed", "miserable", "unhappy",
        "upset", "annoyed", "annoying", "irritating", "irritated", "frustrating",
        # entertainment 
        "boring", "bored", "dull", "tedious", "monotonous", "tiresome",
        "uninteresting", "bland", "flat", "lifeless", "pointless",
        "slow", "dragging", "waste", "wasted", "worthless",
        "stupid", "dumb", "silly", "ridiculous", "absurd", "nonsense",
        "lame", "pathetic", "laughable", "joke",
        # quality 
        "mess", "messy", "sloppy", "clumsy", "awkward", "confusing",
        "confused", "chaotic", "incoherent", "incomprehensible",
        "fail", "failed", "fails", "failure", "flawed", "broken",
        "lacking", "missing", "empty", "hollow", "shallow",
        "predictable", "cliché", "clichéd", "formulaic", "derivative",
        "unconvincing", "unbelievable", "implausible",
        # extreme 
        "garbage", "trash", "rubbish", "junk", "crap",
        "painful", "excruciating", "unbearable", "intolerable"
    }
    
    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """Counts positive and negative sentiment words"""
        words = [word.lower() for word in text.split()]
        
        pos_count = sum(1 for word in words if word in self.POSITIVE_WORDS)
        neg_count = sum(1 for word in words if word in self.NEGATIVE_WORDS)
        
        features = {
            "positive_count": float(pos_count),
            "negative_count": float(neg_count),
        }
        
        # add binary features for presence
        if pos_count > 0:
            features["has_positive"] = 1.0
        if neg_count > 0:
            features["has_negative"] = 1.0
        
        return self.prefix_with_name(features)


FEATURE_CLASSES_MAP = {c.name: c for c in [BagOfWords, SentenceLength, NegationFeatures, PunctuationFeatures, SentimentLexicon]}


def make_featurize(
    feature_types: Set[str],
) -> Callable[[str], Dict[str, float]]:
    featurize_fns = [FEATURE_CLASSES_MAP[n].featurize for n in feature_types]

    def _featurize(text: str):
        f = ChainMap(*[fn(text) for fn in featurize_fns])
        return dict(f)

    return _featurize


__all__ = ["make_featurize"]

if __name__ == "__main__":
    text = "I love this movie"
    print(text)
    print(BagOfWords.featurize(text))
    featurize = make_featurize({"bow", "len"})
    print(featurize(text))
