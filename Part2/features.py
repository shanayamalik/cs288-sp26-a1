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


'''
TODO: uncomment SentimentLexicon if professor confirms that features only need to 
improve performance on average (not 100% of runs). this feature improves SST-2 dev 
accuracy from ~74% to ~76% on average, but results vary slightly across runs. 
If features must improve every single run, delete this and implement a different feature.
'''

# class SentimentLexicon(FeatureMap):
#     name = "sentiment"
#     
#     # sentiment lexicons  
#     POSITIVE_WORDS = {
#         # strong 
#         "excellent", "amazing", "wonderful", "fantastic", "brilliant", "perfect",
#         "outstanding", "superb", "terrific", "awesome", "incredible", "magnificent",
#         "spectacular", "phenomenal", "exceptional", "marvelous", "sensational",
#         "fabulous", "extraordinary", "stunning", "breathtaking", "impressive",
#         "remarkable", "masterpiece", "genius", "flawless", "sublime",
#         # general 
#         "good", "great", "nice", "fine", "best", "better", "superior", "top",
#         "beautiful", "lovely", "charming", "delightful", "pleasant", "attractive",
#         "gorgeous", "pretty", "elegant", "graceful", "wonderful",
#         # emotions 
#         "love", "loved", "loves", "loving", "enjoy", "enjoyed", "enjoys", "enjoying",
#         "like", "liked", "likes", "liking", "adore", "adored", "adores", "adoring",
#         "happy", "joy", "joyful", "cheerful", "pleased", "glad", "delighted",
#         "excited", "thrilled", "enthusiastic", "passionate", "inspired",
#         # entertainment 
#         "entertaining", "fun", "funny", "hilarious", "amusing", "humorous",
#         "witty", "clever", "smart", "interesting", "engaging", "compelling",
#         "captivating", "gripping", "thrilling", "exciting", "powerful",
#         "moving", "touching", "emotional", "heartwarming", "uplifting",
#         # quality 
#         "quality", "well", "strong", "solid", "effective", "successful",
#         "worthy", "valuable", "recommended", "must-see", "worth",
#         "refreshing", "original", "creative", "innovative", "unique",
#         # satisfaction
#         "satisfying", "satisfied", "pleased", "content", "fulfilled"
#     }
#     
#     NEGATIVE_WORDS = {
#         # strong 
#         "terrible", "awful", "horrible", "dreadful", "atrocious", "abysmal",
#         "appalling", "deplorable", "disastrous", "catastrophic", "horrendous",
#         "horrid", "hideous", "ghastly", "monstrous", "nightmare",
#         # general 
#         "bad", "poor", "worst", "worse", "inferior", "low", "weak",
#         "disappointing", "disappointed", "disappoints", "disappointment",
#         "unfortunate", "inadequate", "unsatisfactory", "subpar",
#         # emotions 
#         "hate", "hated", "hates", "hating", "dislike", "disliked", "dislikes",
#         "despise", "despised", "despises", "detest", "loathe",
#         "sad", "depressing", "depressed", "miserable", "unhappy",
#         "upset", "annoyed", "annoying", "irritating", "irritated", "frustrating",
#         # entertainment 
#         "boring", "bored", "dull", "tedious", "monotonous", "tiresome",
#         "uninteresting", "bland", "flat", "lifeless", "pointless",
#         "slow", "dragging", "waste", "wasted", "worthless",
#         "stupid", "dumb", "silly", "ridiculous", "absurd", "nonsense",
#         "lame", "pathetic", "laughable", "joke",
#         # quality 
#         "mess", "messy", "sloppy", "clumsy", "awkward", "confusing",
#         "confused", "chaotic", "incoherent", "incomprehensible",
#         "fail", "failed", "fails", "failure", "flawed", "broken",
#         "lacking", "missing", "empty", "hollow", "shallow",
#         "predictable", "cliché", "clichéd", "formulaic", "derivative",
#         "unconvincing", "unbelievable", "implausible",
#         # extreme 
#         "garbage", "trash", "rubbish", "junk", "crap",
#         "painful", "excruciating", "unbearable", "intolerable"
#     }
#     
#     @classmethod
#     def featurize(self, text: str) -> Dict[str, float]:
#         """Counts positive and negative sentiment words"""
#         words = [word.lower() for word in text.split()]
#         
#         pos_count = sum(1 for word in words if word in self.POSITIVE_WORDS)
#         neg_count = sum(1 for word in words if word in self.NEGATIVE_WORDS)
#         
#         features = {
#             "positive_count": float(pos_count),
#             "negative_count": float(neg_count),
#         }
#         
#         # add binary features for presence
#         if pos_count > 0:
#             features["has_positive"] = 1.0
#         if neg_count > 0:
#             features["has_negative"] = 1.0
#         
#         return self.prefix_with_name(features)


FEATURE_CLASSES_MAP = {c.name: c for c in [BagOfWords, SentenceLength]}


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
