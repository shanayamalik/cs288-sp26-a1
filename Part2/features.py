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


class TechnicalTerms(FeatureMap):
    name = "technical"
    
    # Expanded domain-specific vocabulary for 20 newsgroups categories
    
    # comp.* groups (computers)
    COMP_GRAPHICS = {
        "graphics", "image", "rendering", "3d", "polygon", "pixel", "resolution",
        "opengl", "gif", "jpeg", "tiff", "bitmap", "raytracing", "animation"
    }
    
    COMP_OS_WINDOWS = {
        "windows", "win3", "win31", "dos", "msdos", "microsoft", "ini", "driver",
        "dll", "exe", "bat", "config", "autoexec", "winword", "excel"
    }
    
    COMP_SYS_IBM_MAC = {
        "ibm", "pc", "mac", "apple", "macintosh", "powerbook", "thinkpad",
        "motherboard", "bios", "scsi", "ide", "floppy", "harddrive", "vga"
    }
    
    COMP_WINDOWS_X = {
        "xwindows", "x11", "motif", "openwindows", "xterm", "widget", "twm",
        "xlib", "unix", "server", "display", "xview"
    }
    
    # sci.* groups (science)
    SCI_SPACE = {
        "space", "nasa", "orbit", "satellite", "shuttle", "moon", "mars",
        "launch", "rocket", "spacecraft", "astronaut", "mission", "jpl",
        "payload", "venus", "jupiter", "telescope", "astronomy", "comet"
    }
    
    SCI_ELECTRONICS = {
        "circuit", "voltage", "resistor", "capacitor", "transistor", "diode",
        "amplifier", "oscilloscope", "breadboard", "solder", "ic", "chip",
        "pcb", "schematic", "breadboard", "multimeter"
    }
    
    SCI_MED = {
        "medical", "doctor", "patient", "disease", "symptom", "diagnosis",
        "treatment", "hospital", "clinic", "medicine", "physician", "health",
        "surgery", "therapy", "prescription", "medication"
    }
    
    SCI_CRYPT = {
        "encryption", "decrypt", "cipher", "cryptography", "rsa", "des",
        "pgp", "privacy", "keypair", "plaintext", "ciphertext", "hash",
        "secure", "nsa", "clipper", "escrow"
    }
    
    # rec.* groups (recreation)
    REC_AUTOS = {
        "car", "auto", "vehicle", "engine", "transmission", "brake", "tire",
        "dealer", "honda", "toyota", "ford", "bmw", "audi", "mustang",
        "mazda", "nissan", "clutch", "horsepower", "turbo"
    }
    
    REC_MOTORCYCLES = {
        "motorcycle", "bike", "yamaha", "harley", "kawasaki", "suzuki",
        "helmet", "rider", "ride", "touring", "cruiser", "sportbike"
    }
    
    REC_SPORT_BASEBALL = {
        "baseball", "pitcher", "batter", "homerun", "innings", "sox",
        "yankees", "dodgers", "cubs", "mets", "braves", "batting", "rbi"
    }
    
    REC_SPORT_HOCKEY = {
        "hockey", "nhl", "puck", "goalie", "penguins", "rangers", "bruins",
        "maple", "canadiens", "playoff", "stanley", "cup", "rink"
    }
    
    # talk.* groups (debate/politics)
    TALK_POLITICS_GUNS = {
        "gun", "firearm", "weapon", "rifle", "handgun", "ammunition",
        "nra", "amendment", "second", "militia", "atf", "assault"
    }
    
    TALK_POLITICS_MIDEAST = {
        "israel", "israeli", "arab", "palestinian", "lebanon", "syria",
        "egypt", "jordan", "gaza", "westbank", "peace", "plo", "jews"
    }
    
    TALK_POLITICS_MISC = {
        "clinton", "congress", "senate", "legislation", "vote", "election",
        "democrat", "republican", "tax", "budget", "policy"
    }
    
    TALK_RELIGION_MISC = {
        "god", "jesus", "christ", "christian", "bible", "church", "faith",
        "prayer", "worship", "scripture", "salvation", "sin", "heaven"
    }
    
    # alt.atheism
    ALT_ATHEISM = {
        "atheist", "atheism", "secular", "agnostic", "deity", "theist",
        "religion", "belief", "evidence", "rational", "logic"
    }
    
    # soc.religion.christian
    SOC_RELIGION = {
        "gospel", "testament", "apostle", "disciple", "baptism", "communion",
        "resurrection", "crucifixion", "trinity", "catholic", "protestant"
    }
    
    # misc.forsale
    MISC_FORSALE = {
        "sale", "sell", "selling", "forsale", "price", "offer", "shipping",
        "condition", "mint", "obo", "asking", "interested", "email"
    }
    
    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """Detects technical/domain-specific terms for newsgroup classification"""
        # lowercase and tokenize
        tokens = set(word.lower().strip(",.!?;:\"'()[]{}") for word in text.split())
        
        features = {}
        
        # count terms from each specific newsgroup category
        comp_graphics_count = sum(1 for token in tokens if token in self.COMP_GRAPHICS)
        comp_os_windows_count = sum(1 for token in tokens if token in self.COMP_OS_WINDOWS)
        comp_sys_count = sum(1 for token in tokens if token in self.COMP_SYS_IBM_MAC)
        comp_windows_x_count = sum(1 for token in tokens if token in self.COMP_WINDOWS_X)
        
        sci_space_count = sum(1 for token in tokens if token in self.SCI_SPACE)
        sci_electronics_count = sum(1 for token in tokens if token in self.SCI_ELECTRONICS)
        sci_med_count = sum(1 for token in tokens if token in self.SCI_MED)
        sci_crypt_count = sum(1 for token in tokens if token in self.SCI_CRYPT)
        
        rec_autos_count = sum(1 for token in tokens if token in self.REC_AUTOS)
        rec_motorcycles_count = sum(1 for token in tokens if token in self.REC_MOTORCYCLES)
        rec_baseball_count = sum(1 for token in tokens if token in self.REC_SPORT_BASEBALL)
        rec_hockey_count = sum(1 for token in tokens if token in self.REC_SPORT_HOCKEY)
        
        talk_guns_count = sum(1 for token in tokens if token in self.TALK_POLITICS_GUNS)
        talk_mideast_count = sum(1 for token in tokens if token in self.TALK_POLITICS_MIDEAST)
        talk_politics_count = sum(1 for token in tokens if token in self.TALK_POLITICS_MISC)
        talk_religion_count = sum(1 for token in tokens if token in self.TALK_RELIGION_MISC)
        
        alt_atheism_count = sum(1 for token in tokens if token in self.ALT_ATHEISM)
        soc_religion_count = sum(1 for token in tokens if token in self.SOC_RELIGION)
        misc_forsale_count = sum(1 for token in tokens if token in self.MISC_FORSALE)
        
        # add features for each category (only if present)
        if comp_graphics_count > 0:
            features["comp_graphics"] = float(comp_graphics_count)
        if comp_os_windows_count > 0:
            features["comp_windows"] = float(comp_os_windows_count)
        if comp_sys_count > 0:
            features["comp_sys"] = float(comp_sys_count)
        if comp_windows_x_count > 0:
            features["comp_x11"] = float(comp_windows_x_count)
            
        if sci_space_count > 0:
            features["sci_space"] = float(sci_space_count)
        if sci_electronics_count > 0:
            features["sci_electronics"] = float(sci_electronics_count)
        if sci_med_count > 0:
            features["sci_med"] = float(sci_med_count)
        if sci_crypt_count > 0:
            features["sci_crypt"] = float(sci_crypt_count)
            
        if rec_autos_count > 0:
            features["rec_autos"] = float(rec_autos_count)
        if rec_motorcycles_count > 0:
            features["rec_motorcycles"] = float(rec_motorcycles_count)
        if rec_baseball_count > 0:
            features["rec_baseball"] = float(rec_baseball_count)
        if rec_hockey_count > 0:
            features["rec_hockey"] = float(rec_hockey_count)
            
        if talk_guns_count > 0:
            features["talk_guns"] = float(talk_guns_count)
        if talk_mideast_count > 0:
            features["talk_mideast"] = float(talk_mideast_count)
        if talk_politics_count > 0:
            features["talk_politics"] = float(talk_politics_count)
        if talk_religion_count > 0:
            features["talk_religion"] = float(talk_religion_count)
            
        if alt_atheism_count > 0:
            features["alt_atheism"] = float(alt_atheism_count)
        if soc_religion_count > 0:
            features["soc_religion"] = float(soc_religion_count)
        if misc_forsale_count > 0:
            features["misc_forsale"] = float(misc_forsale_count)
        
        return self.prefix_with_name(features)


FEATURE_CLASSES_MAP = {c.name: c for c in [BagOfWords, SentenceLength, NegationFeatures, PunctuationFeatures, SentimentLexicon, TechnicalTerms]}


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
