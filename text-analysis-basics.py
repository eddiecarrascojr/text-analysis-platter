import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob



list_of_nlk = ["names", "stopwords", "state_union", "twitter_samples", "movie_reviews",
               "averaged_perceptron_tagger", "vader_lexicon", "punkt", ]

nltk.download(list_of_nlk)

sentiment = SentimentIntensityAnalyzer()

sample_text = """ For some quick analysis, creating a corpus could be overkill. If all you need is a word list, there are simpler ways to achieve that goal."""

sam_text_two = """
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
"""

frequencies = nltk.FreqDist(sample_text)
frequencies = nltk.FreqDist([w.lower() for w in frequencies])

print("Top 10 most used words", frequencies.most_common(10))

polarity = sentiment.polarity_scores(sample_text)
print("Polarity of sample text", polarity)

blob = TextBlob(sam_text_two)
print("Tags of second sample text", blob.tags)

for index, sentence in enumerate(blob.sentences):
    print("Sentiment of " , index, " polarity", sentence.sentiment.polarity)

