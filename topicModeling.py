#Author: Nayan Arora 
#please note that on running this script locally the word cloud library malfunctions to produce ZeroDivision error. 
#In case this happens, just rerun the code and it should work just fine!

import pandas as pd
import csv
csv.field_size_limit(1000000000)

df=pd.read_csv("/Users/nayanarora/Desktop/AIT/Ass2_Qn_&_Data/state-of-the-union.csv")

df.columns=['speech_year','speech_text']

print(df.head)


#using the nltk library to make a list of stopwords which wont help in topic modeling and will thus be removed from ['speech_text']
#thus this step helps visualize the stopwords
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)


#now using the gensim python package, we perfrom lemmatization and stemming using wordnet from nltk library. 
#for creating the dictionary, we keep only the explicit entries and remove the ones 
#which appeared only once as they have really low significance (must be stopwords)- sourced from gensim documentation.

import gensim
import re
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('wordnet')

def lemmatization_stemming(text):
    return (WordNetLemmatizer().lemmatize(text, pos='v')) 

def preprocessData(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatization_stemming(token))
    return result



stoplist = stopwords #basically a list of frequently used words
speech_corpus=df['speech_text']

#using the suggestion in instructions given. convert all speech documents for all years into lowercase and 
#filter out the stopwords by using while using white space as the delimiter
speech_words = [[re.sub(r"[0-9]+","",word) 
          for word in preprocessData(speech) if word not in stoplist] 
          for speech in speech_corpus]

#visualizing the pre processing - count frequency of the words and save values using defaultdict (word,freq)->(val,key)from the collections library
from collections import defaultdict
freq = defaultdict(int) 

for word in speech_words:
    for token in word:
        freq[token] += 1

#now use the visualization of word frequency from above to remove words that only accur once. As they would not help
#in understanding data patterns
processed_speech_corpus = [[token for token in word 
                            if (freq[token] > 1 )] 
                            for word in speech_words]

while('' in processed_speech_corpus):
    processed_speech_corpus.remove('') 

print(processed_speech_corpus[0]) #check if the preprocessing and split worked
print('=========================')

#now we have completed the data preprocessing and we have the resultant processed_speech_corpus

#next we use the processed speech corpus to create a dictionary(lexicon)
#by creating a dictionary, we ensure every unique word in the processed speech corpus has a corresponding ID
#using corpora from gensim (given linked reference in the instructions)

from gensim import corpora

myDict = corpora.Dictionary(processed_speech_corpus)
print(myDict)
print('=========================')

bow_corpus = [myDict.doc2bow(word) for word in processed_speech_corpus]
print(bow_corpus[5]) #visaulize random entry to ensure the bag of words represenation works fine. 
print('=========================')


#now we have indexed the term frequency for all processed documents. 
#Next we to the generate the tf-idf weighted document vectors

from gensim import models

tfidf = models.TfidfModel(bow_corpus)
vector_corpus = tfidf[bow_corpus]
#print(vector_corpus[0])


#using the towards data science article shared we now compute the coherence scores to guage appropriate number of topics 

from gensim.models import LsiModel
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel

#coherence scores computed by giving start, stop and step values to the lsi modeling algorithm. 
#this will return us the coherence values along with the list of models which we can plot on a graph to 
#be able to make an intelligent guess on the nu,ber of topics needed for LDA modeling. 

def coherence(modeldict, tfidf_matrix, processed_corpus, stop, start, step):
    coherence_values = []
    list_models = []

#for each LSA model generated in the loop, it calculates the coherence of the topics using the CoherenceModel from Gensim.
    for num_topics in range(start, stop, step): 
        # generate Latent Sematic Analysis model to infer topic coherence
        lsi_model = LsiModel(tfidf_matrix, num_topics=num_topics, id2word = modeldict)  # train model
        list_models.append(lsi_model)
        coherence_model = CoherenceModel(model=lsi_model, texts=processed_corpus, dictionary=modeldict, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
    return list_models, coherence_values

def coherence_plot(processed_corpus,start, stop, step):
    #dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    tfidf_matrix = vector_corpus
    #processed_corpus = processed_speech_corpus
    list_models, coherence_values = coherence(myDict, tfidf_matrix, processed_corpus, stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Topics")
    plt.ylabel("Coherence Values")
    plt.legend(("coherence_values"))
    plt.show()

# if __name__ == '__main__':
#     start,stop,step=10,30,1
#     coherence_plot(processed_speech_corpus,start,stop,step)
 
#=========================
#to get the coherence plot use the above threee lines of code to execute the code and ensure
#that you comment the rest of the code for the time being as it runs the entire script from start for 50 iteartions.
#alternatively i have uploaded the code in colab as an ipynb to eliminate this problem by running peices of code in cells.
#here is the link for colab notebook https://drive.google.com/drive/folders/16fLmj5EAHvceJeIJsZucJpiQqEhLzPPp?usp=share_link 

#now we do the latent semantic indexing with number of topics as 35
lsi_model = models.LsiModel(vector_corpus, id2word=myDict, num_topics=35)  
#we transform the original corpus from bag of words to tdif vectors to now folded in lsi
lsi_model_corpus = lsi_model[vector_corpus] 

breakFlag = 0
#visualize checkpoint ---- check for lsi and check the result for num_topics value 35 in both the lsi corpus and original corpus
for i,j in zip(lsi_model_corpus,speech_corpus):
  print("Mixing Proportions(lsi_corpus values and corresponding Text from the speech")
  print(i,j)
  breakFlag = breakFlag + 1
  if(breakFlag>4):
    break
  

print('=========================')  
print("LSI MODEL")
print('=========================')

#now we plot wordclouds as suggested in the gensim library to visualize our topic results
#we use 50 words as max limit or each of the 35 topics
from wordcloud import WordCloud 
lsi_topics = lsi_model.print_topics(35,50)
x, y = plt.subplots(6, 6, sharex='col', sharey='row', figsize=(25,25))

for i in range(35):
    wordDict = {}
    for word in lsi_topics[i][1].split(" ")[1:]:
        if word != '+':
            wordDict[word.split('*')[1]] = (float)(word.split('*')[0])
    wordcloud = WordCloud(max_words=1000, contour_width=2, contour_color='blue')
    wordcloud.generate_from_frequencies(wordDict)
    y[i//6][i%6].imshow(wordcloud)
    y[i//6][i%6].set_title('Topic '+str(i+1), fontdict=dict(size=12))
    plt.axis('off')
plt.show()
#Print out the resulting topics, each topic as a list of word coefficients
for i in range(35):
    print("Topic %s:"%(i+1), lsi_model.print_topic(i))
    print()


print('=========================')  
print("LDA MODEL")
print('=========================')

#on repeating earlier steps, we can re run the coherence model to the plot.
#attached plot in the report tells us that for LDA 35 is a more appropraite number for number of topics than 35 which was used for lsi.
lda_topic_model = models.LdaModel(vector_corpus, id2word=myDict, num_topics=25,minimum_probability=0)
lda_model_corpus = lda_topic_model[vector_corpus]

lda_topics = lda_topic_model.print_topics(25,50)
x, y = plt.subplots(5, 5, sharex='col', sharey='row', figsize=(25,25))

for i in range(25):
    wordDict = {}
    for word in lda_topics[i][1].split(" ")[1:]:
        if word != '+':
            wordDict[word.split('*')[1]] = (float)(word.split('*')[0])
    wordcloud = WordCloud(max_words=1000, contour_width=2, contour_color='blue')
    wordcloud.generate_from_frequencies(wordDict)
    y[i//5][i%5].imshow(wordcloud)
    y[i//5][i%5].set_title('Topic '+str(i+1), fontdict=dict(size=12))
    plt.axis('off')
plt.show()
#Print out the resulting topics, each topic as a list of word coefficients
for i in range(25):
    print("Topic %s:"%(i+1), lda_topic_model.print_topic(i))
    print()

###############
#Analyze topics for each decade using topic modeling by the decade

#So we basically redo the steps to do topic modeling but this time for each decade 



#for all decades in the 20th and 21st century
#please note during data preprocessing and visualizations, it was interpreted that the last data column had year 2012
#so we won't go past 2011-2020.

data_dictionary_decades = {}

decade_list=['1901-1910','1911-1920','1921-1930',
         '1931-1940','1941-1950','1951-1960',
         '1961-1970','1971-1980','1981-1990',
         '1991-2000','2001-2010','2011-2020']

#manually checked using hit and trial method to infer the start value for the coulumn year of the speech = 1991 (20th  century)

loop_start = 111 
#this loop handles the coulumn index to manually separate out the data for each decade.
for i in range(12):
  loop_break = loop_start + 10
  if(loop_start == 221):
    loop_break = 225
  df_decades=""
  #this loop uses the indexes we set to copy data for each decade from original df in to new df_decades
  for j in range(loop_start,loop_break):
    df_decades=df_decades+df['speech_text'][j]
  loop_start=loop_break
  data_dictionary_decades[decade_list[i]]=df_decades

#ensure if the data is distributed into decades by checking dictionary key values
print(data_dictionary_decades.keys())



#=========================
#now we preprocess the data dictionary we just created. 

stoplist = stopwords

dec_words = [[re.sub(r"[0-9]+","",word) 
          for word in preprocessData(speech) if word not in stoplist] 
          for speech in data_dictionary_decades.values()]

for word in dec_words:
    for token in word:
        freq[token] += 1

processed_decade_corpus = [[token for token in word 
                            if (freq[token] > 1 )] 
                            for word in dec_words]

while('' in processed_decade_corpus):
    processed_decade_corpus.remove('') 

# print(processed_decade_corpus[0]) 
print('=========================')

myDict_decade = corpora.Dictionary(processed_decade_corpus)
# print(myDict_decade)
print('=========================')

bow_corpus_decade = [myDict_decade.doc2bow(word) for word in processed_decade_corpus]
# print(bow_corpus[0]) 
print('=========================')


tfidf_decade = models.TfidfModel(bow_corpus_decade)
vector_corpus_decade = tfidf[bow_corpus_decade]

lda_model_decade = models.LdaModel(vector_corpus_decade, id2word=myDict_decade, num_topics=20,minimum_probability=0.0, passes = 20)
lda_corpus_decade=lda_model_decade[vector_corpus_decade]


#lda_model_decade.print_topics(20,50)


lda_decade_topics = lda_model_decade.print_topics(20,50)
x, y = plt.subplots(5, 5, sharex='col', sharey='row', figsize=(25,25))

# for i in range(20):
#     wordDict = {}
#     for word in lda_decade_topics[i][1].split(" ")[1:]:
#         if word != '+':
#             wordDict[word.split('*')[1]] = (float)(word.split('*')[0])
#     wordcloud = WordCloud(max_words=1000, contour_width=2, contour_color='blue')
#     wordcloud.generate_from_frequencies(wordDict)
#     y[i//5][i%5].imshow(wordcloud)
#     y[i//5][i%5].set_title('Topic '+str(i+1), fontdict=dict(size=12))
#     plt.axis('off')
# for i in range(20):
#     wordDict = {}
#     for word in lda_decade_topics[i][1].split(" ")[1:]:
#         if word != '+':
#             word_weight = (float)(word.split('*')[0])
#             if word_weight > 0.0:
#                 wordDict[word.split('*')[1]] = word_weight

#     # Check if there is at least one word with non-zero weight
#     if len(wordDict) > 0:
#         wordcloud = WordCloud(max_words=1000, contour_width=2, contour_color='blue')
#         wordcloud.generate_from_frequencies(wordDict)
#         y[i // 5][i % 5].imshow(wordcloud)
#         y[i // 5][i % 5].set_title('Topic ' + str(i + 1), fontdict=dict(size=12))
#         plt.axis('off')
# plt.show()
#Print out the resulting topics, each topic as a list of word coefficients
for i in range(12):
    print("Topic %s:"%(i+1), lda_model_decade.print_topic(i))
    print()

for i in range(12):
  print(decade_list[i],lda_corpus_decade[i]) 
  print()
  print()