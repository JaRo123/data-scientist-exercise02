#!/usr/bin/env python
# coding: utf-8

# # RTI CDS Analytics Exercise 02
# 
# ## Javad Roostaei, June 2020

# # Part 2: Exploratory data analysis and natural language processing on JSON files

# ## Step 0: Loading JSON files and Converting it to a Panda dataframe

# In[ ]:

import pandas as pd
import numpy as np

# libraries for plotting 
from matplotlib import pyplot as plt
import json

import nltk
import re 
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
stopwords = nltk.corpus.stopwords.words('english')

# In[2]:


import os
	
# Change work directory
os.chdir(r"D:\RTI_Exercise\data-scientist-exercise02-master\data") #You need to change this to your local folder
print("Current Working Directory " , os.getcwd())


# In[52]:


json_df = []
df_col=['EventId','narrative','probable_cause']
json_df = pd.DataFrame(columns = df_col)
json_df.head


# In[53]:


for filename in os.listdir(os.getcwd()):
    if filename.startswith("N"):
        with open(filename) as read_file:
            r_json = json.load(read_file)
            tmp_json_df = pd.DataFrame(r_json['data'])
        json_df = json_df.append(tmp_json_df)


# In[54]:


json_df.shape


# In[55]:


# replace empty values with numpy nan
json_df = json_df.replace('',np.nan)


# In[56]:


# count the number of nan values in each column
print(json_df.isnull().sum())


# In[57]:


# Store columns as a list
col_list2 = list(json_df.columns)
col_list2


# ## Step 1. Some initial Text Analysis and Natural Language Processing

# ### For the "narrative" column

# In[58]:


#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

import nltk
import re 
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
stopwords = nltk.corpus.stopwords.words('english')


# In[59]:


# Remove rows containes NA from the narrative
json_df['narrative'].dropna(inplace=True)
json_df['narrative']


# ### Regular Expressions Preprocessing

# In[60]:


# A function for Regular Expressions Preprocessing
def preprocess(text):
    """ This function is for Regular Expressions Preprocessing
         to remove some charachters and also make lowecase etc"""
    clean_data = []
    for x in (text[:]): 
        new_text = re.sub('<.*?>', '', x)   # remove HTML tags
        new_text = re.sub(r'[^\w\s]', '', new_text) # remove punc.
        new_text = re.sub(r'\d+','',new_text)# remove numbers
        new_text = new_text.lower() # lower case, .upper() for upper          
        if new_text != '':
            clean_data.append(new_text)
    return clean_data


# In[61]:


# Regular Expressions Preprocessing
regular_ls = preprocess(json_df['narrative'])
print(len(regular_ls))
regular_ls[0]


# ### Make a list of all words

# In[62]:


words_ls = [] # an empty list to store all words

# this for loop goes through the above dataframe and makes a list
for string in regular_ls:
    words_ls.extend(string.split())


# In[63]:


print(len(words_ls))
print(words_ls[:20]) # top 20 words


# In[64]:


#remove stopwords
words_ls_NSW = [word for word in words_ls if word not in stopwords] 


# In[65]:


print(len(words_ls_NSW))
print(words_ls_NSW[:20]) # top 20 words


# In[66]:


freq_dist = nltk.FreqDist(words_ls_NSW)


# In[67]:


freq_dist


# In[68]:


freq_dist_df = pd.DataFrame(freq_dist.items(), columns=['word', 'frequency'])
freq_dist_df = freq_dist_df.sort_values(by=['frequency'], ascending=False)
freq_dist_df[0:20]


# ### Creat word clould

# In[69]:


# Try with diffrent method
# A function to remove stopwords and make a frequency dataframe 
def re_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in texts]

# run the function for the preprocessed Regular Expressions 
regular_ls_nsw = re_stopwords(regular_ls)

#make a dataframe
regular_ls_nsw_df = pd.DataFrame(regular_ls_nsw)

# Calculate the word frequency for the narrative columns after removing the stop words 
wo_fr = []
wo_fr =regular_ls_nsw_df.stack().value_counts()  


# In[92]:


plt.figure(figsize=(8,6))
df = wo_fr[0:20]
df.plot(kind='barh',color='grey')

plt.xticks(rotation=0, size =12)
plt.yticks(rotation=0, size =12)
plt.xlabel("Frequency", size=12)
plt.ylabel("Top 20 most frequent words", size=12)
plt.title("The top 20 most frequent words in narrative column", size=15)
plt.savefig("23.Top20FrequentWord.png", dpi=300)


# In[72]:


#Turn all items in a Dataframe to strings
wo_fr_str = wo_fr.to_string() 


# In[74]:


# Create and generate a word cloud image:
wordcloud = WordCloud(max_font_size=100, max_words=50, background_color="white").generate(wo_fr_str)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# Save the image in the img folder:
wordcloud.to_file("24.Top50FrequentWordCloud.png")
plt.show()


# ### Sentence Tokenization

# In[75]:


# A Sentence Tokenization function
def tokenization_s(sentences):
    s_new = []
    for sent in (sentences[:]): 
        s_token = sent_tokenize(sent)
        if s_token != '':
            s_new.append(s_token)
    return s_new


# In[76]:


tokenization_ls = tokenization_s(json_df['narrative'])
print(len(tokenization_ls))
tokenization_ls


# ### Stemming

# In[77]:


from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer


# In[78]:


#create an object of class PorterStemmer
porter = PorterStemmer() #PorterStemmer uses Suffix Stripping to produce stems
lancaster=LancasterStemmer()


# In[79]:


for word in regular_ls_nsw[0]: #A list of words to be stemmed
    print("{0:20}{1:20}{2:20}".format(word,porter.stem(word),lancaster.stem(word)))


# ### Lemmatization

# In[80]:


lemmatizer = WordNetLemmatizer()
def lemmatization(words):
    new = []
    lem_words = [lemmatizer.lemmatize(x) for x in (words[:][0])]
    new.append(lem_words)
    return new


# In[81]:


lemtest = lemmatization(regular_ls_nsw[:])
print(lemtest)


# ### For the "probable_cause" column

# In[82]:


# Remove rows containes NA from the narrative
json_df['probable_cause'].dropna(inplace=True)
json_df['probable_cause']


# In[83]:


#Regular Expressions Preprocessing
regular_ls_PC = preprocess(json_df['probable_cause'])
print(len(regular_ls_PC))
regular_ls_PC[0:5]


# In[84]:


# run the function for the preprocessed Regular Expressions 
regular_ls_PC_nsw = re_stopwords(regular_ls_PC)


# In[85]:


# show the results
print(len(regular_ls_PC_nsw))
regular_ls_PC_nsw[0]


# In[86]:


regular_ls_PC_nsw_df = pd.DataFrame(regular_ls_PC_nsw)
regular_ls_PC_nsw_df.shape


# In[87]:


# Calculate the word frequency for the probable_cause columns after removing the stop words 
wo_fr2 = []
wo_fr2 =regular_ls_PC_nsw_df.stack().value_counts()  


# In[88]:


# The top 20 most frequent words that are used in probable_cause columns 
wo_fr2[0:20]


# In[91]:


plt.figure(figsize=(8,6))
df2 = wo_fr2[0:20]
df.plot(kind='barh',color='orange')

plt.xticks(rotation=0, size =12)
plt.yticks(rotation=0, size =12)
plt.xlabel("Frequency", size=12)
plt.ylabel("Top 20 most frequent words", size=12)
plt.title("The top 20 most frequent words in probable_cause column", size=15)
plt.savefig("25.Top20FrequentWord_PC.png", dpi=300)


# ### Creat word clould

# In[95]:


#Turn all items in a Dataframe to strings
wo_fr2_str = wo_fr2.to_string() 


# In[97]:


# Create and generate a word cloud image:
wordcloud = WordCloud(max_font_size=100, max_words=50, background_color="white").generate(wo_fr2_str)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# Save the image in the img folder:
wordcloud.to_file("27.Top50FrequentWordCloud2.png")
plt.show()
