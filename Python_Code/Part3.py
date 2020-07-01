# # RTI CDS Analytics Exercise 02
# 
# ## Javad Roostaei, June 2020

# # Part 3: Join the XML and JSON files for further analysis

# ### Use merge left to joint the XML and JSON Data frames based on "EventId" columns

# In[98]:

# Loading necessary Python Libraries
import xml.etree.ElementTree as ET
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


# In[3]:


# reading and parsing XML Section
tree = ET.parse('AviationData.xml')
root = tree.getroot()
print(root)


# In[4]:


Xml_Data = [] # empty list to store the rows

# For loop to go through each row and append the empty list
for row in root[0]: 
    Xml_Data.append(row.attrib)
    
# Make pandas DataFrame
AviationData_df = pd.DataFrame(Xml_Data)


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


# In[7]:


# replace empty values with numpy nan
AviationData_df = AviationData_df.replace('',np.nan)


# In[8]:


# Convert date variables to a correct format
AviationData_df['EventDate'] = pd.to_datetime(AviationData_df['EventDate'],format='%m/%d/%Y',errors='coerce')
AviationData_df['PublicationDate'] = pd.to_datetime(AviationData_df['PublicationDate'],format='%m/%d/%Y',errors='coerce')


# In[9]:


# Convert numerical variables to a numeric format
numerical_var = ['Latitude','Longitude','NumberOfEngines','TotalFatalInjuries','TotalSeriousInjuries','TotalMinorInjuries','TotalUninjured']
for i in numerical_var:
    AviationData_df[i] = pd.to_numeric(AviationData_df[i], errors='coerce')


# In[10]:


# count the number of nan values in each column
print(AviationData_df.isnull().sum())


# In[11]:


# count the number of nan values in all dataframe
total_miss = AviationData_df.isnull().sum().sum()
total_miss




# Drop the three rows in the EventDate Columns with NAN value
AviationData_df = AviationData_df.dropna(subset=['EventDate'])

# Extract years information from the EventDate column and add it to the dataframe
AviationData_df['Flight_Year'] = pd.DatetimeIndex(AviationData_df['EventDate']).year.astype(int)

# Extract months information from the EventDate column and add it to the dataframe
AviationData_df['Flight_Month'] = pd.DatetimeIndex(AviationData_df['EventDate']).month.astype(int)


# In[21]:


# Drop years before 1982
AviationData_df2 = AviationData_df[AviationData_df['Flight_Year'] > 1981]
AviationData_df2.shape

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



XML_JSON_df = AviationData_df.merge(json_df, how='left')
XML_JSON_df.shape


# In[99]:


# Show the first five rows of the data frame
XML_JSON_df.head()


# In[100]:


print(XML_JSON_df.columns)

# Store columns as a list
col_list = list(XML_JSON_df.columns)


# ### Do some analysis for most recent incidents from 2010 - 2015

# In[101]:


most_recent_df = XML_JSON_df[(XML_JSON_df['Flight_Year'] > 2010) & (XML_JSON_df['Flight_Year'] <= 2015)]


# In[103]:


# Remove rows containes NA from the narrative
most_recent_df['narrative'].dropna(inplace=True)


# In[104]:


# Regular Expressions Preprocessing
regular_ls_recent = preprocess(most_recent_df['narrative'])
print(len(regular_ls_recent))
regular_ls_recent[0]


# In[105]:


words_ls_recent = [] # an empty list to store all words

# this for loop goes through the above dataframe and makes a list
for string in regular_ls_recent:
    words_ls_recent.extend(string.split())


# In[106]:


print(len(words_ls_recent))
print(words_ls_recent[:20]) # top 20 words


# In[107]:


#remove stopwords
words_ls_NSW2 = [word for word in words_ls_recent if word not in stopwords] 


# In[108]:


print(len(words_ls_NSW2))
print(words_ls_NSW2[:20]) # top 20 words


# In[125]:


freq_dist_recent_df = nltk.FreqDist(words_ls_NSW2)
print(len(freq_dist_recent_df))


# In[110]:


freq_dist_recent_df = pd.DataFrame(freq_dist_recent_df.items(), columns=['word', 'frequency'])
freq_dist_recent_df = freq_dist_recent_df.sort_values(by=['frequency'], ascending=False)
freq_dist_recent_df[0:20]


# ### Do some analysis for 2000 to 2005

# In[111]:


Yr2000_df = XML_JSON_df[(XML_JSON_df['Flight_Year'] > 2000) & (XML_JSON_df['Flight_Year'] <= 2055)]


# In[112]:


# Remove rows containes NA from the narrative
Yr2000_df['narrative'].dropna(inplace=True)


# In[113]:


# Regular Expressions Preprocessing
regular_ls_Yr2000 = preprocess(Yr2000_df['narrative'])
print(len(regular_ls_Yr2000))
regular_ls_Yr2000[0]


# In[114]:


words_ls_Yr2000 = [] # an empty list to store all words

# this for loop goes through the above dataframe and makes a list
for string in regular_ls_Yr2000:
    words_ls_Yr2000.extend(string.split())


# In[115]:


print(len(words_ls_Yr2000))
print(words_ls_Yr2000[:20]) # top 20 words


# In[116]:


#remove stopwords
words_ls_NSW3 = [word for word in words_ls_Yr2000 if word not in stopwords] 


# In[117]:


print(len(words_ls_NSW3))
print(words_ls_NSW3[:20]) # top 20 words


# In[118]:


freq_dist_recent_df2 = nltk.FreqDist(words_ls_NSW3)


# In[119]:


freq_dist_Yr2000_df = pd.DataFrame(freq_dist_recent_df2.items(), columns=['word', 'frequency'])
freq_dist_Yr2000_df = freq_dist_Yr2000_df.sort_values(by=['frequency'], ascending=False)
freq_dist_Yr2000_df[0:20]


# #### Observation: The top 20 words are mostly similar from set1: 2000 to 2005  and set2: 2010 to 2015

# ## Check if certain words appeared more 

# In[127]:



