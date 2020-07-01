#!/usr/bin/env python
# coding: utf-8

# # RTI CDS Analytics Exercise 02
# 
# ## Javad Roostaei, June 2020

# # Part 1: Working on the XML file

# ## Step 0: Loading the XML file and Converting it to a Panda dataframe

# In[1]:


# Loading necessary Python Libraries
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# libraries for plotting 
from matplotlib import pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')

# libraries for mapping
import geopandas as gpd
import descartes
from shapely.geometry import Point, Polygon


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

# Check the shape of the data
AviationData_df.shape


# ## Step 1: Evaluate the dataset and define the data type

# In[5]:


# Show the first five rows of the data frame
AviationData_df.head()


# In[6]:


print(AviationData_df.columns)

# Store columns as a list
col_list = list(AviationData_df.columns)


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


# In[12]:


# Describe the numerical variables dataset
print(AviationData_df.describe())


#  ## Step2: Present some graphs for the numerical variables 

# In[17]:


# Make a boxplot for the Number of Engines
ax = sb.boxplot(data=AviationData_df['NumberOfEngines'], orient="h", palette="Set2")
plt.title('Boxplot for Number Of Engines')
plt.savefig('0.Boxplot_NumberOfEngines.png', dpi=300)


# In[18]:


Injuries_vars = AviationData_df[['TotalFatalInjuries','TotalSeriousInjuries','TotalMinorInjuries','TotalUninjured']]
# Make a boxplot for the Injuries variable
ax = sb.boxplot(data=Injuries_vars, orient="h", palette="Set2")
plt.title('Boxplot for Injuries')
plt.savefig('1.Boxplot_Injuries.png', dpi=300)
# This graph show how far the extreme values are from most of the data 


# In[19]:


sb.pairplot(AviationData_df)
# Pairplot draw scatterplots for joint relationships and histograms for univariate distributions
# Sometimes it is easy to see by pair plot  how the numerical variables are scattered pairwise 
# Also, we can spot outliers in these types of graphs as well. 
plt.title('pairplot for all numerical variables')
plt.savefig('2.pairplot.png', dpi=800)


# ### A. First some analysis based on “EventDate” variable

# In[20]:


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


# In[22]:


# create different pivot tables for injuries columns 
TotalFatalInjuries_df = pd.pivot_table(AviationData_df2,index=['Flight_Year'],values=['TotalFatalInjuries'],aggfunc=[np.sum, np.min, np.mean, np.max, len])
TotalSeriousInjuries_df = pd.pivot_table(AviationData_df2,index=['Flight_Year'],values=['TotalSeriousInjuries'],aggfunc=[np.sum, np.min, np.mean, np.max, len])
TotalMinorInjuries_df = pd.pivot_table(AviationData_df2,index=['Flight_Year'],values=['TotalMinorInjuries'],aggfunc=[np.sum, np.min, np.mean, np.max, len])
TotalUninjured_df = pd.pivot_table(AviationData_df2,index=['Flight_Year'],values=['TotalUninjured'],aggfunc=[np.sum, np.min, np.mean, np.max, len])

# Show one of them as a sample
TotalFatalInjuries_df
# len is the number of flight accident recorded in each year


# In[23]:


temp_df = pd.DataFrame(TotalFatalInjuries_df['sum'], TotalFatalInjuries_df.index)
temp_df2 = pd.DataFrame(TotalSeriousInjuries_df['sum'], TotalSeriousInjuries_df.index)
temp_df4 = pd.DataFrame(TotalSeriousInjuries_df['len'], TotalSeriousInjuries_df.index)


# In[24]:


plt.rcParams["figure.figsize"] = [12, 6]

temp_df.plot(kind='bar', color='grey')

plt.xlabel('Flight Year')
plt.ylabel('Sum of Total Injuries Reported')
plt.title('Sum of Total Injuries Reported Per Year')
plt.legend()
plt.savefig('3.Injuries.png', dpi=300)

temp_df2.plot(kind='bar', color='orange')

plt.xlabel('Flight Year')
plt.ylabel('Sum of Total Serious Injuries Reported')
plt.title('Sum of Total Serious Injuries Reported Per Year')
plt.legend()
plt.savefig('4.SeriousInjuries.png', dpi=300)


temp_df4.plot(kind='bar', color='lightgreen')

plt.xlabel('Flight Year')
plt.ylabel('Sum of Total Incidents with Serious Injuries Reported')
plt.title('Sum of Total Incidents with Serious Injuries Reported Per Year')
plt.legend()
plt.savefig('5.IncidentSeriousInjuries.png', dpi=300)


# In[25]:


plt.rcParams["figure.figsize"] = [8, 6]
# Create a line plot for counting the injuries (fatal and non-fatal) in each year 
plt.plot(TotalFatalInjuries_df.index, TotalFatalInjuries_df['sum'], label='Sum of Total Fatal Injuries')
plt.plot(TotalSeriousInjuries_df.index, TotalSeriousInjuries_df['sum'], label='Sum of Total Serious Injuries')
plt.plot(TotalSeriousInjuries_df.index, TotalMinorInjuries_df['len'], label='Total number of incident with Serious Injuries')

plt.xticks(rotation=0, size =15)
plt.yticks(rotation=0, size =16)
plt.xlabel('Years', size=18)
plt.ylabel('Sum of Total Injuries Reported', size=18)
plt.title('Sum of Total Injuries Reported Per Year', size=18)
plt.legend(prop={'size': 12}, loc = 1)
plt.savefig('6.TotalInjuries_Incidents.png', dpi=1000)


# In[23]:


# Group by year
df_by_year = AviationData_df.groupby('Flight_Year')
df_by_year.describe().head(10)


# In[26]:


# Create a line plot for counting the injuries (fatal and non-fatal) in each year 
plt.plot(TotalFatalInjuries_df.index, TotalFatalInjuries_df['sum'], label='Sum of Total Fatal Injuries')
plt.plot(TotalSeriousInjuries_df.index, TotalSeriousInjuries_df['sum'], label='Sum of Total Serious Injuries')
plt.plot(TotalMinorInjuries_df.index, TotalMinorInjuries_df['sum'], label='Sum of Total Minor Injuries')

plt.xlabel('Years', size=18)
plt.ylabel('Sum of Total Injuries Reported', size=18)
plt.title('Sum of Total Injuries Reported Per Year', size=18)
plt.legend()
plt.savefig('7.injeriesallyears.png', dpi=300)


# In[27]:


plt.plot(TotalUninjured_df.index, TotalUninjured_df['sum'], label='Sum of Total Uninjured')

plt.xlabel('Years')
plt.ylabel('Sum of Total Uninjured Reported')
plt.title('Sum of Total Uninjured Reported Per Year')
plt.legend()
plt.savefig('8.Uninjuredallyears.png', dpi=300)


# In[28]:


# Make a pivot table around Flight_Months
pd.pivot_table(AviationData_df,index=['Flight_Month'])


# In[29]:


# creat diffrent pivot table for injuries columns and month
TotalFatalInjuries_m_df = pd.pivot_table(AviationData_df2,index=['Flight_Month'],values=['TotalFatalInjuries'],aggfunc=[np.sum, np.min, np.mean, np.max, len])
TotalFatalInjuries_m_df


# In[30]:


temp_df3 = pd.DataFrame(TotalFatalInjuries_m_df['len'], TotalFatalInjuries_m_df.index)
temp_df3


# In[31]:


plt.rcParams["figure.figsize"] = [10, 6]

temp_df3.plot(kind='bar', color='blue', legend=None).grid(axis='y')
plt.xticks(rotation=0, size =15)
plt.yticks(rotation=0, size =16)
plt.xlabel('Flight Month', size=18)
plt.ylabel('Total Fatal Injuries', size=18)
plt.title('Total Incidents Reported Per Months(1982-2015)', size=22)
plt.savefig('9.Incidents_month.png', dpi=300)


# In[26]:


temp_df4 = pd.DataFrame(TotalFatalInjuries_m_df['sum'], TotalFatalInjuries_m_df.index)
temp_df4


# In[32]:


plt.rcParams["figure.figsize"] = [10, 6]

temp_df3.plot(kind='bar', color='lightblue', legend=None).grid(axis='y')
plt.xticks(rotation=0, size =15)
plt.yticks(rotation=0, size =16)
plt.xlabel('Flight Month', size=18)
plt.ylabel('Total Fatal Injuries', size=18)
plt.title('Total Injuries Reported Per Months(1982-2015)', size=22)
plt.savefig('10.Injuries_month.png', dpi=300)


# ### B. Present graphs for some of the categorical variables 

# In[33]:


# Plot the top 50 factory that makes airplain 
AviationData_df['Make'].value_counts()[:50].plot(kind='bar',figsize = (15,15), title='Top 50 companies that make airplanes which had accidents')
plt.savefig('11.Top 50 companies that make airplanes.png', dpi=300)


# In[34]:


plt.figure(figsize=(10,8))
# or we can use simpler version by using countplot at seaborn
p = sb.countplot(data=AviationData_df, y = 'PurposeOfFlight',
                order=AviationData_df['PurposeOfFlight'].value_counts().index)

# set labels
plt.xlabel("Count in Purpose Of Flight", size=12)
plt.ylabel("Purpose Of Flight", size=12)
plt.title("Number of Incidents Based on the Purpose Of Flight", size=15)
plt.savefig("12.PurposeOfFlight.png", dpi=300)


# In[35]:


plt.figure(figsize=(6,4))
# or we can use simpler version by using countplot at seaborn
p = sb.countplot(data=AviationData_df, y = 'PurposeOfFlight',
                order=AviationData_df['PurposeOfFlight'].value_counts().iloc[:5].index)

# set labels
plt.xlabel("Count in Purpose Of Flight", size=12)
plt.ylabel("Purpose Of Flight", size=12)
plt.title("Number of Incidents Based on the Purpose Of Flight Top 5", size=10)
plt.savefig("13.PurposeOfFlight_top5.png", dpi=300)


# In[36]:


# or we can use simpler version by using countplot at seaborn
p = sb.countplot(data=AviationData_df, y = 'BroadPhaseOfFlight',
                order=AviationData_df['BroadPhaseOfFlight'].value_counts().iloc[:5].index)
# set labels
plt.xlabel("Count in Each Phase", size=12)
plt.ylabel("Broad Phase of Flight", size=12)
plt.title("Top 5 Number of Incidents Based on the Phase of Flights", size=15)
plt.savefig("14.Top5_BroadPhaseOfFlight.png", dpi=300)


# In[37]:


# or we can use simpler version by using countplot at seaborn
p = sb.countplot(data=AviationData_df, y = 'BroadPhaseOfFlight',
                order=AviationData_df['BroadPhaseOfFlight'].value_counts().index)
# set labels
plt.xlabel("Count in Each Phase", size=12)
plt.ylabel("Broad Phase of Flight", size=12)
plt.title("Number of Incidents Based on the Phase of Flights", size=15)
plt.savefig("15.BroadPhaseOfFlight.png", dpi=300)


# In[40]:


# or we can use simpler version by using countplot at seaborn
p = sb.countplot(data=AviationData_df, x = 'WeatherCondition',
                order=AviationData_df['WeatherCondition'].value_counts().index)
# set labels
plt.xlabel("Count the Weather Condition", size=12)
plt.ylabel("Weather Condition", size=12)
plt.title("Number of Incidents Based on the Weather Condition", size=15)
plt.savefig("16.WeatherCondition.png", dpi=300)


# In[41]:


sb.countplot(x= AviationData_df["InvestigationType"])
plt.title("Investigation Type")
plt.savefig("17.InvestigationType.png", dpi=300)


# In[42]:


sb.countplot(x= AviationData_df["InjurySeverity"],
            order=AviationData_df['InjurySeverity'].value_counts().iloc[:10].index)
plt.title("Injury Severity Type")
plt.savefig("18.Injury Severity Type top10.png", dpi=300)


# In[43]:


# or we can use simpler version by using countplot at seaborn
p = sb.countplot(data=AviationData_df, y = 'AircraftCategory',
                order=AviationData_df['AircraftCategory'].value_counts().iloc[:5].index)
# set labels
plt.xlabel("Count", size=12)
plt.ylabel("Aircraft Category", size=12)
plt.title("Top 5 Number of Incidents Based on Aircraft Category", size=15)
plt.savefig("19.Top5_AircraftCategory.png", dpi=300)


# In[44]:


sb.countplot(x= AviationData_df["AircraftDamage"])
plt.title("Aircraft Damage Type")
plt.savefig("20.Aircraft Damage Type.png", dpi=300)


# ### C. Use mapping tools in python to visualize variables based on location

# In[46]:


World_map = gpd.read_file(r'D:\RTI_Exercise\data-scientist-exercise02-master\Shapefile\World.shp')


# In[47]:


geometry = [Point(xy) for xy in zip(AviationData_df["Longitude"], AviationData_df["Latitude"])]


# In[48]:


geo_df = gpd.GeoDataFrame(AviationData_df, crs = 'EPSG: 4326', geometry = geometry)
geo_df.head()


# In[50]:


fig,ax = plt.subplots(figsize = (18,12))
World_map.plot(ax = ax, alpha = 5.0, color="grey")
geo_df[geo_df['InvestigationType'] == "Accident"].plot(ax = ax, markersize = 5, color = "blue", marker = "o", label = "Accident")
geo_df[geo_df['InvestigationType'] == "Incident"].plot(ax = ax, markersize = 2, color = "red", marker = "^", label = "Incident")
plt.legend(prop={'size': 15})
plt.title("Map of Investigation Type", size=15)
plt.savefig("21.InvestigationType_Map.png", dpi=600)


# In[51]:


fig,ax = plt.subplots(figsize = (18,12))
World_map.plot(ax = ax, alpha = 5.0, color="grey")
geo_df[geo_df['WeatherCondition'] == "VMC"].plot(ax = ax, markersize = 5, color = "blue", marker = "o", label = "Visual meteorological conditions (VMC)")
geo_df[geo_df['WeatherCondition'] == "IMC"].plot(ax = ax, markersize = 2, color = "red", marker = "^", label = "Instrument meteorological conditions (IMC)")
geo_df[geo_df['WeatherCondition'] == "UNK"].plot(ax = ax, markersize = 3, color = "orange", marker = "o", label = "Unknown (UNK)")
plt.legend(prop={'size': 15}, loc = 1)
plt.title("Map of reported weather condition during the incidents", size=20)
plt.savefig("22.WeatherCondition_Map.png", dpi=800)


#