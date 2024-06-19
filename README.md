---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.7
  nbformat: 4
  nbformat_minor: 5
---

::: {#292b61a0 .cell .markdown}
# Project Overview

In this project, we aim to analyze aviation data assisting a company
looking to diversify into the aircraft industry. Our goal is to identify
the lowest-risk aircraft for commercial and private operations. By
leveraging data cleaning, imputation, analysis, and visualization
techniques, we will provide insights to guide the company\'s
decision-making process.

## Business Problem

The company is venturing into the aviation sector and needs guidance on
selecting aircraft with minimal risk. As the analyst, your task is to
analyze the data to recommend the safest aircraft for the company\'s new
business venture. Your findings will be crucial in helping the head of
the aviation division make informed decisions on aircraft purchases.
:::

::: {#8c2b62f3-8467-4a02-afce-af224a6633f1 .cell .markdown}
## Importing Libraries
:::

::: {#63be342e-9270-43ca-9d96-996b7f5257b6 .cell .code execution_count="2"}
``` python
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Set display options
pd.set_option('display.max_rows', 100)  # Max rows to display
pd.set_option('display.max_columns', 40)  # Max columns to display
```
:::

::: {#22a43c54-994b-4066-92c9-723401eb19ae .cell .code execution_count="3"}
``` python
df = pd.read_csv(r'Data/AviationData.csv',encoding='windows-1252',low_memory=False)
```
:::

::: {#e50977ea-607e-4ce9-aa9d-7529f7ce7087 .cell .code execution_count="4"}
``` python
df.isna().sum()
```

::: {.output .execute_result execution_count="4"}
    Event.Id                      0
    Investigation.Type            0
    Accident.Number               0
    Event.Date                    0
    Location                     52
    Country                     226
    Latitude                  54507
    Longitude                 54516
    Airport.Code              38757
    Airport.Name              36185
    Injury.Severity            1000
    Aircraft.damage            3194
    Aircraft.Category         56602
    Registration.Number        1382
    Make                         63
    Model                        92
    Amateur.Built               102
    Number.of.Engines          6084
    Engine.Type                7096
    FAR.Description           56866
    Schedule                  76307
    Purpose.of.flight          6192
    Air.carrier               72241
    Total.Fatal.Injuries      11401
    Total.Serious.Injuries    12510
    Total.Minor.Injuries      11933
    Total.Uninjured            5912
    Weather.Condition          4492
    Broad.phase.of.flight     27165
    Report.Status              6384
    Publication.Date          13771
    dtype: int64
:::
:::

::: {#79bdee24-a2da-46a2-9dcd-4532de0048b8 .cell .markdown}
## getting column data type
:::

::: {#072124d4-e39e-4241-8522-600c1256b905 .cell .code execution_count="5"}
``` python
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 88889 entries, 0 to 88888
    Data columns (total 31 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Event.Id                88889 non-null  object 
     1   Investigation.Type      88889 non-null  object 
     2   Accident.Number         88889 non-null  object 
     3   Event.Date              88889 non-null  object 
     4   Location                88837 non-null  object 
     5   Country                 88663 non-null  object 
     6   Latitude                34382 non-null  object 
     7   Longitude               34373 non-null  object 
     8   Airport.Code            50132 non-null  object 
     9   Airport.Name            52704 non-null  object 
     10  Injury.Severity         87889 non-null  object 
     11  Aircraft.damage         85695 non-null  object 
     12  Aircraft.Category       32287 non-null  object 
     13  Registration.Number     87507 non-null  object 
     14  Make                    88826 non-null  object 
     15  Model                   88797 non-null  object 
     16  Amateur.Built           88787 non-null  object 
     17  Number.of.Engines       82805 non-null  float64
     18  Engine.Type             81793 non-null  object 
     19  FAR.Description         32023 non-null  object 
     20  Schedule                12582 non-null  object 
     21  Purpose.of.flight       82697 non-null  object 
     22  Air.carrier             16648 non-null  object 
     23  Total.Fatal.Injuries    77488 non-null  float64
     24  Total.Serious.Injuries  76379 non-null  float64
     25  Total.Minor.Injuries    76956 non-null  float64
     26  Total.Uninjured         82977 non-null  float64
     27  Weather.Condition       84397 non-null  object 
     28  Broad.phase.of.flight   61724 non-null  object 
     29  Report.Status           82505 non-null  object 
     30  Publication.Date        75118 non-null  object 
    dtypes: float64(5), object(26)
    memory usage: 21.0+ MB
:::
:::

::: {#932404f4-2bd2-4815-9d74-f1aa3f087957 .cell .markdown}
## Calculating the percentage of missing data
:::

::: {#b259c7aa-b285-457f-8485-3c1660004b71 .cell .code execution_count="6"}
``` python
percentage_missing = (df.isna().sum()/len(df)*100).round(2)
percentage_missing
```

::: {.output .execute_result execution_count="6"}
    Event.Id                   0.00
    Investigation.Type         0.00
    Accident.Number            0.00
    Event.Date                 0.00
    Location                   0.06
    Country                    0.25
    Latitude                  61.32
    Longitude                 61.33
    Airport.Code              43.60
    Airport.Name              40.71
    Injury.Severity            1.12
    Aircraft.damage            3.59
    Aircraft.Category         63.68
    Registration.Number        1.55
    Make                       0.07
    Model                      0.10
    Amateur.Built              0.11
    Number.of.Engines          6.84
    Engine.Type                7.98
    FAR.Description           63.97
    Schedule                  85.85
    Purpose.of.flight          6.97
    Air.carrier               81.27
    Total.Fatal.Injuries      12.83
    Total.Serious.Injuries    14.07
    Total.Minor.Injuries      13.42
    Total.Uninjured            6.65
    Weather.Condition          5.05
    Broad.phase.of.flight     30.56
    Report.Status              7.18
    Publication.Date          15.49
    dtype: float64
:::
:::

::: {#18989faa-e1ee-4a04-a391-009c29a919fa .cell .code execution_count="7"}
``` python
df.head()
```

::: {.output .execute_result execution_count="7"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Event.Id</th>
      <th>Investigation.Type</th>
      <th>Accident.Number</th>
      <th>Event.Date</th>
      <th>Location</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Airport.Code</th>
      <th>Airport.Name</th>
      <th>Injury.Severity</th>
      <th>Aircraft.damage</th>
      <th>Aircraft.Category</th>
      <th>Registration.Number</th>
      <th>Make</th>
      <th>Model</th>
      <th>Amateur.Built</th>
      <th>Number.of.Engines</th>
      <th>Engine.Type</th>
      <th>FAR.Description</th>
      <th>Schedule</th>
      <th>Purpose.of.flight</th>
      <th>Air.carrier</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
      <th>Total.Uninjured</th>
      <th>Weather.Condition</th>
      <th>Broad.phase.of.flight</th>
      <th>Report.Status</th>
      <th>Publication.Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20001218X45444</td>
      <td>Accident</td>
      <td>SEA87LA080</td>
      <td>1948-10-24</td>
      <td>MOOSE CREEK, ID</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Fatal(2)</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>NC6404</td>
      <td>Stinson</td>
      <td>108-3</td>
      <td>No</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Personal</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>UNK</td>
      <td>Cruise</td>
      <td>Probable Cause</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20001218X45447</td>
      <td>Accident</td>
      <td>LAX94LA336</td>
      <td>1962-07-19</td>
      <td>BRIDGEPORT, CA</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Fatal(4)</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>N5069P</td>
      <td>Piper</td>
      <td>PA24-180</td>
      <td>No</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Personal</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>UNK</td>
      <td>Unknown</td>
      <td>Probable Cause</td>
      <td>19-09-1996</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20061025X01555</td>
      <td>Accident</td>
      <td>NYC07LA005</td>
      <td>1974-08-30</td>
      <td>Saltville, VA</td>
      <td>United States</td>
      <td>36.922223</td>
      <td>-81.878056</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Fatal(3)</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>N5142R</td>
      <td>Cessna</td>
      <td>172M</td>
      <td>No</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Personal</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>IMC</td>
      <td>Cruise</td>
      <td>Probable Cause</td>
      <td>26-02-2007</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20001218X45448</td>
      <td>Accident</td>
      <td>LAX96LA321</td>
      <td>1977-06-19</td>
      <td>EUREKA, CA</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Fatal(2)</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>N1168J</td>
      <td>Rockwell</td>
      <td>112</td>
      <td>No</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Personal</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Cruise</td>
      <td>Probable Cause</td>
      <td>12-09-2000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20041105X01764</td>
      <td>Accident</td>
      <td>CHI79FA064</td>
      <td>1979-08-02</td>
      <td>Canton, OH</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Fatal(1)</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>N15NY</td>
      <td>Cessna</td>
      <td>501</td>
      <td>No</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Personal</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>VMC</td>
      <td>Approach</td>
      <td>Probable Cause</td>
      <td>16-04-1980</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#26c55828-ae21-499c-9a15-456e0c5e91fa .cell .markdown}
## retreiving only relevant column to the Project . {#retreiving-only-relevant-column-to-the-project-}
:::

::: {#f1485ba4-7a75-4ef1-9a07-e03ced7c5edc .cell .code execution_count="8"}
``` python
relevant_columns=[
    'Event.Date', 'Location', 'Country',
    'Injury.Severity', 'Aircraft.damage', 'Aircraft.Category',
    'Make', 'Model', 'Total.Fatal.Injuries', 'Total.Serious.Injuries',
    'Total.Minor.Injuries','Number.of.Engines','Engine.Type' , 'Total.Uninjured', 'Weather.Condition',
    'Broad.phase.of.flight'
]
```
:::

::: {#ad383b96-780c-4dfc-aba0-ab6684f464ee .cell .code execution_count="9"}
``` python
df = df[relevant_columns]
```
:::

::: {#b5f45c7d-b2e1-4209-9b1d-4f9e1b9a07a3 .cell .code execution_count="10"}
``` python
df.head()
```

::: {.output .execute_result execution_count="10"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Event.Date</th>
      <th>Location</th>
      <th>Country</th>
      <th>Injury.Severity</th>
      <th>Aircraft.damage</th>
      <th>Aircraft.Category</th>
      <th>Make</th>
      <th>Model</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
      <th>Number.of.Engines</th>
      <th>Engine.Type</th>
      <th>Total.Uninjured</th>
      <th>Weather.Condition</th>
      <th>Broad.phase.of.flight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1948-10-24</td>
      <td>MOOSE CREEK, ID</td>
      <td>United States</td>
      <td>Fatal(2)</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>Stinson</td>
      <td>108-3</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>UNK</td>
      <td>Cruise</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1962-07-19</td>
      <td>BRIDGEPORT, CA</td>
      <td>United States</td>
      <td>Fatal(4)</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>Piper</td>
      <td>PA24-180</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>UNK</td>
      <td>Unknown</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1974-08-30</td>
      <td>Saltville, VA</td>
      <td>United States</td>
      <td>Fatal(3)</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>Cessna</td>
      <td>172M</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>NaN</td>
      <td>IMC</td>
      <td>Cruise</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1977-06-19</td>
      <td>EUREKA, CA</td>
      <td>United States</td>
      <td>Fatal(2)</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>Rockwell</td>
      <td>112</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Cruise</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1979-08-02</td>
      <td>Canton, OH</td>
      <td>United States</td>
      <td>Fatal(1)</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>Cessna</td>
      <td>501</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>VMC</td>
      <td>Approach</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#989afeaf-bea8-4c4a-bd2f-5504276f6a8e .cell .markdown}
### rechecking missing value percentage in relevant columns
:::

::: {#03348c85-06dc-4ae7-86b8-eacfff4f9b07 .cell .code execution_count="11"}
``` python
def percentage_missing (df):
    percentages=[]
    percentages= (df.isna().sum()/len(df)*100).round(2)
    return percentages
percentage_missing(df)
```

::: {.output .execute_result execution_count="11"}
    Event.Date                 0.00
    Location                   0.06
    Country                    0.25
    Injury.Severity            1.12
    Aircraft.damage            3.59
    Aircraft.Category         63.68
    Make                       0.07
    Model                      0.10
    Total.Fatal.Injuries      12.83
    Total.Serious.Injuries    14.07
    Total.Minor.Injuries      13.42
    Number.of.Engines          6.84
    Engine.Type                7.98
    Total.Uninjured            6.65
    Weather.Condition          5.05
    Broad.phase.of.flight     30.56
    dtype: float64
:::
:::

::: {#cbf53709 .cell .markdown}
## Normalizing data
:::

::: {#29f849b7 .cell .markdown}
### Deleting data before 1982.As much of the data is missing. {#deleting-data-before-1982as-much-of-the-data-is-missing}
:::

::: {#992d1bae .cell .code execution_count="12"}
``` python

df = df.drop(index=list(df[df['Event.Date']<'1985'].index.values.tolist())).reset_index(drop=True)
```
:::

::: {#32dad725 .cell .markdown}
### Dropping all null values for columns with missing value percentage less than 10
:::

::: {#9c3f6cf5 .cell .code execution_count="13"}
``` python
df.dropna(subset=['Make','Model'],inplace=True)
```
:::

::: {#f2bafeac .cell .code execution_count="14"}
``` python
df['Model'].unique
```

::: {.output .execute_result execution_count="14"}
    <bound method Series.unique of 0          PA-34-200T
    1                310N
    2             727-225
    3        LM-1 "NIKKO"
    4                150J
                 ...     
    78271       PA-28-151
    78272            7ECA
    78273           8GCBC
    78274            210N
    78275       PA-24-260
    Name: Model, Length: 78185, dtype: object>
:::
:::

::: {#c2323bdf .cell .markdown}
### Normalizing text in Model column. {#normalizing-text-in-model-column}

we normalize text by converting it to uppercase,stripping leading and
trailing whitespace, and removing punctuation .
:::

::: {#a0e1b8a2 .cell .code execution_count="15"}
``` python
# Convert "Model" column to lowercase
df['Model'] = df['Model'].str.upper()

# Remove leading and trailing whitespaces
df['Model'] = df['Model'].str.strip()

# Replace multiple whitespaces with a single whitespace
df['Model'] = df['Model'].str.replace('-', '', regex=True)

```
:::

::: {#12962dc1-d55b-4aa3-9166-d628e62a1e3a .cell .code execution_count="16"}
``` python
df['Make'].unique
```

::: {.output .execute_result execution_count="16"}
    <bound method Series.unique of 0                             Piper
    1                            Cessna
    2                            Boeing
    3                              Fuji
    4                            Cessna
                        ...            
    78271                         PIPER
    78272                      BELLANCA
    78273    AMERICAN CHAMPION AIRCRAFT
    78274                        CESSNA
    78275                         PIPER
    Name: Make, Length: 78185, dtype: object>
:::
:::

::: {#70d42d68 .cell .markdown}
### Normalizing text in make column. {#normalizing-text-in-make-column}

define a function to normalize text by converting it to lowercase,
removing punctuation, and stripping leading and trailing whitespace. We
then apply this normalization to the \'Make\' column in the dataset.
:::

::: {#61635fcb-fc3f-4bfe-a82e-0dfe5a577ec6 .cell .code execution_count="17"}
``` python
# Define a function to normalize text
def normalize_text(text):
    # Convert to lowercase
    text = text.title()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Strip leading and trailing whitespace
    text = text.strip()
    return text

# Create a new column for the normalized values
df['Make'] = df['Make'].apply(normalize_text)
```
:::

::: {#41934f28 .cell .code execution_count="18"}
``` python
df.isnull().sum()
```

::: {.output .execute_result execution_count="18"}
    Event.Date                    0
    Location                     48
    Country                     179
    Injury.Severity             979
    Aircraft.damage            2964
    Aircraft.Category         49504
    Make                          0
    Model                         0
    Total.Fatal.Injuries      11334
    Total.Serious.Injuries    12427
    Total.Minor.Injuries      11848
    Number.of.Engines          5916
    Engine.Type                7023
    Total.Uninjured            5871
    Weather.Condition          4438
    Broad.phase.of.flight     27082
    dtype: int64
:::
:::

::: {#629c0165-5d3a-4844-9473-bfbce252059e .cell .code execution_count="19"}
``` python
df.shape
percentage_missing(df)
```

::: {.output .execute_result execution_count="19"}
    Event.Date                 0.00
    Location                   0.06
    Country                    0.23
    Injury.Severity            1.25
    Aircraft.damage            3.79
    Aircraft.Category         63.32
    Make                       0.00
    Model                      0.00
    Total.Fatal.Injuries      14.50
    Total.Serious.Injuries    15.89
    Total.Minor.Injuries      15.15
    Number.of.Engines          7.57
    Engine.Type                8.98
    Total.Uninjured            7.51
    Weather.Condition          5.68
    Broad.phase.of.flight     34.64
    dtype: float64
:::
:::

::: {#3a847667-9e62-4a91-8cd4-3a33dd7841e5 .cell .markdown}
#### Applying it to the dataframe\'s Aircraft category column. {#applying-it-to-the-dataframes-aircraft-category-column}
:::

::: {#dfc4731f-cc7f-4dfc-b42f-cfce8f6f4846 .cell .code execution_count="20"}
``` python
df.head()
```

::: {.output .execute_result execution_count="20"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Event.Date</th>
      <th>Location</th>
      <th>Country</th>
      <th>Injury.Severity</th>
      <th>Aircraft.damage</th>
      <th>Aircraft.Category</th>
      <th>Make</th>
      <th>Model</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
      <th>Number.of.Engines</th>
      <th>Engine.Type</th>
      <th>Total.Uninjured</th>
      <th>Weather.Condition</th>
      <th>Broad.phase.of.flight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985-01-01</td>
      <td>HOPKINTON, NH</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>Piper</td>
      <td>PA34200T</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Approach</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985-01-01</td>
      <td>EDGEWOOD, NM</td>
      <td>United States</td>
      <td>Fatal(2)</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>Cessna</td>
      <td>310N</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Cruise</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985-01-01</td>
      <td>LA PAZ, Bolivia</td>
      <td>Bolivia</td>
      <td>Fatal(29)</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>Boeing</td>
      <td>727225</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Turbo Fan</td>
      <td>0.0</td>
      <td>UNK</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985-01-01</td>
      <td>ODESSA, FL</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>NaN</td>
      <td>Fuji</td>
      <td>LM1 "NIKKO"</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>VMC</td>
      <td>Maneuvering</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985-01-01</td>
      <td>DUBLIN, NC</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>NaN</td>
      <td>Cessna</td>
      <td>150J</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>1.0</td>
      <td>VMC</td>
      <td>Cruise</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#1abdf635 .cell .code execution_count="21"}
``` python
df.columns
```

::: {.output .execute_result execution_count="21"}
    Index(['Event.Date', 'Location', 'Country', 'Injury.Severity',
           'Aircraft.damage', 'Aircraft.Category', 'Make', 'Model',
           'Total.Fatal.Injuries', 'Total.Serious.Injuries',
           'Total.Minor.Injuries', 'Number.of.Engines', 'Engine.Type',
           'Total.Uninjured', 'Weather.Condition', 'Broad.phase.of.flight'],
          dtype='object')
:::
:::

::: {#ea325c39 .cell .code execution_count="22"}
``` python
df.dropna(subset=['Location', 'Country', 'Injury.Severity'],inplace=True)
```
:::

::: {#4e64b9cb .cell .code execution_count="23"}
``` python
percentage_missing(df)
```

::: {.output .execute_result execution_count="23"}
    Event.Date                 0.00
    Location                   0.00
    Country                    0.00
    Injury.Severity            0.00
    Aircraft.damage            3.18
    Aircraft.Category         63.90
    Make                       0.00
    Model                      0.00
    Total.Fatal.Injuries      14.69
    Total.Serious.Injuries    16.11
    Total.Minor.Injuries      15.36
    Number.of.Engines          6.73
    Engine.Type                8.09
    Total.Uninjured            7.60
    Weather.Condition          4.59
    Broad.phase.of.flight     33.85
    dtype: float64
:::
:::

::: {#10a83e63 .cell .markdown}
### Processing Injury Severity Data

well extract the number of fatal injuries and clean up the severity
descriptions in the dataset.The number of fatal injuries is added to a
new column and simplifies the severity descriptions by removing the
fatality numbers from the Injury.Severity column.

#### Steps:

1.  Define a function to extract the number of fatal injuries from the
    severity description.
2.  Apply the function to create a new column for extracted fatal
    injuries.
3.  Define a function to simplify the severity description by removing
    the fatality number.
4.  Apply the function to clean the Injury.Severity column.
:::

::: {#27d5bf75 .cell .code execution_count="24"}
``` python
# import re module .This module provides regular expression matching operations
import re

def extract_fatalities(severity, total_fatal_injuries):
    match = re.search(r'Fatal\((\d+)\)', severity)
    if match:
        return int(match.group(1))
    else:
        return total_fatal_injuries

# Applying the function to create a new column for extracted fatalities
df['Total.Fatal.Injuries'] = df.apply(lambda row: extract_fatalities(row['Injury.Severity'], row['Total.Fatal.Injuries']), axis=1)

# Function to remove the fatality number from the Injury.Severity column
def simplify_severity(severity):
    return re.sub(r'\(.*\)', '', severity).strip()

# Applying the function to clean the Injury.Severity column
df['Injury.Severity'] = df['Injury.Severity'].apply(simplify_severity)
```
:::

::: {#4f6e9f61 .cell .markdown}
#### Imputing weather condition , aircraft damage anull values with mode {#imputing-weather-condition--aircraft-damage-anull-values-with-mode}

The code fills missing values in the \'Weather.Condition\' and
\'Aircraft.damage\' columns with the most common value. It then converts
the \'Weather.Condition\' values to uppercase for consistency.
:::

::: {#9b9e1147 .cell .code execution_count="25"}
``` python
df['Weather.Condition'].fillna(df['Weather.Condition'].mode()[0], inplace=True)
df['Aircraft.damage'].fillna(df['Aircraft.damage'].mode()[0], inplace=True)
df['Weather.Condition']=df['Weather.Condition'].str.upper()
```
:::

::: {#d242f31c .cell .markdown}
### Handling Missing Values in Aircraft Data

Missing values in various columns related to aircraft injuries, engines,
engine type, and flight phase are filled using the mode of the
respective columns within the same \'Make\' and \'Model\' groups. This
approach ensures that missing data is imputed with the most common
values specific to each aircraft make and model.

#### Steps: {#steps}

1.  Impute missing values for \'Total.Fatal.Injuries\',
    \'Total.Serious.Injuries\', \'Total.Minor.Injuries\',
    \'Total.Uninjured\', \'Number.of.Engines\', \'Engine.Type\',
    \'Aircraft.Category\', and \'Broad.phase.of.flight\'.
2.  Group the data by \'Make\' and \'Model\' to calculate the mode for
    each column within these groups.
3.  Fill missing values in each column with the mode value corresponding
    to the respective \'Make\' and \'Model\' combination.
:::

::: {#0f347cb2 .cell .code execution_count="26"}
``` python
df['Total.Fatal.Injuries'] = df.groupby(['Make', 'Model'])['Total.Fatal.Injuries'].transform(lambda x: x.fillna(x.mode().max()))
df['Total.Serious.Injuries'] = df.groupby(['Make', 'Model'])['Total.Serious.Injuries'].transform(lambda x: x.fillna(x.mode().max()))
df['Total.Minor.Injuries'] = df.groupby(['Make', 'Model'])['Total.Minor.Injuries'].transform(lambda x: x.fillna(x.mode().max()))
df['Total.Uninjured']= df.groupby(['Make', 'Model'])['Total.Uninjured'].transform(lambda x: x.fillna(x.mode().max()))
df['Number.of.Engines'] = df.groupby(['Make', 'Model'])['Number.of.Engines'].transform(lambda x: x.fillna(x.mode().max()))
df['Engine.Type'] = df.groupby(['Make', 'Model'])['Engine.Type'].transform(lambda x: x.fillna(x.mode().max()))
df['Aircraft.Category'] = df.groupby(['Make'])['Aircraft.Category'].transform(lambda x: x.fillna(x.mode().max()))
df['Broad.phase.of.flight'] = df.groupby(['Make', 'Model'])['Broad.phase.of.flight'].transform(lambda x: x.fillna(x.mode().max()))

```
:::

::: {#efc95457 .cell .code execution_count="27"}
``` python
percentage_missing (df)
```

::: {.output .execute_result execution_count="27"}
    Event.Date                 0.00
    Location                   0.00
    Country                    0.00
    Injury.Severity            0.00
    Aircraft.damage            0.00
    Aircraft.Category          4.99
    Make                       0.00
    Model                      0.00
    Total.Fatal.Injuries       2.57
    Total.Serious.Injuries     2.88
    Total.Minor.Injuries       2.87
    Number.of.Engines          1.58
    Engine.Type                1.92
    Total.Uninjured            1.76
    Weather.Condition          0.00
    Broad.phase.of.flight     11.01
    dtype: float64
:::
:::

::: {#61706f8f .cell .code execution_count="28"}
``` python
df.head(5)
```

::: {.output .execute_result execution_count="28"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Event.Date</th>
      <th>Location</th>
      <th>Country</th>
      <th>Injury.Severity</th>
      <th>Aircraft.damage</th>
      <th>Aircraft.Category</th>
      <th>Make</th>
      <th>Model</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
      <th>Number.of.Engines</th>
      <th>Engine.Type</th>
      <th>Total.Uninjured</th>
      <th>Weather.Condition</th>
      <th>Broad.phase.of.flight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985-01-01</td>
      <td>HOPKINTON, NH</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Piper</td>
      <td>PA34200T</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Approach</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985-01-01</td>
      <td>EDGEWOOD, NM</td>
      <td>United States</td>
      <td>Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>310N</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Cruise</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985-01-01</td>
      <td>LA PAZ, Bolivia</td>
      <td>Bolivia</td>
      <td>Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Boeing</td>
      <td>727225</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Turbo Fan</td>
      <td>0.0</td>
      <td>UNK</td>
      <td>Cruise</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985-01-01</td>
      <td>ODESSA, FL</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Fuji</td>
      <td>LM1 "NIKKO"</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>VMC</td>
      <td>Maneuvering</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985-01-01</td>
      <td>DUBLIN, NC</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>150J</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>1.0</td>
      <td>VMC</td>
      <td>Cruise</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#a39939ff .cell .code execution_count="29"}
``` python
percentage_missing(df)
```

::: {.output .execute_result execution_count="29"}
    Event.Date                 0.00
    Location                   0.00
    Country                    0.00
    Injury.Severity            0.00
    Aircraft.damage            0.00
    Aircraft.Category          4.99
    Make                       0.00
    Model                      0.00
    Total.Fatal.Injuries       2.57
    Total.Serious.Injuries     2.88
    Total.Minor.Injuries       2.87
    Number.of.Engines          1.58
    Engine.Type                1.92
    Total.Uninjured            1.76
    Weather.Condition          0.00
    Broad.phase.of.flight     11.01
    dtype: float64
:::
:::

::: {#0e0c29a5 .cell .markdown}
#### Imputing the rest with mode. {#imputing-the-rest-with-mode}
:::

::: {#4833e580 .cell .code execution_count="30"}
``` python
df['Total.Serious.Injuries'] = df['Total.Serious.Injuries'].fillna(df['Total.Serious.Injuries'].mode()[0])
df['Total.Minor.Injuries'] = df['Total.Minor.Injuries'].fillna(df['Total.Minor.Injuries'].mode()[0])
df['Number.of.Engines'] = df['Number.of.Engines'].fillna(df['Number.of.Engines'].mode()[0])
df['Total.Uninjured'] = df['Total.Uninjured'].fillna(df['Total.Uninjured'].mode()[0])

```
:::

::: {#7f5bdb6a .cell .code execution_count="31"}
``` python
percentage_missing(df)
```

::: {.output .execute_result execution_count="31"}
    Event.Date                 0.00
    Location                   0.00
    Country                    0.00
    Injury.Severity            0.00
    Aircraft.damage            0.00
    Aircraft.Category          4.99
    Make                       0.00
    Model                      0.00
    Total.Fatal.Injuries       2.57
    Total.Serious.Injuries     0.00
    Total.Minor.Injuries       0.00
    Number.of.Engines          0.00
    Engine.Type                1.92
    Total.Uninjured            0.00
    Weather.Condition          0.00
    Broad.phase.of.flight     11.01
    dtype: float64
:::
:::

::: {#26a95aee .cell .markdown}
### Normalization of the Injury severity by extracting the casualties to the right column(Total.Fatatl.Injuries) {#normalization-of-the-injury-severity-by-extracting-the-casualties-to-the-right-columntotalfatatlinjuries}
:::

::: {#0043bda4 .cell .markdown}
#### droppping remaining null containing rows
:::

::: {#38dc1d9b .cell .code execution_count="32"}
``` python
df = df.dropna(subset=['Total.Fatal.Injuries', 'Engine.Type'])
```
:::

::: {#9220e122 .cell .code execution_count="33"}
``` python
percentage_missing(df)
```

::: {.output .execute_result execution_count="33"}
    Event.Date                0.00
    Location                  0.00
    Country                   0.00
    Injury.Severity           0.00
    Aircraft.damage           0.00
    Aircraft.Category         4.30
    Make                      0.00
    Model                     0.00
    Total.Fatal.Injuries      0.00
    Total.Serious.Injuries    0.00
    Total.Minor.Injuries      0.00
    Number.of.Engines         0.00
    Engine.Type               0.00
    Total.Uninjured           0.00
    Weather.Condition         0.00
    Broad.phase.of.flight     9.76
    dtype: float64
:::
:::

::: {#c99c5bdd .cell .code execution_count="34"}
``` python
df.head()
```

::: {.output .execute_result execution_count="34"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Event.Date</th>
      <th>Location</th>
      <th>Country</th>
      <th>Injury.Severity</th>
      <th>Aircraft.damage</th>
      <th>Aircraft.Category</th>
      <th>Make</th>
      <th>Model</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
      <th>Number.of.Engines</th>
      <th>Engine.Type</th>
      <th>Total.Uninjured</th>
      <th>Weather.Condition</th>
      <th>Broad.phase.of.flight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985-01-01</td>
      <td>HOPKINTON, NH</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Piper</td>
      <td>PA34200T</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Approach</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985-01-01</td>
      <td>EDGEWOOD, NM</td>
      <td>United States</td>
      <td>Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>310N</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Cruise</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985-01-01</td>
      <td>LA PAZ, Bolivia</td>
      <td>Bolivia</td>
      <td>Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Boeing</td>
      <td>727225</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Turbo Fan</td>
      <td>0.0</td>
      <td>UNK</td>
      <td>Cruise</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985-01-01</td>
      <td>ODESSA, FL</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Fuji</td>
      <td>LM1 "NIKKO"</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>VMC</td>
      <td>Maneuvering</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985-01-01</td>
      <td>DUBLIN, NC</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>150J</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>1.0</td>
      <td>VMC</td>
      <td>Cruise</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#ed6a62b1 .cell .code execution_count="35"}
``` python
df['Event.Date']= pd.to_datetime(df['Event.Date'],format='%Y-%m-%d')
df['Year']=df['Event.Date'].dt.year
df['Month']=df['Event.Date'].dt.month
df['Day']=df['Event.Date'].dt.day

df.head()
```

::: {.output .execute_result execution_count="35"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Event.Date</th>
      <th>Location</th>
      <th>Country</th>
      <th>Injury.Severity</th>
      <th>Aircraft.damage</th>
      <th>Aircraft.Category</th>
      <th>Make</th>
      <th>Model</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
      <th>Number.of.Engines</th>
      <th>Engine.Type</th>
      <th>Total.Uninjured</th>
      <th>Weather.Condition</th>
      <th>Broad.phase.of.flight</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985-01-01</td>
      <td>HOPKINTON, NH</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Piper</td>
      <td>PA34200T</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Approach</td>
      <td>1985</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985-01-01</td>
      <td>EDGEWOOD, NM</td>
      <td>United States</td>
      <td>Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>310N</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Cruise</td>
      <td>1985</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985-01-01</td>
      <td>LA PAZ, Bolivia</td>
      <td>Bolivia</td>
      <td>Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Boeing</td>
      <td>727225</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Turbo Fan</td>
      <td>0.0</td>
      <td>UNK</td>
      <td>Cruise</td>
      <td>1985</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985-01-01</td>
      <td>ODESSA, FL</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Fuji</td>
      <td>LM1 "NIKKO"</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>VMC</td>
      <td>Maneuvering</td>
      <td>1985</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985-01-01</td>
      <td>DUBLIN, NC</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>150J</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>1.0</td>
      <td>VMC</td>
      <td>Cruise</td>
      <td>1985</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#184b14ad .cell .markdown}
#### Calculate New columns to be used in the visualizations:
:::

::: {#3079eece .cell .code execution_count="36"}
``` python
df['Total.Injuries'] = df['Total.Fatal.Injuries'] + df['Total.Serious.Injuries'] + df['Total.Minor.Injuries']
df['Injury.Severity.Index'] = df['Total.Fatal.Injuries']*3 + df['Total.Serious.Injuries']*2 + df['Total.Minor.Injuries']
```
:::

::: {#fe83a683 .cell .code execution_count="37"}
``` python
df.sort_values(by='Event.Date').head(20)
```

::: {.output .execute_result execution_count="37"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Event.Date</th>
      <th>Location</th>
      <th>Country</th>
      <th>Injury.Severity</th>
      <th>Aircraft.damage</th>
      <th>Aircraft.Category</th>
      <th>Make</th>
      <th>Model</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
      <th>Number.of.Engines</th>
      <th>Engine.Type</th>
      <th>Total.Uninjured</th>
      <th>Weather.Condition</th>
      <th>Broad.phase.of.flight</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Total.Injuries</th>
      <th>Injury.Severity.Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985-01-01</td>
      <td>HOPKINTON, NH</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Piper</td>
      <td>PA34200T</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Approach</td>
      <td>1985</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985-01-01</td>
      <td>EDGEWOOD, NM</td>
      <td>United States</td>
      <td>Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>310N</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Cruise</td>
      <td>1985</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985-01-01</td>
      <td>LA PAZ, Bolivia</td>
      <td>Bolivia</td>
      <td>Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Boeing</td>
      <td>727225</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Turbo Fan</td>
      <td>0.0</td>
      <td>UNK</td>
      <td>Cruise</td>
      <td>1985</td>
      <td>1</td>
      <td>1</td>
      <td>29.0</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985-01-01</td>
      <td>ODESSA, FL</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Fuji</td>
      <td>LM1 "NIKKO"</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>VMC</td>
      <td>Maneuvering</td>
      <td>1985</td>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985-01-01</td>
      <td>DUBLIN, NC</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>150J</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>1.0</td>
      <td>VMC</td>
      <td>Cruise</td>
      <td>1985</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1985-01-02</td>
      <td>PAWNEE CITY, NE</td>
      <td>United States</td>
      <td>Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>150F</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>VMC</td>
      <td>Maneuvering</td>
      <td>1985</td>
      <td>1</td>
      <td>2</td>
      <td>2.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1985-01-02</td>
      <td>LORDSBURG, NM</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>210B</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>VMC</td>
      <td>Cruise</td>
      <td>1985</td>
      <td>1</td>
      <td>2</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1985-01-02</td>
      <td>YODER, CO</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Piper</td>
      <td>PA28181</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>3.0</td>
      <td>VMC</td>
      <td>Cruise</td>
      <td>1985</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1985-01-02</td>
      <td>MT STERLING, IL</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>172M</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>5.0</td>
      <td>VMC</td>
      <td>Landing</td>
      <td>1985</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1985-01-03</td>
      <td>SALT LAKE CITY, UT</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Destroyed</td>
      <td>Helicopter</td>
      <td>Aerospatiale</td>
      <td>SA315B</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Turbo Shaft</td>
      <td>0.0</td>
      <td>VMC</td>
      <td>Maneuvering</td>
      <td>1985</td>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1985-01-03</td>
      <td>MANHATTAN, MT</td>
      <td>United States</td>
      <td>Fatal</td>
      <td>Destroyed</td>
      <td>NaN</td>
      <td>Polliwagen</td>
      <td>2 PLACE</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>VMC</td>
      <td>Descent</td>
      <td>1985</td>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1985-01-03</td>
      <td>ELLISONORE, MO</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>182P</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>4.0</td>
      <td>IMC</td>
      <td>Cruise</td>
      <td>1985</td>
      <td>1</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1985-01-03</td>
      <td>SANTA BARBARA, CA</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Helicopter</td>
      <td>Bell</td>
      <td>20611</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Turbo Shaft</td>
      <td>2.0</td>
      <td>VMC</td>
      <td>Climb</td>
      <td>1985</td>
      <td>1</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1985-01-04</td>
      <td>NUIQSUT, AK</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>207A</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>2.0</td>
      <td>VMC</td>
      <td>Landing</td>
      <td>1985</td>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1985-01-04</td>
      <td>RAWLINS, WY</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Piper</td>
      <td>PA18150</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>0.0</td>
      <td>VMC</td>
      <td>Cruise</td>
      <td>1985</td>
      <td>1</td>
      <td>4</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985-01-04</td>
      <td>CAMARILLO, CA</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Cessna</td>
      <td>172RG</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>3.0</td>
      <td>VMC</td>
      <td>Landing</td>
      <td>1985</td>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985-01-04</td>
      <td>BILLINGS, MT</td>
      <td>United States</td>
      <td>Non-Fatal</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Beech</td>
      <td>B36TC</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Reciprocating</td>
      <td>1.0</td>
      <td>VMC</td>
      <td>Cruise</td>
      <td>1985</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1985-01-04</td>
      <td>ST. LOUIS, MO</td>
      <td>United States</td>
      <td>Incident</td>
      <td>Substantial</td>
      <td>Airplane</td>
      <td>Swearingen</td>
      <td>SA226TC</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Turbo Prop</td>
      <td>5.0</td>
      <td>VMC</td>
      <td>Cruise</td>
      <td>1985</td>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1985-01-04</td>
      <td>NEWARK, NJ</td>
      <td>United States</td>
      <td>Fatal</td>
      <td>Minor</td>
      <td>Helicopter</td>
      <td>Hughes</td>
      <td>500D</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Turbo Shaft</td>
      <td>2.0</td>
      <td>VMC</td>
      <td>Standing</td>
      <td>1985</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985-01-04</td>
      <td>WEST POINT, VA</td>
      <td>United States</td>
      <td>Fatal</td>
      <td>Destroyed</td>
      <td>Airplane</td>
      <td>Mitsubishi</td>
      <td>MU2B25</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Turbo Prop</td>
      <td>0.0</td>
      <td>IMC</td>
      <td>Approach</td>
      <td>1985</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#978b8b28 .cell .code execution_count="38"}
``` python
percentage_missing(df)
```

::: {.output .execute_result execution_count="38"}
    Event.Date                0.00
    Location                  0.00
    Country                   0.00
    Injury.Severity           0.00
    Aircraft.damage           0.00
    Aircraft.Category         4.30
    Make                      0.00
    Model                     0.00
    Total.Fatal.Injuries      0.00
    Total.Serious.Injuries    0.00
    Total.Minor.Injuries      0.00
    Number.of.Engines         0.00
    Engine.Type               0.00
    Total.Uninjured           0.00
    Weather.Condition         0.00
    Broad.phase.of.flight     9.76
    Year                      0.00
    Month                     0.00
    Day                       0.00
    Total.Injuries            0.00
    Injury.Severity.Index     0.00
    dtype: float64
:::
:::

::: {#40d963aa .cell .code execution_count="39"}
``` python
df['Aircraft.Category'] = df.groupby(['Make','Model'])['Aircraft.Category'].bfill()
df['Broad.phase.of.flight'] = df.groupby(['Make','Model'])['Broad.phase.of.flight'].bfill()
```
:::

::: {#dfe1ee2f .cell .code execution_count="40"}
``` python
df['Broad.phase.of.flight']=df['Broad.phase.of.flight'].fillna('Unknown')
df.dropna(subset=['Aircraft.Category'],axis=0 , inplace=True)
```
:::

::: {#bab44271 .cell .code execution_count="41"}
``` python
percentage_missing(df)
```

::: {.output .execute_result execution_count="41"}
    Event.Date                0.0
    Location                  0.0
    Country                   0.0
    Injury.Severity           0.0
    Aircraft.damage           0.0
    Aircraft.Category         0.0
    Make                      0.0
    Model                     0.0
    Total.Fatal.Injuries      0.0
    Total.Serious.Injuries    0.0
    Total.Minor.Injuries      0.0
    Number.of.Engines         0.0
    Engine.Type               0.0
    Total.Uninjured           0.0
    Weather.Condition         0.0
    Broad.phase.of.flight     0.0
    Year                      0.0
    Month                     0.0
    Day                       0.0
    Total.Injuries            0.0
    Injury.Severity.Index     0.0
    dtype: float64
:::
:::

::: {#c21b27d2 .cell .code execution_count="42"}
``` python
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    Index: 70563 entries, 0 to 78275
    Data columns (total 21 columns):
     #   Column                  Non-Null Count  Dtype         
    ---  ------                  --------------  -----         
     0   Event.Date              70563 non-null  datetime64[ns]
     1   Location                70563 non-null  object        
     2   Country                 70563 non-null  object        
     3   Injury.Severity         70563 non-null  object        
     4   Aircraft.damage         70563 non-null  object        
     5   Aircraft.Category       70563 non-null  object        
     6   Make                    70563 non-null  object        
     7   Model                   70563 non-null  object        
     8   Total.Fatal.Injuries    70563 non-null  float64       
     9   Total.Serious.Injuries  70563 non-null  float64       
     10  Total.Minor.Injuries    70563 non-null  float64       
     11  Number.of.Engines       70563 non-null  float64       
     12  Engine.Type             70563 non-null  object        
     13  Total.Uninjured         70563 non-null  float64       
     14  Weather.Condition       70563 non-null  object        
     15  Broad.phase.of.flight   70563 non-null  object        
     16  Year                    70563 non-null  int32         
     17  Month                   70563 non-null  int32         
     18  Day                     70563 non-null  int32         
     19  Total.Injuries          70563 non-null  float64       
     20  Injury.Severity.Index   70563 non-null  float64       
    dtypes: datetime64[ns](1), float64(7), int32(3), object(10)
    memory usage: 11.0+ MB
:::
:::

::: {#1de1c888 .cell .markdown}
### Standardizing Identifiers in Airplane Data

In the code below, we are standardizing the identifiers for \'Make\' and
\'Model\' columns in the airplane data CSV file using fuzzy matching.
This process aims to ensure consistency and accuracy in the
identification of aircraft makes and models.

#### Steps: {#steps}

1.  Load the airplane data CSV file.
2.  Define functions to standardize \'Make\' and \'Model\' identifiers
    using fuzzy matching.
3.  Generate standard dictionaries for \'Make\' and \'Model\'.
4.  Update the \'Make\' and \'Model\' columns with the standardized
    identifiers.

Let\'s proceed with the standardization process:
:::

::: {#a91ae64b .cell .code execution_count="43"}
``` python
import pandas as pd
from fuzzywuzzy import process

# Load the airplane data CSV file
df_airplane = pd.read_csv('Data/airplane_data.csv')  # Replace 'path_to_airplane_data.csv' with the actual file path

# Function to standardize identifiers using fuzzy matching for 'Make'
def standardize_make(df):
    standard_dict_make = {}

    for make in df['Make'].unique():
        matches = process.extractOne(make, df['Make'].unique())
        standard_dict_make[make] = matches[0]

    return standard_dict_make

# Function to standardize identifiers using fuzzy matching for 'Model'
def standardize_model(df):
    standard_dict_model = {}

    for model in df['Model'].unique():
        matches = process.extractOne(model, df['Model'].unique())
        standard_dict_model[model] = matches[0]

    return standard_dict_model

# Generate the standard dictionaries for 'Make' and 'Model'
standard_dict_make = standardize_make(df_airplane)
standard_dict_model = standardize_model(df_airplane)

# Update the 'Make' and 'Model' columns with standardized identifiers
df_airplane['Make'] = df_airplane['Make'].map(standard_dict_make)
df_airplane['Model'] = df_airplane['Model'].map(standard_dict_model)
```
:::

::: {#94a59c08 .cell .markdown}
## PLOTTING
:::

::: {#db4b7aef .cell .markdown}
#### 1. Trend of Total Accidents Over Time {#1-trend-of-total-accidents-over-time}

#### Objective: To analyze the trend of accidents over the years by plotting a Line chart

#### showing the number of accidents per year. {#showing-the-number-of-accidents-per-year}
:::

::: {#16d27969 .cell .code execution_count="44"}
``` python
# Group the data by 'Year' and count the number of occurrences (accidents) for each year
df_yearly = df.groupby('Year').size()

# Create a new figure with the specified size
plt.figure(figsize=(10, 6))

# Plot the data as a line chart
# 'marker' sets the marker style for the data points
plt.plot(df_yearly.index, df_yearly.values, marker='o')

# Set the title of the plot
plt.title('Trend of Total Accidents Over Time')

# Set the x-axis label
plt.xlabel('Year')

# Set the y-axis label
plt.ylabel('Number of Accidents')

# Add a grid to the plot for better readability
plt.grid(True)

# Display the plot
plt.show()
```

::: {.output .display_data}
![](vertopal_5ceed5ccf1f543d8af87389819b9ee48/d4c920b3098379dec7d1b06f4c6657b6c1ea6c56.png)
:::
:::

::: {#e8fa0e39 .cell .markdown}
### 2. Top Aircraft Makes by least Number of Accidents {#2-top-aircraft-makes-by-least-number-of-accidents}

**Objective:** Identify the top 10 aircraft makes (companies) by the
number of accidents.
:::

::: {#bd2f7c6f .cell .code execution_count="45"}
``` python
# Count each unique value in the 'Make' column and select the top 10 most frequent ones
df_make = df['Make'].value_counts().tail(10)

# Create a new figure of specific size
plt.figure(figsize=(10, 6))

# Plot the data as a bar chart
# 'kind' specifies the type of plot (bar plot)
# 'color' sets the color of the bars
df_make.plot(kind='bar', color='skyblue')

# Set the title of the plot
plt.title('Top 10 Aircraft Makes (Companies) by Least Number of Accidents')

# Set the x-axis label
plt.xlabel('Aircraft Make')

# Set the y-axis label
plt.ylabel('Number of Accidents')

# Rotate the x-axis labels for better readability (55 degrees)
plt.xticks(rotation=55)

# Display the plot
plt.show()
```

::: {.output .display_data}
![](vertopal_5ceed5ccf1f543d8af87389819b9ee48/df0e7512433a97739a8aa691854e0af0333709dc.png)
:::
:::

::: {#b6c4ecab .cell .markdown}
### 3. Number of Accidents Under Different Weather Conditions {#3-number-of-accidents-under-different-weather-conditions}

**Objective:** Analyze the number of accidents under different weather
conditions.
:::

::: {#4a3244b4 .cell .code execution_count="46"}
``` python
# Count  each unique value in the 'Weather.Condition' column
df_weather = df['Weather.Condition'].value_counts()

# figure with the specified size
plt.figure(figsize=(10, 6))

# Plot the data as a bar chart
# 'kind' specifies the type of plot (bar plot)
# 'color' sets the color of the bars
df_weather.plot(kind='bar', color='lightcoral')

# Set the title of the plot
plt.title('Number of Accidents Under Different Weather Conditions')

# Set the x-axis label
plt.xlabel('Weather Condition')

# Set the y-axis label
plt.ylabel('Number of Accidents')

# Rotate the x-axis labels for better readability (45 degrees)
plt.xticks(rotation=45)

# Display the plot
plt.show()
```

::: {.output .display_data}
![](vertopal_5ceed5ccf1f543d8af87389819b9ee48/2c9dd9fcdd8dc4c168cdcce8069ac8475d5d5fd6.png)
:::
:::

::: {#dd5880f4 .cell .markdown}
**Objective:** To simplify plotting.
:::

::: {#98e52a07 .cell .markdown}
### 4. Accidents by Injury Severity Over Time {#4-accidents-by-injury-severity-over-time}

**Objective:** Analyze the number of accidents by injury severity over
time using a stacked bar chart.
:::

::: {#01f085ce .cell .code execution_count="47"}
``` python
# Group the data by 'Year' and 'Injury.Severity', then count the occurrences
# This will give us a DataFrame with counts of each severity type for each year
df_severity = df.groupby(['Year', 'Injury.Severity']).size().unstack()

# Plot data as a stacked bar chart
# 'kind' specifies the type of plot (bar plot)
# 'stacked=True' makes the bars stacked on top of each other for each year
# 'figsize' sets the size of the figure
# 'colormap' sets the color palette for the bars
df_severity.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis')

# Set the title of the plot
plt.title('Accidents by Injury Severity Over Time')

# Set the x-axis label
plt.xlabel('Year')

# Set the y-axis label
plt.ylabel('Number of Accidents')

# Add a legend with the title 'Injury Severity'
plt.legend(title='Injury Severity')

# Display the plot
plt.show()
```

::: {.output .display_data}
![](vertopal_5ceed5ccf1f543d8af87389819b9ee48/3b2562777456791474b756ca64429be61610e8a4.png)
:::
:::

::: {#70f0205f .cell .markdown}
### 5. Heatmap of Accidents by Country and Year {#5-heatmap-of-accidents-by-country-and-year}

**Objective:** Visualize the number of accidents by country and year
using a heatmap.
:::

::: {#9e26aeb9 .cell .code execution_count="48"}
``` python
df_country_year = df.groupby(['Country', 'Year']).size().unstack(fill_value=0)

# Create a new figure with the specified size
plt.figure(figsize=(14, 10))

# Create a heatmap with the data
# 'cmap' specifies the color palette for the heatmap
# 'linewidths' sets the width of the lines that will divide each cell
sns.heatmap(df_country_year, cmap='coolwarm', linewidths=.5)

# Set the title of the plot
plt.title('Heatmap of Accidents by Country and Year')

# Set the x-axis label
plt.xlabel('Year')

# Set the y-axis label
plt.ylabel('Country')

# Display the plot
plt.show()
```

::: {.output .display_data}
![](vertopal_5ceed5ccf1f543d8af87389819b9ee48/d8dfe720b6ac5977658ee04812a5f40c005e9b61.png)
:::
:::

::: {#1091f796 .cell .markdown}
### 6. Fatal vs Non-Fatal Accidents by Aircraft Category {#6-fatal-vs-non-fatal-accidents-by-aircraft-category}

**Objective:** Compare the number of fatal and non-fatal accidents by
aircraft category.
:::

::: {#3e49b9a2 .cell .code execution_count="49"}
``` python
# Group the data by 'Aircraft.Category' and 'Injury.Severity', getting count
df_fatal_nonfatal = df.groupby(['Aircraft.Category', 'Injury.Severity']).size().unstack()

# Plot is a bar chart
# 'kind' specifies the type of plot (bar plot)
# 'figsize' sets the size of the figure
# 'colormap' sets the color palette for the bars
df_fatal_nonfatal.plot(kind='bar', figsize=(12, 8), colormap='Set1')

# Set the title of the plot
plt.title('Fatal vs Non-Fatal Accidents by Aircraft Category')

# Set the x-axis label
plt.xlabel('Aircraft Category')

# Set the y-axis label
plt.ylabel('Number of Accidents')

# Add a legend
plt.legend(title='Injury Severity')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.show()
```

::: {.output .display_data}
![](vertopal_5ceed5ccf1f543d8af87389819b9ee48/77ce0d38af2f047a9f7512f073656cf36a7f6333.png)
:::
:::

::: {#e6f03ed4 .cell .markdown}
### 7. Number of Accidents by Number of Engines {#7-number-of-accidents-by-number-of-engines}

**Objective:** Analyze the number of accidents by the number of engines.
:::

::: {#17d0406f .cell .code execution_count="50"}
``` python
# Count  each unique value in the 'Number.of.Engines' column
df_engines = df['Number.of.Engines'].value_counts()

# Create a new figure of specified size
plt.figure(figsize=(10, 6))

# Plot the data as a bar chart
# 'kind' specifies the type of plot (bar plot)
# 'color' sets the color of the bars
df_engines.plot(kind='bar', color='dodgerblue')

# Set title of the plot
plt.title('Number of Accidents by Number of Engines')

# Set the x-axis label
plt.xlabel('Number of Engines')

# Set the y-axis label
plt.ylabel('Number of Accidents')

# Rotate the x-axis labels for better readability (0 degrees means no rotation)
plt.xticks(rotation=0)

# Display the plot
plt.show()
```

::: {.output .display_data}
![](vertopal_5ceed5ccf1f543d8af87389819b9ee48/c6025f802b99d45ab164aaf3912d9bba3cbc89ac.png)
:::
:::

::: {#e4fa0b80 .cell .markdown}
### 8. Monthly Distribution of Accidents {#8-monthly-distribution-of-accidents}

**Objective:** Analyze the monthly distribution of accidents using a
line chart.
:::

::: {#1578e975 .cell .code execution_count="51"}
``` python
# Group the data by 'Month' and count the number accidents for each month
df_monthly = df.groupby('Month').size()
plt.figure(figsize=(10, 6))

# Plot the data as a line chart
# 'marker' sets the marker style for the data points
# 'color' sets the line color
plt.plot(df_monthly.index, df_monthly.values, marker='o', color='darkorange')

# Set the title of the plot
plt.title('Monthly Distribution of Accidents')

# Set the x-axis label
plt.xlabel('Month')

# Set the y-axis label
plt.ylabel('Number of Accidents')

# Add a grid to the plot for better readability
plt.grid(True)

# Display the plot
plt.show()
```

::: {.output .display_data}
![](vertopal_5ceed5ccf1f543d8af87389819b9ee48/d49f0d40c6b59da755aabddff8a25fd6ccd915a7.png)
:::
:::

::: {#4c8301d6 .cell .markdown}
### 9. Aircraft Make over Total Fatalities {#9-aircraft-make-over-total-fatalities}

**Objective:** displaying the top 10 aircraft makes by total fatalities.
Here\'s a step-by-step breakdown :
:::

::: {#1e09cda0-915e-4c75-9369-ceab687c60c6 .cell .code execution_count="52"}
``` python
import matplotlib.pyplot as plt

# Group by 'Make' and sum the 'Total.Fatal.Injuries' for each make
make_to_fatalities = df.groupby('Make')['Total.Fatal.Injuries'].sum().reset_index()

# Sort the results by the number of fatalities in descending order
make_to_fatalities = make_to_fatalities.sort_values(by='Total.Fatal.Injuries', ascending=False).head(10)  # Top 10 makes

# Plot the data
plt.figure(figsize=(12, 8))
plt.bar(make_to_fatalities['Make'], make_to_fatalities['Total.Fatal.Injuries'], color='salmon')
plt.title('Top 10 Aircraft Makes by Total Fatalities')
plt.xlabel('Aircraft Make')
plt.ylabel('Total Fatalities')
plt.xticks(rotation=45)
plt.show()
```

::: {.output .display_data}
![](vertopal_5ceed5ccf1f543d8af87389819b9ee48/d56472748a664e91239ca7ced56c102f6e1e54a6.png)
:::
:::

::: {#4007ffe6 .cell .markdown}
### 10. Analysis of Aircraft Damage Severity by Make {#10-analysis-of-aircraft-damage-severity-by-make}

This analysis aims to map the aircraft damage severity and identify the
top 10 aircraft makes with the highest damage severity index. We then
use a predefined mapping for damage severity, calculate the damage
severity index for each make, and visualize the top 10 makes.
:::

::: {#d36900ba .cell .code execution_count="53"}
``` python
# Define a mapping for damage severity
damage_severity_mapping = {
    'Destroyed': 3,
    'Substantial': 2,
    'Minor': 1,
    'None': 0
}

# Map the 'Aircraft.damage' column to the damage severity index
df['Damage.Severity.Index'] = df['Aircraft.damage'].map(damage_severity_mapping)

# Group by 'Make' and sum the 'Damage.Severity.Index' for each make
make_to_damage_severity = df.groupby('Make')['Damage.Severity.Index'].sum().reset_index()

# Sort the results by the damage severity index in descending order and select the top 10 makes
top_damage_severity_makes = make_to_damage_severity.sort_values(by='Damage.Severity.Index', ascending=False).head(10)

# Plot the data
plt.figure(figsize=(12, 8))
plt.bar(top_damage_severity_makes['Make'], top_damage_severity_makes['Damage.Severity.Index'], color='salmon')
plt.title('Top 10 Aircraft Makes by Damage Severity Index')
plt.xlabel('Aircraft Make')
plt.ylabel('Damage Severity Index')
plt.xticks(rotation=45)
plt.show()
```

::: {.output .display_data}
![](vertopal_5ceed5ccf1f543d8af87389819b9ee48/ab93b6aa009eee74efb1157b4b5689267b4d0a56.png)
:::
:::

::: {#5dd0e7be .cell .code execution_count="54"}
``` python
df.to_excel('Analysed_data.xlsx', index=False)
```
:::

::: {#00fce6f3 .cell .code execution_count="55"}
``` python
df.columns
```

::: {.output .execute_result execution_count="55"}
    Index(['Event.Date', 'Location', 'Country', 'Injury.Severity',
           'Aircraft.damage', 'Aircraft.Category', 'Make', 'Model',
           'Total.Fatal.Injuries', 'Total.Serious.Injuries',
           'Total.Minor.Injuries', 'Number.of.Engines', 'Engine.Type',
           'Total.Uninjured', 'Weather.Condition', 'Broad.phase.of.flight', 'Year',
           'Month', 'Day', 'Total.Injuries', 'Injury.Severity.Index',
           'Damage.Severity.Index'],
          dtype='object')
:::
:::

::: {#022a50cf .cell .code}
``` python
df['Make'].unique
```

::: {.output .execute_result execution_count="136"}
    <bound method Series.unique of 0                             Piper
    1                            Cessna
    2                            Boeing
    3                              Fuji
    4                            Cessna
                        ...            
    78269     Grumman American Avn Corp
    78270                   Air Tractor
    78271                         Piper
    78273    American Champion Aircraft
    78275                         Piper
    Name: Make, Length: 76980, dtype: object>
:::
:::

::: {#1bce10e7 .cell .code}
``` python
```
:::
