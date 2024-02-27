# Predicting quality status for metal part just after casting process based on machine parameters

The purpose of the project is to detect NOK parts immediately after the casting process, before the parts go through machining and other processes.

## Table of Contents

1. [About](#about)
2. [Features](#features)
3. [File structure](#file-structure)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## About

Before a metal car part can be manufactured it has to go a long way, starting with casting and ending with packaging for shipment. This involves costs as well as time.  In most cases, it is only at the end of the process that we find out whether the part meets the quality requirements. But what if we could determine the quality of a part after the casting process itself, based on the parameters with which the part was cast?  

This problem is solved by our project. Which, based on casting parameters such as e.g. final pressure, casting temperature, overflow of cooling circuits, makes a prediction as to whether the cast part will be OK or NOK. 

As a result, we are able to reject a part that does not meet the quality requirements after the first production process, saving both time and money on part processing.

This project uses supervised learning techniques to perform binary classification based on historical data. The classification includes two classes: 

**0 - OK**

**1 - NOK**

## Features

The total scope of the project included:

• reading the data from database (historical data);

• dataset analysis and data processing;

• choosing proper model to work with this task;

• parametrizing model during teaching process;

• creating pipeline to work with production data;


## File structure

[creating_datasets.py](src/creating_datasets.py) - Main file that executes functions to create, train, test and validate datasets for different machines.

[table_functions.py](table_functions.py) - Functions to drop unused columns, combine final table, define prediction class, normalise and standardise data, over- and under-sample, drop columns with too much correlation.

[analyze_visualisation.py](analyze_visualisation.py) - Functions to analyse the data (pair plots, heat maps etc.)

[ml_functions.py](ml_functions.py) - Functions to create confusion matrix, distribution of probability for specific class

[pipeline.py](pipeline.py) - Program to live-load latest records from database and make predictions

[ml_models directory](ml_models) - Python files for each ML algorithm like xgboost, neural networks, random forest etc.


