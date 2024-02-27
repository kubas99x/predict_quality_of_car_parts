# Predicting quality status for metal part just after casting process based on machine parameters

The purpose of the project is to detect NOK parts immediately after the casting process, before the parts go through machining and other processes.

## Table of Contents

1. [About](#about)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## About

Before a metal car part can be manufactured it has to go a long way, starting with casting and ending with packaging for shipment. This involves costs as well as time.  In most cases, it is only at the end of the process that we find out whether the part meets the quality requirements. But what if we could determine the quality of a part after the casting process itself, based on the parameters with which the part was cast?  

This problem is solved by our project. Which, based on casting parameters such as e.g. final pressure, casting temperature, overflow of cooling circuits, makes a prediction as to whether the cast part will be OK or NOK. 

As a result, we are able to reject a part that does not meet the quality requirements after the first production process, saving both time and money on part processing.

This project uses supervised learning techniques to perform binary classification based on historical data. The classification includes two classes: 0 - OK and 1 - NOK.

## Features

List the key features of your project. You can use bullet points for better readability.

- Feature 1
- Feature 2
- Feature 3

## Installation

Provide instructions on how to install your project. Include any prerequisites and steps necessary for setup.

```bash
# Example installation command
npm install my-project