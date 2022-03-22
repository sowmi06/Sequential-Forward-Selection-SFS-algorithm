# Sequential-Forward-Selection-SFS-algorithm
 - The Sequential Forward Selection (SFS) algorithm to identify important features in the dataset. 
 - The program will take one input: a dataset where the last column is the class variable. 
 - The program will load the dataset, and then use the wrapper approach with a sequential forward selection strategy to find a set of important features. 
 - The first feature will be selected using information gain or any other filter method of your choice.  
 - The Support Vector Machine(SVM) supervised learning method is used for measuring the performance (accuracy) in the wrapper approach. 
 - The stratified 5-fold cross-validation is used for measuring accuracy. 
 - The program will keep adding the features as long as there is some improvement in the classification accuracy, or 75% features have been selected. 
 - The output of the program will be the set of important features on the console. 
 - We used inbuilt libraries for the supervised learning algorithm and the filter method for the first feature selection. However, there is no use of libraries for feature selection and 5-fold cross-validation. 

## Dataset
The dataset used for this code is ["Connectionist Bench (Sonar, Mines vs. Rocks) Data Set"](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
 - The dataset has “ 208” instances.
 - The dataset has “ 60 ” features.
 - The dataset has no missing values.
 - All the attributes are continuous.
 - The classes in the dataset for classification are “0” and “1”.
 - We are using a stratified 5-fold cross-validation.
  
  
## Configuration Instructions
The [Project](https://github.com/sowmi06/Sequential-Forward-Selection-SFS-algorithm.git) requires the following tools and libraries to run the source code.
### System Requirements 
- [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/install/)
    - Python version 3.6.0 – 3.9.0
 
- Python IDE (to run ".py" file)
    - [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows), [Spyder](https://www.psych.mcgill.ca/labs/mogillab/anaconda2/lib/python2.7/site-packages/spyder/doc/installation.html) or [VS code](https://code.visualstudio.com/download)

### Tools and Library Requirements 
    
- [Numpy](https://numpy.org/install/)
  
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)

- [Scikit-learn](https://scikit-learn.org/stable/install.html)  


## Operating Instructions

The following are the steps to replicate the exact results acquired from the project:

- Satisfy all the system and the tool & libraries requirements.
- Clone the [Sequential-Forward-Selection-SFS-algorithm](https://github.com/sowmi06/Sequential-Forward-Selection-SFS-algorithm.git) repository into your local machine. 
- Download the [Connectionist Bench (Sonar, Mines vs. Rocks) Data Set](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
- The [Feature_Selection.py](https://github.com/sowmi06/Sequential-Forward-Selection-SFS-algorithm/blob/main/Feature_selection.py) has the code for the final feature selection set.
