# Sequential-Forward-Selection-SFS-algorithm
 - The Sequential Forward Selection (SFS) algorithm to identify important features in the dataset. 
 - The program will take one input: a dataset where the last column is the class variable. 
 - The program will load the dataset, and then use wrapper approach with sequential forward selection strategy to find a set of important features. 
 - The first feature will be selected using information gain or any other filter method of your choice.  
 - The Support Vector Machine(SVM) supervised learning method is used for measuring the performance (accuracy) in the wrapper approach. 
 - The stratified 5-fold cross validation is used for measuring accuracy. 
 - The program will keep adding the features as long as there is some improvement in the classification accuracy, or 75% features have been selected. 
 - The output of the program will be the set of important features on the console. 
 - We used inbuilt libraries for the supervised learning algorithm and the filter method for the first feature selection. However, there is no use of libraries for feature selection and 5-fold cross validation. 
