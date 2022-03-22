import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_selection import mutual_info_classif


# function preprocess dataset
def sonar_Dataset_Preprocessing(path):
    # reading the dataset file
    df = pd.read_csv(path, delimiter=',', header=None)
    dataset = np.asarray(df)
    # splitting the dataset to x and y (i.e) features and class labels
    x = dataset[:, :60]
    y = dataset[:, 60:]
    y = y.reshape(-1)
    # printing the dataset information
    print("-------------------------------------------------")
    print("Data Preprocessing:")
    print("Total instance in dataset:", dataset.shape[0])
    print("Number of Features in dataset:", x.shape[1])
    print("Number of classes in dataset:", np.unique(y))
    print("-------------------------------------------------")
    return dataset, y, x


# function fetch values of features and labels with respect to a certain class
def singleClassDataset(x, y, label):
    # getting the indices of the label(y) in a dataset
    index_of_y = np.where(y == label)
    index_of_y_with_label = index_of_y[0]

    splitted_x = []
    splitted_y = []
    # for each index value in y, fetching the features and class labels with singel class values
    # (i.e) 0 and 1 from our dataset
    for index in index_of_y_with_label:
        x_ind = x[index]
        y_ind = y[index]
        # appending the splitted features and label to a list
        splitted_x.append(x_ind)
        splitted_y.append(y_ind)
    # converting the splitted list to numpy array
    splitted_x = np.asarray(splitted_x)
    splitted_y = np.asarray(splitted_y)
    # return the splitted feature and class label with respect to the labels
    return splitted_x, splitted_y


# function for shuffling x and y with respect to index
def shuffled_dataset(x, y):
    # set the index value with instances in dataset
    index = np.arange(x.shape[0])
    # shuffle the index value
    np.random.shuffle(index)
    # set the shuffled index to the original dataset and missing dataset
    x_index = x[index]
    y_index = y[index]
    return x_index, y_index


# function for a stratified 5 fold cross validataion method
def stratified_5_fold_cv(x, y):
    # setting the value of k
    k = 5
    # assigning the dictionary to store splitted values of x and y
    split_dict_x = {}
    split_dict_y = {}
    # getting the unique y values in the dataset
    unique_labels = np.unique(y)
# making a 5 split of x and y with equal propotion for each unique classes
    for label in unique_labels:
        # assign the x value to features
        features = x
        # calls the created singleClassDataset() function to fetch values of features and labels
        # with respect to a specifeid unique class
        splitted_x, splitted_y  = singleClassDataset(features, y, label)
        # shuffling the values
        np.random.shuffle(splitted_x)
        # splitting the x and y values for 5 folds equally
        split_x = np.split(splitted_x, k)
        split_y = np.split(splitted_y, k)
        # storing each split in dictionary with theri classes
        split_dict_x[label] = split_x
        split_dict_y[label] = split_y


# assigning the splitted values to each fold in a k_fold_list
    # creating k_fold_list to store the splits values
    k_fold_list = []
    k_fold_y_list = []
    # the loop runs for 5 folds
    for i in range(k):
        # a list to combine the different class splits to a single k-folds
        combine_classsplit_x= []
        combine_classsplit_y = []
        # loop runs for number of uniques class times(i.e) 2 times for our dataset
        for label in unique_labels:
            # fetching the x and y values from a dictionary with their classLabels
            split_list_x = split_dict_x[label]
            split_list_y = split_dict_y[label]
            # for k loops assigning the values from stored list
            split_x = split_list_x[i]
            split_y = split_list_y[i]
            # storing the splits for each fold in a list
            combine_classsplit_x.append(split_x)
            combine_classsplit_y.append(split_y)
        # combining the x and y values of different class to a k-folds
        # that is we have a total 205 samples and dataset with
        # label- 1 has 110 instances which is 53.36 % and with
        # label- 0 has  95 instances which is 46.63 %
        # So we have 22 instance for class-1 and  19 instances for class-0
        # in each fold totally 41 instances in each fold
        fold_x = np.concatenate((combine_classsplit_x[0], combine_classsplit_x[1]))
        fold_y = np.concatenate((combine_classsplit_y[0], combine_classsplit_y[1]))

        # calling the created shuffled_dataset() function to shuffle x and y with respect to their index after splitting
        shuffled_x, shuffled_y = shuffled_dataset(fold_x, fold_y)
        # appending each folds to a list
        k_fold_list.append(shuffled_x)
        k_fold_y_list.append(shuffled_y)

    # 5 fold values for x values
    fold_0 = k_fold_list[0]
    fold_1 = k_fold_list[1]
    fold_2 = k_fold_list[2]
    fold_3 = k_fold_list[3]
    fold_4 = k_fold_list[4]
    # 5 fold for the y values
    y_fold0 = k_fold_y_list[0]
    y_fold1 = k_fold_y_list[1]
    y_fold2 = k_fold_y_list[2]
    y_fold3 = k_fold_y_list[3]
    y_fold4 = k_fold_y_list[4]


# implementing the cross validation method
    # for each k we take one fold as a test and (k-1) folds as a training
    for i in range(k):
        # assigning the test fold
        test_fold = k_fold_list[i]
        y_test = k_fold_y_list[i]
        # assigning the train folds for k iterations
        # if k = 0 assign fold_1, fold_2, fold_3, fold_4 as train fold
        if i == 0:
            train_tuple = (fold_1, fold_2, fold_3, fold_4)
            train_fold = np.vstack(train_tuple)
            # labels for train fold
            y_train_tuple = (y_fold1, y_fold2, y_fold3, y_fold4)
            y_train = np.hstack(y_train_tuple)

        # if k = 1 assign fold_0, fold_2, fold_3, fold_4 as train fold
        elif i == 1:
            train_tuple = (fold_0, fold_2, fold_3, fold_4)
            train_fold = np.vstack(train_tuple)
            # labels for train fold
            y_train_tuple = (y_fold0, y_fold2, y_fold3, y_fold4)
            y_train = np.hstack(y_train_tuple)

        # if k = 2 assign fold_0, fold_1, fold_3, fold_4 as train fold
        elif i == 2:
            train_tuple = (fold_0, fold_1, fold_3, fold_4)
            train_fold = np.vstack(train_tuple)
            # labels for train fold
            y_train_tuple = (y_fold0, y_fold1, y_fold3, y_fold4)
            y_train = np.hstack(y_train_tuple)

        # if k = 3 assign fold_0, fold_1, fold_2, fold_4 as train fold
        elif i == 3:
            train_tuple = (fold_0, fold_1, fold_2, fold_4)
            train_fold = np.vstack(train_tuple)
            # labels for train fold
            y_train_tuple = (y_fold0, y_fold1, y_fold2, y_fold4)
            y_train = np.hstack(y_train_tuple)

        # if k = 4 assign fold_0, fold_1, fold_2, fold_3 as train fold
        elif i == 4:
            train_tuple = (fold_0, fold_1, fold_2, fold_3)
            train_fold = np.vstack(train_tuple)
            # labels for train fold
            y_train_tuple = (y_fold0, y_fold1, y_fold2, y_fold3)
            y_train = np.hstack(y_train_tuple)

# assigning a classifier from sk learns
        classifier = svm.SVC(kernel='rbf', C=0.9)
        # classifier = DecisionTreeClassifier()
        # training the classifier
        classifier.fit(train_fold, y_train)
        # testing the classifier
        y_predicted = classifier.predict(test_fold)


# calculating overall testing accuracy for stratifies 5 fold cross validation
        acc =[]
        correct_value = 0
        # calculating the total_instances for accuracy calculation
        total_instances = test_fold.shape[0]
        for i in range(total_instances):
            # for every predicted values
            y_pred = y_predicted[i]
            y_actual = y_test[i]
            # checking if the predicted value is equal to the actual label value
            if y_pred == y_actual:
                # counting the total correct values
                correct_value = correct_value + 1
        # calculating the accuracy
        accuracy = correct_value / total_instances
        # appending the accuracy values for k times to find overall accuracy
        acc.append(accuracy)
    # calculating overall accuracy
    average_accuracy = np.mean(acc)

    return average_accuracy


# function to get the 1 st important feature using filter approach
def filter_approach(x, y):
    # calculating the mutual information between the each feature(x) and class(y)
    mutual_information = mutual_info_classif(x, y)
    # finding the maximum index value of the feature
    index = np.argsort(mutual_information)
    index_0 = index[0]
    # transposing the dataset to access the index of feature-wise
    x_T = np.transpose(x)
    # assigning the 1st important feature to the feature subset
    feature_1 = x_T[index_0]
    feature_subset = np.transpose(feature_1)
    return feature_subset, index_0


# function for the objective function using wrapper approach
def objectiveFunction(features, important_feature_subset, index_0,x,y):
    # defining the objective function
    acc = []
    # index list of the selected features (index_0) is the index fetched from the filter approach
    index_list = [index_0]
    # transposing the dataset to access the index of feature-wise
    x_Trans = np.transpose(x)
    for i in range((len(x_Trans) - (len(index_list)))):
        # fetching one feature from all the feature
        a = features[i]
        # joining the fetched feature to the selected important dataset
        tuple = (important_feature_subset, a)
        subset = np.vstack(tuple)
        # transposing the feature_subset to access the index of feature-wise
        subset = np.transpose(subset)
        # using created stratified_5_fold_cv() function to measure accuracy
        # which is the objective function
        average_accuracy = stratified_5_fold_cv(subset, y)
        # appending the feature abd its respective accuracy
        index_list.append(i)
        acc.append(average_accuracy)
    acc = np.asarray(acc)
    # finding the index of the feature with maximum accuracy
    max_idx = np.argmax(acc)
    index_list.append(max_idx)
    # finding teh feature with maximum accuracy
    accuracy = np.max(acc)
    return max_idx, accuracy


# function for a sequential forward selection feature selection method
def sfs(x, y):
    print("-------------------------------------------------")
    print("Sequentail Forward Selection Method:")

    # fecthing the 1st feature from the created filter_approach() function using filter approach
    feature_subset, index_0 = filter_approach(x, y)
    # transposing the feature_subset and dataset to access the index of feature-wise
    feature_subset = np.transpose(feature_subset)
    x_Trans = np.transpose(x)
    # index list of the selected features (index_0) is the index fetched from the filter approach
    index_list = [index_0]
    # calculating the 75% of features from 60 (total features) which is 45 features
    feature_of_75percent = int(60 * (75/100))
    current_accuracy = 0.0

    # looping for feature selection til 75% of features
    for i in range(1, feature_of_75percent):
        data = x_Trans
        important_feature_subset = feature_subset
        # for each index of calcualted important features
        for index in range(len(index_list)):
            # we delete that particular important feature for the whole features list to find a new important feature
            features = np.delete(data, (index_list[index]), axis=0)
        # taking the previous_accuracy value from last tested important feature
        previous_accuracy = current_accuracy
        # calling the created objectiveFunction_wrapperApproach() function
        # to get the index of important features and their accuracy
        index_important_feature, current_accuracy = objectiveFunction(features, important_feature_subset, index_0, x, y)

        # comparing the previous and current accuracy values and
        # if the previous accuracy is less than current accuracy then continue
        if (previous_accuracy <= current_accuracy) :
             # take the index of newly found important feature
            imp_feature = features[index_important_feature]
             # append it to the feature subset
            tuple = (important_feature_subset, imp_feature)
            feature_subset = np.vstack(tuple)
            index_list.append(index_important_feature)
            print("The objective function accuracy:", current_accuracy)

        # break if the previous accuracy is less than current accuracy
        else:
            break

    print("-------------------------------------------------")
    feature_subset = np.transpose(feature_subset)
    return feature_subset


# function to test the feature subset with the ML classifier
def ML_classifier(feature_subset, y):
    # testing the feature subset using created stratified_5_foldcv() function approach
    testing_accuracy = stratified_5_fold_cv(feature_subset, y)
    print("-------------------------------------------------")
    print("Overall Testing accuracy using the ML Classifier:", testing_accuracy)
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("Feature Subset:")
    print("-------------------------------------------------")
    df = pd.DataFrame(feature_subset)
    # displaying the feature subset
    print(df.head(n=20))
    print("-------------------------------------------------")
    print("Shape of the feature subset:", feature_subset.shape)


def main():
    # assigning dataset path to open the "csv dataset file"
    dataset_path = "./FeatureSelection_dataset.csv"

    # calling the dataset preprocessing function
    dataset, y, x = sonar_Dataset_Preprocessing(dataset_path)

    # calling the sfs(sequential forward selection) function to do the feature selection
    feature_subset = sfs(x, y)

    # testing the classifier using stratified cross validation method to test the feature subset
    ML_classifier(feature_subset, y)


if __name__ == "__main__":
    main()
