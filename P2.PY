import pandas as pd
import math
import csv
import datetime
from copy import copy
import time
Data_length = 0

# if __name__ == '__main__':
#     df = pd.read_csv("./data/test.txt", header=None, delim_whitespace=True)

#numpy_array = df.to_numpy()
# https://www.folkstalk.com/2022/10/python-read-csv-into-array-with-code-examples.html


def ParseData(path):

    # https://www.geeksforgeeks.org/pandas-dataframe-to_numpy-convert-dataframe-to-numpy-array/
    df = pd.read_csv(path, header=None, delim_whitespace=True)
    numpy_array = df.to_numpy()
    # print(type(numpy_array))
    return numpy_array
    # return data


# Pseudo code from matlab code(Project_2_Briefing slide)

def leave_one_out_cross_validation(data, current_set_of_features, feature_to_add):

    #temp = copy(current_set_of_features)

    # current_set_of_features.append(feature_to_add)
    number_correctly_classfied = 0

    for i in range(Data_length):
        # The class we want to test
        lable_object_to_classify = data[i][0]
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        for k in range(Data_length):
            if k != i:
                distance = 0
                # Euclidean Distance Formula, How to convert distance = sqrt(sum((object_to_classify - data(k,2:end)).^2))(slide's matlab code) I asked my friend and he told me basic logic and done by myself
                for feature in current_set_of_features:
                    dif = data[i][feature] - data[k][feature]
                    distance += pow(dif, 2)

                distance = math.sqrt(distance)
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location][0]
                #distance = 0
        # K-fold cross validation
        if lable_object_to_classify == nearest_neighbor_label:
            number_correctly_classfied += 1
    accuracy = number_correctly_classfied / Data_length
    print(
        f'    Feature(s) {current_set_of_features}  Accuracy is {round(accuracy * 100, 2)}%')
    current_set_of_features.clear()

    return accuracy

# Pseudo code from matlab(Project_2_Briefing slide) Basically the same as forward search, the only difference is the bottom up format,We need to store all features first


def backward_elimination(data):
    print('Backward Search ')

    Fullset_accuracy = 0
    current_set_of_features = []

    # add all, consult my friend he told me basic idea, and done by myself
    for i in range(1, len(data[0])):
        current_set_of_features.append(i)
   # print(current_set_of_features)
    # create temp to pass  to leave_one_out_cross_validation
    temp2 = copy(current_set_of_features)
    # print(temp2)
    temp = round(leave_one_out_cross_validation(data, temp2, 0) * 100, 2)

    print('Feature', current_set_of_features,
          ', Accuracy is', temp, '%. Beginning search')
    for i in range(1, len(data[0])-1):  # basic logic same as feature_search
        feature_to_remove_at_this_level = []
        best_so_far_accuracy = 0
        for j in range(1, len(data[0])):
            if j in current_set_of_features:
                # Create temp sets pool for leave_one_out_cross_validation
                temp1 = copy(current_set_of_features)
                temp1.remove(j)

                accuracy = (leave_one_out_cross_validation(data, temp1, j))
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove_at_this_level = j

        # print(3333)
        # print(feature_to_remove_at_this_level)
        current_set_of_features.remove(feature_to_remove_at_this_level)

        print(
            f'Feature {current_set_of_features} was best, accuracy: {round(best_so_far_accuracy * 100, 2)}%')

        if best_so_far_accuracy > Fullset_accuracy:
            Current_best_set = current_set_of_features.copy()
            Fullset_accuracy = best_so_far_accuracy

    Fullset_accuracy = round(Fullset_accuracy * 100, 3)

    print("Finished search! The best feature subset is", Current_best_set,
          ",which has an accuracy of", Fullset_accuracy, "%")

# Pseudo code from matlab(Project_2_Briefing slide)


def feature_search(data):

    print('Forward Search ')

    Fullset_accuracy = 0
    current_set_of_features = []

    for i in range(1, len(data[0])):  # level of search tree

        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0
        for j in range(1, len(data[0])):  # Considering adding the J level
            if j not in current_set_of_features:
                # Create temp sets pool for leave_one_out_cross_validation
                temp = copy(current_set_of_features)
                temp.append(j)
                accuracy = (leave_one_out_cross_validation(
                    data, temp, j))

                if accuracy > best_so_far_accuracy:  # logic same as project2 slide. Update Current best accuracy and feature_to_add_at_this_level
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = j

        current_set_of_features.append(
            feature_to_add_at_this_level)  # append to list

        print(
            f'Feature set {current_set_of_features} was best, accuracy: {round(best_so_far_accuracy * 100, 2)}%')

        if best_so_far_accuracy > Fullset_accuracy:
            Current_best_set = current_set_of_features.copy()  # store current best set
            Fullset_accuracy = best_so_far_accuracy

    
    Fullset_accuracy = round(Fullset_accuracy * 100, 3)

    print("Finished search! The best feature subset is", Current_best_set,
          ",which has an accuracy of", Fullset_accuracy, "%")


if __name__ == '__main__':
    print("Welcome to Bertie Woosters Feature Selection Algorithm.")
    data = []
    path = str(input('Type in the name of the file to test :\n'))
    data = ParseData(path)  # load data using pandas
    c = int(input(
        "Type the number of the algorithm you want to run. \n 1. Forward Search \n2. Backward Search\n"))
    #start_t = datetime.datetime.now()
    start_time = time.time()
    Data_length = len(data)
    Feature_size = len(data[0])
    print("Feature size (not including the class attribute):")
    print(Feature_size-1)
    print("Instances:")
    print(Data_length)
    if c == 1:
        feature_search(data)
    if c == 2:
        backward_elimination(data)
   # Total Time
    end_time = time.time()

    print("Total Time: {:.2f}sec".format(end_time - start_time))

