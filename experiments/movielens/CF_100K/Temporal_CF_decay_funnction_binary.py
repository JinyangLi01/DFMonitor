# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:20:54 2020

@author: Tisana
"""


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import sys


# prediction function

def faster_rating_prediction(k, user_similarity_matrix, time, alpha, user_timestamp_array, user_rating_array, num_user, num_movie):
    # avg rating of all user matrix
    avg_rating_user_matrix = np.mean(user_rating_array, axis=1)
    avg_rating_user_matrix = avg_rating_user_matrix[:, np.newaxis]
    avg_rating_user_matrix = np.repeat(avg_rating_user_matrix, num_movie, axis=1)

    predicted_rating_array = []
    for target_user_index in range(0, num_user):

        # avg rating of target user
        avg_rating_of_target_user = avg_rating_user_matrix[target_user_index, :]
        # print("target_user_index: {}".format(target_user_index))
        # print(type(user_similarity_matrix))
        # print(user_similarity_matrix)

        # find k similar user
        lst = pd.Series(list(user_similarity_matrix[target_user_index, :]))

        i = lst.nlargest(k + 1)
        similar_user_index_list = i.index.values.tolist()
        similar_user_index_list = similar_user_index_list[1:]  # exclude yourself

        # avg rating of similar user
        avg_rating_of_similar_user = avg_rating_user_matrix[similar_user_index_list, :]
        rating_of_similar_user = user_rating_array[similar_user_index_list, :]
        diff_of_similar_user = rating_of_similar_user - avg_rating_of_similar_user

        time_diff = weighted_time(target_user_index, similar_user_index_list, alpha, user_timestamp_array)

        # check for time
        if time == True:
            diff_of_similar_user = diff_of_similar_user * time_diff

        # second term
        similarity_to_target_user = user_similarity_matrix[target_user_index, similar_user_index_list]
        similarity_to_target_user = similarity_to_target_user[:, np.newaxis]
        numerator = sum(diff_of_similar_user * similarity_to_target_user)

        if time == True:
            denominator = sum(similarity_to_target_user * time_diff)
        else:
            denominator = sum(similarity_to_target_user)

        second_term = numerator / denominator

        # prediction
        predicted_rating_of_target_user = avg_rating_of_target_user + second_term
        predicted_rating_array.append(predicted_rating_of_target_user)

    predicted_rating_array = np.array(predicted_rating_array)
    return predicted_rating_array


########################################################################
########################################################################
########################################################################
# MAE function
def MAE_calculator(predicted_user_rating_array, user_rating_array, data_set):
    # change predict matrix to have only known value
    filter_matrix = np.copy(user_rating_array)
    filter_matrix[filter_matrix > 0] = 1
    predicted_user_rating_array = predicted_user_rating_array * filter_matrix

    num_predict = np.count_nonzero(predicted_user_rating_array)
    MAE = (abs(predicted_user_rating_array - user_rating_array).sum()) / num_predict

    # get binary classification result: 1 if MAE < 0.5, 0 otherwise, since when k = 11, MAE of dynamic user interest is about 1.0
    predicted_user_rating_array_binary = np.zeros(predicted_user_rating_array.shape)
    predicted_user_rating_array_binary[MAE < 1.0] = 1
    predicted_user_rating_array_binary[MAE >= 1.0] = 0

    # for the case without consider dynamic user interest, MAE is about 1.6, so the threshold is 0.8
    filter_matrix_binary = np.zeros(filter_matrix.shape)
    filter_matrix_binary[filter_matrix >= 1.6] = 1
    filter_matrix_binary[filter_matrix < 1.6] = 0

    # print the accuracy
    accuracy = np.count_nonzero(predicted_user_rating_array_binary == filter_matrix) / num_predict
    print("accuracy: ", accuracy)
    # print the binary result for each user
    print("predicted_user_rating_array_binary: ", predicted_user_rating_array_binary)
    # add a column to the original dataset
    # save the dataset
    np.savetxt("predicted_user_rating_array_binary.csv", predicted_user_rating_array_binary, delimiter=",")
    return MAE, predicted_user_rating_array_binary, accuracy


########################################################################
########################################################################
########################################################################

# generate abs time diff matrix
def weighted_time(target_user_index, similar_user_index_list, alpha, user_timestamp_array):
    a = user_timestamp_array[target_user_index, :]
    b = user_timestamp_array[similar_user_index_list, :]
    time_diff_matrix = abs(a - b)

    # standardization
    # from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    time_diff_matrix = scaler.fit_transform(time_diff_matrix)
    lam_matrix = np.exp(-1 * time_diff_matrix * alpha)
    return lam_matrix


# wrap all the above into one function
def temporal_CF_decay_function_binary(data_set, num_user, num_movie, time_window, alpha=1.7, k=11):
    print("Here is the temporal_CF_decay_function_binary")
    ##clean data
    ########################################################################
    ########################################################################
    ########################################################################
    #compute user rating matrix and timestamp matrix
    print("num_user: ", num_user)
    print("num_movie: ", num_movie)

    user_rating_dict={} # size is num_user
    #key is user id : value are rating of all movie
    for user_id in range(1,num_user+1):
        # print("user_id: ", user_id)
        user_rating_dict[user_id]=np.array([0]*num_movie)

    user_timestamp_dict={} # size is num_user
    for user_id in range(1,num_user+1):
        user_timestamp_dict[user_id]=np.array([0]*num_movie)

    user_id_list=data_set['user_id'].unique()
    movie_id_list=data_set['movie_id'].unique()

    #append rating data set to user rating dict
    data_set_list=data_set.values.tolist()
    for each_row in data_set_list:
        # print("each_row: ", each_row)
        user_id=each_row[0]
        mmovie_id=each_row[1]
        rating=each_row[2]
        movie_index=mmovie_id-1
        timestamp=each_row[3]
        #append to dictionary
        user_rating_dict[user_id][movie_index]=rating
        user_timestamp_dict[user_id][movie_index]=timestamp


    user_rating_array=[]
    for user_id, each in user_rating_dict.items():
        if user_id in user_id_list:
            user_rating_array.append(each)
        else:
            user_rating_array.append([0]*num_movie)
    user_rating_array=np.array(user_rating_array) #index by user index (user id -1)

    #convert rating matrix to user-like matrix
    user_like_matrix=[]
    for i in range(0,num_user):
        row_list=[]
        for j in range(0,num_movie):
            rating=user_rating_array[i,j]
            if rating>=3:
                row_list.append(1)
            else:
                row_list.append(0)
        user_like_matrix.append(np.array(row_list))
    user_like_matrix=np.array(user_like_matrix)

    #convert user-like matrix to user-user network
    user_user_network=[]
    for i in range(0,num_user):
        # if i%10==0:
        #     print(i)
        row_list=[]
        for j in range(0,num_user):
            common_prefered_item=user_like_matrix[i,:]*user_like_matrix[j,:]
            row_list.append(common_prefered_item)
        row_list=np.array(row_list).sum(axis=1)
        user_user_network.append(row_list)
    user_user_network=np.array(user_user_network)

    #normalization
    row_mean=np.mean(user_rating_array,axis=1)
    row_mean=row_mean[:,np.newaxis]
    print(len(row_mean), len(row_mean[0]), len(user_rating_array), len(user_rating_array[0]))
    user_rating_array=(user_rating_array-row_mean)

    for each_row in range(0,num_user):
        for each_column in range(0,num_movie):
            if np.isnan(user_rating_array[each_row,each_column])==True:
                user_rating_array[each_row,each_column]=0

    ########################################################################
    ########################################################################
    ########################################################################
    #get timestamp matrix
    user_timestamp_array=[]
    for each in user_timestamp_dict.values():
        user_timestamp_array.append(each)
    user_timestamp_array=np.array(user_timestamp_array)

    ########################################################################
    ########################################################################
    ########################################################################
    #compute user similarity matrix

    # load from file if it exists, otherwise compute it
    if os.path.exists("user_similarity_matrix_"+str(time_window)+".npy"):
        print("load user similarity matrix")
        user_similarity_matrix=np.load("user_similarity_matrix_"+str(time_window)+".npy")
    else:
        # from sklearn.metrics.pairwise import cosine_similarity
        # from scipy.stats import pearsonr
        user_similarity_matrix=[]
        for i in range(0,num_user):
            if i%10==0:
                print(i," out of ",num_user)
            user_1_id=i+1
            row=[]
            for j in range(0,num_user):
                user_2_id=j+1
                similarity=pearsonr(user_rating_array[user_1_id-1],user_rating_array[user_2_id-1])[0]
                #similarity=cosine_similarity([user_rating_array[user_1_id-1]],[user_rating_array[user_2_id-1]])[0][0]
                row.append(similarity)
            user_similarity_matrix.append(np.array(row))
        user_similarity_matrix=np.array(user_similarity_matrix)
        print("save user similarity matrix")
        np.save("user_similarity_matrix_"+str(time_window)+".npy",user_similarity_matrix)



    ########################################################################
    ########################################################################
    ########################################################################
    #find best value of alpha (1.7)
    # best_alpha=1.7
    # print("best_alpha=",best_alpha)

    ########################################################################
    ########################################################################
    ########################################################################
    #compare performance of no time and time
    # alpha=best_alpha
    MAE_time_list=[]
    MAE_no_time_list=[]
    k_list=[]

    # use k = 11, which has the lowest MAE
    # k = 11
    # print("k : ", k)
    time = faster_rating_prediction(k, user_similarity_matrix, True, alpha, user_timestamp_array, user_rating_array, num_user, num_movie)
    no_time = faster_rating_prediction(k, user_similarity_matrix, False, alpha, user_timestamp_array, user_rating_array, num_user, num_movie)

    # print("time: ", time)
    # print("no_time: ", no_time)

    MAE_time, predicted_user_rating_array_binary_time, accuracy_time = MAE_calculator(time, user_rating_array, data_set)
    MAE_no_time, predicted_user_rating_array_binary_no_time, accuracy_no_time = MAE_calculator(no_time, user_rating_array, data_set)


    MAE_time_list.append(MAE_time)
    MAE_no_time_list.append(MAE_no_time)
    k_list.append(k)

    print("k = {}, alpha = {}, MAE_time = {}, MAE_no_time = {}".format(k, alpha, MAE_time, MAE_no_time))
    return (MAE_time, MAE_no_time, predicted_user_rating_array_binary_time, predicted_user_rating_array_binary_no_time,
            accuracy_time, accuracy_no_time)




if __name__ == '__main__':
    sys.path.append("experiments/movielens/CF_100K")
    # 10 time windows
    num_time_windows = 10

    for i in range(0, num_time_windows):
        # Load the data
        # test data is the current time window
        data_set = pd.read_csv('../../../data/movielens/CF_100K/shuffle_assume_timestamp/u' + str(i)
                               + '.data', sep='\s+')

        num_all_user = 943
        num_all_movie = 1682
        alpha = 1.7
        k = 11
        # calculate the temporal CF decay function

        (MAE_time, MAE_no_time, predicted_user_rating_array_binary_time, predicted_user_rating_array_binary_no_time,
         accuracy_time, accuracy_no_time) = temporal_CF_decay_function_binary(
            data_set, num_all_user, num_all_movie, i, alpha, k)

#
#     print("Here is to execute the Temporal_CF_decay_function_binary file")
#
#     ##import data
#     # read from u.data into a dataframe, seperator is any number of black spaces
#     data_set=pd.read_csv("../../../data/movielens/ml-100k/u.data", header=None, sep='\s+').values
#
#
#     ##clean data
#     ########################################################################
#     ########################################################################
#     ########################################################################
#     #compute user rating matrix and timestamp matrix
#     num_user=943 #user id 1 to 943
#     num_movie=1682 #movie id 1 to 1682
#
#     user_rating_dict={}
#     #key is user id : value are rating of all movie
#     for user_id in range(1,num_user+1):
#         user_rating_dict[user_id]=np.array([0]*num_movie)
#
#     user_timestamp_dict={}
#     for user_id in range(1,num_user+1):
#         user_timestamp_dict[user_id]=np.array([0]*num_movie)
#
#     #append rating data set to user rating dict
#     data_set_list=data_set.tolist()
#     for each_row in data_set_list:
#         user_id=each_row[0]
#         mmovie_id=each_row[1]
#         rating=each_row[2]
#         movie_index=mmovie_id-1
#         timestamp=each_row[3]
#         #append to dictionary
#         user_rating_dict[user_id][movie_index]=rating
#         user_timestamp_dict[user_id][movie_index]=timestamp
#
#     user_rating_array=[]
#     for each in user_rating_dict.values():
#         user_rating_array.append(each)
#     user_rating_array=np.array(user_rating_array) #index by user index (user id -1)
#
#     #convert rating matrix to user-like matrix
#     user_like_matrix=[]
#     for i in range(0,num_user):
#         row_list=[]
#         for j in range(0,num_movie):
#             rating=user_rating_array[i,j]
#             if rating>=3:
#                 row_list.append(1)
#             else:
#                 row_list.append(0)
#         user_like_matrix.append(np.array(row_list))
#     user_like_matrix=np.array(user_like_matrix)
#
#     #convert user-like matrix to user-user network
#     user_user_network=[]
#     for i in range(0,num_user):
#         if i%10==0:
#             print(i)
#         row_list=[]
#         for j in range(0,num_user):
#             common_prefered_item=user_like_matrix[i,:]*user_like_matrix[j,:]
#             row_list.append(common_prefered_item)
#         row_list=np.array(row_list).sum(axis=1)
#         user_user_network.append(row_list)
#     user_user_network=np.array(user_user_network)
#     #normalization
#     row_mean=np.mean(user_rating_array,axis=1)
#     row_mean=row_mean[:,np.newaxis]
#     user_rating_array=(user_rating_array-row_mean)*(user_rating_array)/(user_rating_array)
#
#     for each_row in range(0,num_user):
#         for each_column in range(0,num_movie):
#             if np.isnan(user_rating_array[each_row,each_column])==True:
#                 user_rating_array[each_row,each_column]=0
#
#     ########################################################################
#     ########################################################################
#     ########################################################################
#     #get timestamp matrix
#     user_timestamp_array=[]
#     for each in user_timestamp_dict.values():
#         user_timestamp_array.append(each)
#     user_timestamp_array=np.array(user_timestamp_array)
#
#     ########################################################################
#     ########################################################################
#     ########################################################################
#     #compute user similarity matrix
#     # from sklearn.metrics.pairwise import cosine_similarity
#     # from scipy.stats import pearsonr
#     # user_similarity_matrix=[]
#
#     # print("computing user similarity matrix")
#
#     # for i in range(0,num_user):
#     #     if i%10==0:
#     #         print(i," out of ",num_user)
#     #     user_1_id=i+1
#     #     row=[]
#     #     for j in range(0,num_user):
#     #         user_2_id=j+1
#     #         similarity=pearsonr(user_rating_array[user_1_id-1],user_rating_array[user_2_id-1])[0]
#     #         #similarity=cosine_similarity([user_rating_array[user_1_id-1]],[user_rating_array[user_2_id-1]])[0][0]
#     #         row.append(similarity)
#     #     user_similarity_matrix.append(np.array(row))
#     # user_similarity_matrix=np.array(user_similarity_matrix)
#     # print("save user similarity matrix")
#     # np.save("user_similarity_matrix.npy",user_similarity_matrix)
#
#     print("load user similarity matrix")
#     user_similarity_matrix=np.load("user_similarity_matrix.npy")
#
#
#
#     ########################################################################
#     ########################################################################
#     ########################################################################
#     #find best value of alpha (1.7)
#     # k=3
#     # MAE_list=[]
#     # alpha_list=[]
#     # alpha=0
#     # for i in range(0,100):
#     #     predicted_rating=faster_rating_prediction(k,user_similarity_matrix,True,alpha,user_timestamp_array)
#     #     MAE=MAE_calculator(predicted_rating,user_rating_array)
#     #     MAE_list.append(MAE)
#     #     alpha_list.append(alpha)
#     #     print(" MAE : ",MAE)
#     #     alpha+=0.1
#     # plt.title("find best value of alpha")
#     # plt.ylabel("MAE")
#     # plt.xlabel("alpha")
#     # plt.plot(alpha_list,MAE_list)
#     # best_alpha_index=MAE_list.index(min(MAE_list))
#     # best_alpha=alpha_list[best_alpha_index]
#
#     best_alpha=1.7
#     print("best_alpha=",best_alpha)
#
#     ########################################################################
#     ########################################################################
#     ########################################################################
#     #compare performance of no time and time
#     alpha=best_alpha
#     MAE_time_list=[]
#     MAE_no_time_list=[]
#     k_list=[]
#
#     # use k = 11, which has the lowest MAE
#
#     k = 11
#     print("k : ", k)
#     time = faster_rating_prediction(k, user_similarity_matrix, True, alpha, user_timestamp_array)
#     no_time = faster_rating_prediction(k, user_similarity_matrix, False, alpha, user_timestamp_array)
#
#     MAE_time, predicted_user_rating_array_binary_time, accuracy_time = MAE_calculator(time, user_rating_array, data_set)
#     MAE_no_time, predicted_user_rating_array_binary_no_time, accuracy_no_time = MAE_calculator(no_time, user_rating_array, data_set)
#
#
#     MAE_time_list.append(MAE_time)
#     MAE_no_time_list.append(MAE_no_time)
#     k_list.append(k)
#
#
#     print("k = {}, MAE_time = {}, MAE_no_time = {}".format(k, MAE_time, MAE_no_time))
#
#
