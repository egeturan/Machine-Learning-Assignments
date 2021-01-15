import csv
import numpy as np
import math
import time

'''
author:Ege Turan
'''

#variables
labels = []
rowCount = 0
tokenized_corpus = []
split_place = 4460


'''
*********************** METHODS STARTS ***********************
'''

'''
Calculate Spam and Ham occurences for Q2.2 and Q2.3
'''
def calculateSpamCountandTj(labels):
    T_j_spam = np.zeros([d], dtype=int)
    T_j_ham = np.zeros([d], dtype=int)
    Nspam = 0
    Nham = 0    
    total_spam =  0
    total_ham =  0
    for i in range(len(labels)):
        if labels[i] == 1:
            Nspam = Nspam + 1
            T_j_spam = T_j_spam + feature_matrix[i]
            total_spam = total_spam + np.sum(feature_matrix[i], axis=0)
        else: 
            T_j_ham = T_j_ham + feature_matrix[i]
            total_ham = total_ham + np.sum(feature_matrix[i], axis=0)
    
    Nham = len(labels) - Nspam
    return T_j_spam, T_j_ham, Nspam, Nham



'''
Calculate parameters for Q2.2 and Q2.3
'''
def calculate_teta_probabilities(T_j_spam, T_j_ham):
    sum_spam_words = np.sum(T_j_spam, axis=0)
    sum_ham_words = np.sum(T_j_ham, axis=0)
    teta_j_spam = T_j_spam / sum_spam_words
    teta_j_ham = T_j_ham / sum_ham_words
    teta_j_spam_laplace = (T_j_spam + 1.0) / (sum_spam_words + d)
    teta_j_ham_laplace = (T_j_ham + 1.0) / (sum_ham_words + d)
    return teta_j_spam, teta_j_ham, teta_j_spam_laplace, teta_j_ham_laplace, sum_spam_words, sum_ham_words


'''
Naive Bayes prediction Q2.2
'''
def naive_bayes_model(training_data, labels = None):
    result_matrix = np.zeros([len(training_data)], dtype=int)
    pb_spam = Nspam / split_place
    pb_ham = 1 - pb_spam
    log_pb_spam = math.log(pb_spam)
    log_pb_ham = math.log(pb_ham)
    
    
    for i in range(len(training_data)):
        result_spam = 0.0
        result_ham = 0.0
        
        for j in range(d):
            if teta_j_spam[j] == 0:
                if training_data[i][j] != 0.0:
                    result_spam = result_spam + (-math.inf)
            else:
                result_spam = result_spam + (training_data[i][j] * math.log(teta_j_spam[j]))
    
            if teta_j_ham[j] == 0:
                if training_data[i][j] != 0.0:
                    result_ham = result_ham + (-math.inf)
            else:
                result_ham = result_ham + (training_data[i][j] * math.log(teta_j_ham[j]))
            
        result_spam =  result_spam + log_pb_spam
        result_ham = result_ham + log_pb_ham

        if result_spam >= result_ham:
            result_matrix[i] = 1
        else:
            result_matrix[i] = 0
        
        
    num_of_true = 0
    if labels != None:
        print("Accuracy calculation")
        for i in range(len(training_data)):
            if labels[i] == result_matrix[i]:
                num_of_true = num_of_true + 1
     
    return result_matrix, (num_of_true / len(training_data))


def naive_bayes_model_laplace(training_data, labels = None):
    result_matrix = np.zeros([len(training_data)], dtype=int)
    pb_spam = Nspam / split_place
    pb_ham = 1 - pb_spam
    log_pb_spam = math.log(pb_spam)
    log_pb_ham = math.log(pb_ham)
    
    for i in range(len(training_data)):
        result_spam = 0.0
        result_ham = 0.0
        for j in range(d):
            result_spam = result_spam + (training_data[i][j] * math.log(teta_j_spam_laplace[j]))
            result_ham = result_ham + (training_data[i][j] * math.log(teta_j_ham_laplace[j]))
        
        result_spam =  result_spam + log_pb_spam
        result_ham = result_ham + log_pb_ham
        
        if result_spam >= result_ham:
            result_matrix[i] = 1
        else:
            result_matrix[i] = 0
    
    num_of_true = 0
    if labels != None:
        for i in range(len(training_data)):
            if labels[i] == result_matrix[i]:
                num_of_true = num_of_true + 1
                      
    return result_matrix, (num_of_true / len(training_data))
 
    
'''
Training part of the Q3.1
''' 
def train(trainig_set, trainig_labels):
    T_j_spam = 0
    T_j_ham = 0
    Nspam = 0
    
    for i in range(len(trainig_labels)):
        if trainig_labels[i] == 1:
            Nspam = Nspam + 1
            T_j_spam = T_j_spam + trainig_set[i]
        else: 
            T_j_ham = T_j_ham + trainig_set[i]
       
    teta_j_spam_laplace = (T_j_spam + 1.0) / (np.sum(T_j_spam, axis=0) + len(trainig_set))
    teta_j_ham_laplace = (T_j_ham + 1.0) / (np.sum(T_j_ham, axis=0) + len(training_set))

    return Nspam, teta_j_spam_laplace, teta_j_ham_laplace
  
'''
Prediction part of the Q3.1
'''  
def predict(test_set, labels, Nspam, teta_j_spam_laplace, teta_j_ham_laplace):
    result_matrix_spam = 0
    result_matrix_ham = 0
    pb_spam = Nspam / split_place
    pb_ham = 1 - pb_spam
    log_pb_spam = math.log(pb_spam)
    log_pb_ham = math.log(pb_ham)
    
    teta_j_spam_laplace = np.log(teta_j_spam_laplace)
    teta_j_ham_laplace = np.log(teta_j_ham_laplace)
    
    result_matrix_spam = test_set @ teta_j_spam_laplace
    result_matrix_ham = test_set @ teta_j_ham_laplace

    result_matrix_spam = result_matrix_spam + log_pb_spam
    result_matrix_ham = result_matrix_ham + log_pb_ham

    results = np.zeros((len(test_set)), dtype=int)
    
    for i in range(len(test_set)):   
        if result_matrix_spam[i] >= result_matrix_ham[i]:
            results[i] = 1
        else:
            results[i] = 0
        
    num_of_true = 0
    for i in range(len(test_set)):
        if labels[i] == results[i]:
            num_of_true = num_of_true + 1
                       
    return (num_of_true / len(test_set))

'''
Forward Selection Q3.1
Main Forward Selection method
'''
def feature_selection(feature_set, training_labels):
    print("Feature Selection starts.")
    F = []
    G = []
    A = []
    training_set = feature_set[0: split_place]
    test_set = feature_set[split_place: message_count]
    height, width = feature_set.shape
    total_acc = 0.01
    prev = 0
    while total_acc > prev:
      prev = total_acc
      for i in range(width):
          G.append(i)
          accuracy = evaluate(feature_set, training_set[:, G], test_set[:, G], training_labels)
          F.append(accuracy)
          G.remove(i)
      target_index = F.index(max(F))
      if target_index not in G:
        if len(A) == 0:
          total_acc = max(F)
          A.append(max(F))
          G.append(target_index)
        else:
          if max(F) > max(A):
            total_acc = max(F)
            A.append(max(F))
            G.append(target_index)
      F.clear()
      print("total_acc is: " + str(total_acc))
      print("indexes are: " + str(G))

    np.savetxt("forward_selection.csv", G, fmt='%d', delimiter=',')
    print("Feature Selection is completed.")
    return G, A, total_acc

def evaluate(all_training_set ,training_set, test_set, training_labels):
    Nspam, teta_j_spam_laplace, teta_j_ham_laplace = train(training_set, training_labels)
    acc = predict(test_set, test_labels, Nspam, teta_j_spam_laplace, teta_j_ham_laplace)
    return acc 
    

'''
Frequescy features Q3.2
Sorts the new feature set and trains and test it for 
'''
def trainforkfrequent(new_feature_matrix, training_labels, k):
  sorted_feature_matrix = new_feature_matrix[:,np.argsort(new_feature_matrix.sum(axis=0))[::-1]]
  training_set = sorted_feature_matrix[0: split_place]
  test_set = sorted_feature_matrix[split_place: message_count]
  height, width = sorted_feature_matrix.shape
  G = []
  accuracies = []
  for i in range(k):
    G.append(i)
    accuracy = evaluate(sorted_feature_matrix, training_set[:, G], test_set[:, G], training_labels)
    accuracies.append(accuracy)
    
  np.savetxt("frequency_selection.csv", accuracies, delimiter=',')  
  
      
'''
*********************** METHODS ENDS ***********************
'''

      
'''
*********************** USAGE OF THE METHODS FOR THE SOLUTION OF THE QUESTIONS ***********************
'''

#read message tokens
with open('tokenized_corpus.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        tokenized_corpus.append(row)
        rowCount = rowCount + 1   

N = len(tokenized_corpus)

#create vocabulary
vocabulary = []
for i in range(N):
    for key in tokenized_corpus[i]:
        if key not in vocabulary:
            vocabulary.append(key)

d = len(vocabulary)

#create feature set
feature_matrix = np.zeros([N, d], dtype=int)
message_count  = N
for i in range(N):
    for j in range(d):
        if vocabulary[j] in tokenized_corpus[i]:
            feature_matrix[i][j] = tokenized_corpus[i].count(vocabulary[j])

#SAVE FEATURE_SET
np.savetxt("feature_set.csv", feature_matrix, fmt='%d', delimiter=',')
#divide feature set into training set and test set
training_set = feature_matrix[0: split_place]
test_set = feature_matrix[split_place: N]

#read SMS labels
with open('labels.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        labels.append(int(row[0]))

#divide labels
training_labels = labels[0: split_place]
test_labels = labels[split_place: N] 
N = len(training_set)

#### Use acquired data for developing Naive Bayes Classifier

'''
Calculate parameters for Q2.2 and Q2.3
'''
T_j_spam, T_j_ham, Nspam, Nham = calculateSpamCountandTj(training_labels)
teta_j_spam, teta_j_ham, teta_j_spam_laplace, teta_j_ham_laplace , sum_spam_words, sum_ham_words = calculate_teta_probabilities(T_j_spam, T_j_ham)

'''
Question 2.2 Naive Bayes Train and Test with naive_bayes_model(test_set, test_labels)
'''
naive_bayes_accuracy_array = []
result, accuracy = naive_bayes_model(test_set, test_labels)
print("Accuracy naive_bayes_model: " + str(accuracy))
naive_bayes_accuracy_array.append(accuracy)
np.savetxt("test_accuracy.csv", naive_bayes_accuracy_array , delimiter=',')

'''
Dirichlet prior
Question 2.3 Naive Bayes Train and Test with naive_bayes_model_laplace(test_set, test_labels)
'''
naive_bayes_laplace_accuracy_array = []
result, accuracy = naive_bayes_model_laplace(test_set, test_labels)
print("Accuracy with laplace smoothing: " + str(accuracy))
naive_bayes_laplace_accuracy_array.append(accuracy)
np.savetxt("test_accuracy_laplace.csv", naive_bayes_laplace_accuracy_array , delimiter=',')


'''
create new vocabulary and feature set for the Part 3
'''
occurence_matrix = np.zeros([d], dtype=int)
test =  np.zeros([d], dtype=int)
vocab2_count = 0
vocabulary2 = []
for i in range(d):
    temp = feature_matrix[:, i]
    occurence_matrix[i] = np.sum(temp, axis = 0) 
    if occurence_matrix[i] < 10:
        occurence_matrix[i] = -99
    else:
        vocabulary2.append(vocabulary[i])
        vocab2_count = vocab2_count + 1
        
    
new_feature_matrix =  np.zeros([message_count, len(vocabulary2)], dtype=int)
occurence_matrix2 = np.zeros(len(vocabulary2), dtype=int)
removed_feature_count  = 0
j = 0
for i in range(d):
    if(occurence_matrix[i] >= 10):
        new_feature_matrix[:, j] = feature_matrix[:, i]
        temp = feature_matrix[:, i]
        occurence_matrix2[j] = np.sum(temp, axis = 0) 
        j = j + 1
    else:
        removed_feature_count = removed_feature_count + 1


##Call feature selection with new created data Q3.1
tic = time.perf_counter()  
print("\n")
F = feature_selection(new_feature_matrix, training_labels)
print("Accuracies are:")
print(F)
toc = time.perf_counter()
print(f"Time takes for forward selection is: {toc - tic:0.4f} seconds")


##Call features with most frequet to less reorder handled in the function 
## for the Q3.2
print("\n")
print("Frequency based training and prediction works")
trainforkfrequent(new_feature_matrix, training_labels, len(vocabulary2))



