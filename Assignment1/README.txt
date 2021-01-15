Program Explation: 
main.py file includes required methods and creates the reqired files.
Terminal Command: write "python main.py".

Notes about Q2.1:
These calculates parameters:

T_j_spam, T_j_ham, Nspam, Nham = calculateSpamCountandTj(training_labels)
teta_j_spam, teta_j_ham, teta_j_spam_laplace, teta_j_ham_laplace , sum_spam_words, sum_ham_words = calculate_teta_probabilities(T_j_spam, T_j_ham)

These makes test and returns accuracy:
result, accuracy = naive_bayes_model(test_set, test_labels)
print("Accuracy naive_bayes_model: " + str(accuracy))

Notes about Q2.2
Explanation:
These makes test and returns accuracy

Instuctions:
result, accuracy = naive_bayes_model_laplace(test_set, test_labels)
print("Accuracy with laplace smoothing: " + str(accuracy))
Notes about Q3.1
Explanation:
indexes are found with following after new_feature_matrix created

Instuction:
F = feature_selection(new_feature_matrix, training_labels)

Notes about Q3.2
Explanation:
feature_set is sorted inside the function 
training and test handled in method with k = volcabulary length for new feature_set

Instuction:
trainforkfrequent(new_feature_matrix, training_labels, len(vocabulary2))