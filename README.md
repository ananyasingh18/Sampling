# Sampling



We use a given dataset which is imbalanced and are required to firstly attain a balanced dataset and further use 5 sampling methods on 5 Ml models.

We Determine which techniwue gives higher accuracy for which model chosen.

Models chosen here are :
1.Logistic Regression      
2.Naive Bayes            
3.KNN                   
4.Decision Tree           
5.Random Forest Classifier 


Step one: Balancing the imbalanced dataset:
no of rows belonging to class 0: 763
no of rows belonging to class 1: 9
There are two main approaches to random resampling for imbalanced classification; they are oversampling and undersampling.
Random Oversampling: Randomly duplicate examples in the minority class.
Random Undersampling: Randomly delete examples in the majority class.

Dataset Before:
<img width="375" alt="Screenshot 2023-02-20 at 4 18 16 AM" src="https://user-images.githubusercontent.com/73638083/219979966-b88d1a36-94af-4a9e-908e-93182d0be988.png">
Dataset After:
<img width="375" alt="Screenshot 2023-02-20 at 4 19 23 AM" src="https://user-images.githubusercontent.com/73638083/219980006-09a13f65-f579-4de7-a99b-0b735ac46216.png">


Step two: Creating 5 samples:
We create 5 samples usinf different strategies like random sampling, stratified sampling, systemic sampling, cluster etc.
In all cases the appropriate sampling size is taken keeping in mind  the following factors:
1. Size of population
2. Margin of error
3. Confidence of error

Step three: applying Ml models on each sample to find out accuracy
<img width="693" alt="Screenshot 2023-02-20 at 4 25 06 AM" src="https://user-images.githubusercontent.com/73638083/219980265-c3fd6a17-1c82-4622-a7ab-d31a747408bb.png">

Conclusion: 
Random Forest classifer on a sample that has been Stratified gives best result.

