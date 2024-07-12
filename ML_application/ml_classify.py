def run_2():    
    import pandas as pd
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    ##import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.base import clone
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
    from sklearn.svm import SVC, SVR
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix, accuracy_score, f1_score

    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

    def run_file(X, y): 
        ##main_df, output_parameter
        ##X= main_df
        ##y= output_parameter
        acc_dt, mean_dt= ml_classify(X,y)
        rf_cv_acc, mean_rf= ml_classify_rf(X,y)
        acc_svm, mean_svm= ml_classify_svm(X,y)
        acc_gbm, mean_gbm= ml_classify_gbm(X,y)
        acc_logr, mean_logr= ml_classify_log_reg(X,y)
        acc_knn, mean_knn = classify_grid_KNN(X,y)
        acc_nb, mean_nb= classify_grid_Naive_Bayes(X,y)
        y_arr= [acc_dt, rf_cv_acc, acc_svm, acc_gbm, acc_logr, acc_knn, acc_nb]
        
        fig= plt.figure(figsize=(10, 8))
        ml_mod= ['Decision Tree', 'Random Forest', 'SVM', 'GBM', 'Logistic Regression', 'K-NN', 'Naive Bayes']
        for iii, values in enumerate(y_arr):
            plt.bar(ml_mod[iii], values, edgecolor= 'black', label= ml_mod[iii])
        plt.xlabel('Models accuracy', fontsize= 14)
        plt.ylabel('Accuracy')
        plt.title('Plot showing the accuracy for different Classification model', fontsize= 16)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig)


    def ml_classify(X, y):
        ##X = df.drop(['Mini Trip', 'Classification'], axis=1)
        ##y = df['Classification']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        
        cv_scores = cross_val_score(clf, X, y, cv=10)  
        print (cv_scores)
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)

        print ("Mean Acc:", mean_accuracy)
        print ("Std Acc:", std_accuracy)
        
        y_pred = clf.predict(X_test)
        data_pred=(classification_report(y_test, y_pred))
        report_lines = data_pred.strip().split('\n')
        accuracy_line = report_lines[-3]
        accuracy_tokens = accuracy_line.strip().split()
        accuracy = float(accuracy_tokens[-2])
        print (data_pred)
        return accuracy, mean_accuracy

    def ml_classify_rf(X, y):
        ##X = df.drop(['Mini Trip', 'Classification'], axis=1)
        ##y = df['Classification']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        cv_scores = cross_val_score(clf, X, y, cv=10)  
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)

        print ("Mean Acc:", mean_accuracy)
        print ("Std Acc:", std_accuracy)
        
        y_pred = clf.predict(X_test)
        data_pred = classification_report(y_test, y_pred)
        report_lines = data_pred.strip().split('\n')
        accuracy_line = report_lines[-3]
        accuracy_tokens = accuracy_line.strip().split()
        accuracy = float(accuracy_tokens[-2])
        print(data_pred)
        return accuracy, mean_accuracy

    ##rf_cv_acc, mean_rf= ml_classify_rf(info_data_3)
    def ml_classify_svm(X, y):
        ##X = df.drop(['Mini Trip', 'Classification'], axis=1)
        ##y = df['Classification']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        svm_classifier = SVC(kernel='linear', random_state=42)
        svm_classifier.fit(X_train, y_train)

        cv_scores = cross_val_score(svm_classifier, X, y, cv=10)  
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)

        print ("Mean Acc:", mean_accuracy)
        print ("Std Acc:", std_accuracy)

        y_pred = svm_classifier.predict(X_test)
        data_pred = classification_report(y_test, y_pred)
        report_lines = data_pred.strip().split('\n')
        accuracy_line = report_lines[-3]
        accuracy_tokens = accuracy_line.strip().split()
        accuracy = float(accuracy_tokens[-2])
        print(data_pred)
        return accuracy, mean_accuracy

    ##acc_svm, mean_svm= ml_classify_svm(info_data_3)
    ##Pit's code
    def ml_classify_gbm(X, y):
        ##X = df.drop(['Mini Trip', 'Classification'], axis=1)
        ##y = df['Classification']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        gbm = GradientBoostingClassifier(random_state= 42)   ###n_estimators=100, learning_rate= 0.1, 
        gbm.fit(X_train, y_train)
        
        cv_scores = cross_val_score(gbm, X, y, cv=10)  
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = (np.std(cv_scores))
        
        print ("Mean Acc:", mean_accuracy)
        print ("Std Acc:", std_accuracy)
        
        y_pred = gbm.predict(X_test)
        data_pred = classification_report(y_test, y_pred)
        report_lines = data_pred.strip().split('\n')
        accuracy_line = report_lines[-3]
        accuracy_tokens = accuracy_line.strip().split()
        accuracy = float(accuracy_tokens[-2])
        print(data_pred)

        return accuracy, mean_accuracy

    ##acc_gbm, mean_gbm= ml_classify_gbm(info_data_3)
    ##Pit's code
    def ml_classify_log_reg(X, y):
    ## X = df.drop(['Mini Trip', 'Classification'], axis=1)
        ##y = df['Classification']
        
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define a pipeline that includes scaling and logistic regression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Scaling step
            ('log_reg', LogisticRegression(max_iter=1000, random_state=96))  # Logistic Regression step
        ])

        # Fit the pipeline on the training data
        pipeline.fit(X_train, y_train)

        cv_scores = cross_val_score(pipeline, X, y, cv=10)  
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = (np.std(cv_scores))
        
        print ("Mean Acc:", mean_accuracy)
        print ("Std Acc:", std_accuracy)
        
        # Predicting the Test set results
        y_pred = pipeline.predict(X_test)

        # Classification Report
        data_pred = classification_report(y_test, y_pred)
        print(data_pred)

        # Extract accuracy from classification report
        report_lines = data_pred.strip().split('\n')
        accuracy_line = report_lines[-3]
        accuracy_tokens = accuracy_line.strip().split()
        accuracy = float(accuracy_tokens[-2])
        print("Accuracy:", accuracy)

        return accuracy, mean_accuracy

    ##Shawn's code
    def classify_grid_KNN(X, y):
        ##X = df.drop(['Mini Trip', 
                    ## 'Classification',                 
                ## ], axis=1)
    ## y = df['Classification']

        # Splitting the dataset into training（80%） and testing datasets(20%）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define a grid of parameters for hyperparameter tuning
        param_grid = {
        'knn__n_neighbors': range(1, 10),
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
        }

        # Define a pipeline that includes scaling and k-Nearest Neighbors classifier
        pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scaling step
        ('knn', KNeighborsClassifier())
        ])

        # Grid search with cross-validation
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring='accuracy')
        grid_search.fit(X_train, y_train)   

        # Best parameters and best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)

        # Using the best estimator directly
        best_pipeline = grid_search.best_estimator_

        # Cross-validation score
        cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=10)
        print("Cross-Validation Scores:", cv_scores)
        print("Mean CV Accuracy:", np.mean(cv_scores))
        mean_accuracy= np.mean(cv_scores)
        # Predicting the Test set results
        y_pred = best_pipeline.predict(X_test)
        
        data_pred = classification_report(y_test, y_pred)
        ##accuracy = grid_search.best_score_  
        print(data_pred)
        report_lines = data_pred.strip().split('\n')
        accuracy_line = report_lines[-3]
        accuracy_tokens = accuracy_line.strip().split()
        accuracy = float(accuracy_tokens[-2])
        print("Accuracy:", accuracy)

        # F1 Score
        f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass classification
        print("F1 Score:", f1)
        
        ##print("The accuracy is:", accuracy)  
        return accuracy, mean_accuracy
    ##acc_knn, mean_knn = classify_grid_KNN(info_data_3)
    ##Shawn's Code
    def classify_grid_Naive_Bayes(X, y):
    ## X = df.drop(['Mini Trip', 
    ##              'Classification',                 
    ##             ], axis=1)
        
        ##y = df['Classification']

        # Splitting the dataset into training（80%） and testing datasets(20%）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define a grid of parameters for hyperparameter tuning
        param_grid = {
        'naive_bayes__var_smoothing': [1e-7, 1e-5, 1e-3, 1e-1, 1],
        }

        # Define a pipeline that includes scaling and k-Nearest Neighbors classifier
        pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scaling step
        ('naive_bayes', GaussianNB())
        ])

        # Grid search with cross-validation
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring='accuracy')
        grid_search.fit(X_train, y_train)   

        # Best parameters and best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)

        # Using the best estimator directly
        best_pipeline = grid_search.best_estimator_

        # Cross-validation score
        cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=10)
        print("Cross-Validation Scores:", cv_scores)
        print("Mean CV Accuracy:", np.mean(cv_scores))
        mean_accuracy= np.mean(cv_scores)
        # Predicting the Test set results
        y_pred = best_pipeline.predict(X_test)
        
        data_pred = classification_report(y_test, y_pred)
        ##accuracy = grid_search.best_score_  
        print(data_pred)
        ##accuracy = grid_search.best_score_  
        print(data_pred)
        report_lines = data_pred.strip().split('\n')
        accuracy_line = report_lines[-3]
        accuracy_tokens = accuracy_line.strip().split()
        accuracy = float(accuracy_tokens[-2])
        print("Accuracy:", accuracy)
        # F1 Score
        f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass classification
        print("F1 Score:", f1)
        
        ##print("The accuracy is:", accuracy)  
        return accuracy, mean_accuracy
    ##acc_nb, mean_nb= classify_grid_Naive_Bayes(info_data_3)

    # Title of the web application
    st.title('Machine Learning for all- Classification Model')

    

    # Input field for the user to enter a number
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        info_data_3 = pd.read_csv(uploaded_file)
        st.success(f'Thank you! Your File has been uploaded!')

        columns_to_drop = st.multiselect(
                "Choose the columns you want to drop:",
                options=info_data_3.columns
            )
        if columns_to_drop:
            df = info_data_3.drop(columns=columns_to_drop)
            output_parameter = st.selectbox("Choose your output for classification:", options=df.columns)
            main_df= df.drop(columns=output_parameter)  
            y_values= df[output_parameter]
            if output_parameter:
                r= st.button ("Click to run the models!")
                if r== True:
                    run_file(main_df, y_values)
                    st.info("The one with the highest accuracy is the best model!")
        else:
            st.info("Please choose from the menu above!")

def end_code():
    import streamlit as st
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        

