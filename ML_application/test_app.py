def run_3():    
    import pandas as pd
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st
    from scipy.stats import norm
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.gaussian_process.kernels import Matern

    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    from sklearn.model_selection import train_test_split, GridSearchCV,KFold
    from sklearn.preprocessing import StandardScaler


    # Compute the Pearson correlation matrix and set target variable
    def correlation_plot_target_gen(df, target_col, y_val):
        df[target_col]= y_val
        corr_matrix = df.corr()
        target_corr = corr_matrix[[target_col]].drop([target_col], axis=0)
        # Plot the heatmap for the target variable correlations
        figg= plt.figure(figsize=(14, 16))
        sns.heatmap(target_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, vmin=-1, vmax=1)
        plt.title(f'Pearson Correlation Heatmap Between Features and {target_col}')
        ##plt.show()
        x=df.drop(columns= target_col)
        st.pyplot(figg)
        return x
        


    def correlation_plot_gen(df, target_col, y_val):
        df[target_col]= y_val
        corr_matrix = df.corr()
        # Plot the heatmap for the target variable correlations
        figg= plt.figure(figsize=(14, 16))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, vmin=-1, vmax=1)
        plt.title(f'Pearson Correlation Heatmap Between Features')
        ##plt.show()
        x= df.drop(columns= target_col)
        st.pyplot(figg)
        return x
        


    def linear_regression(X, y):
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)
        
        # Scaling the dataset
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initializing and training the model
        linear_model = LinearRegression()
        linear_model.fit(X_train_scaled, y_train)
        
        # Making predictions
        y_pred_train = linear_model.predict(X_train_scaled)
        y_pred_test = linear_model.predict(X_test_scaled)
        
        # Evaluating the model
        mae_linear = mean_absolute_error(y_test, y_pred_test)
        mape_linear = mean_absolute_percentage_error(y_test, y_pred_test)
        
        print('MAE:', mae_linear)
        print('MAPE:', mape_linear)
        
        return mae_linear, mape_linear, linear_model

    def knn_regression(X, y):
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling the dataset
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Hyperparameter tuning
        param_grid = {
            'n_neighbors': range(1, 10),
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
        }
        skf = KFold(n_splits=5, random_state=10, shuffle=True)

        grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=skf, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        best_params = grid_search.best_params_
        print("Best Parameters:", best_params)
        
        # Training the model with the best parameters
        knn_model = grid_search.best_estimator_
        knn_model.fit(X_train_scaled, y_train)
        
        # Making predictions
        y_pred_train = knn_model.predict(X_train_scaled)
        y_pred_test = knn_model.predict(X_test_scaled)
        
        # Evaluating the model
        mae_knn = mean_absolute_error(y_test, y_pred_test)
        mape_knn = mean_absolute_percentage_error(y_test, y_pred_test)
        
        print('MAE:', mae_knn)
        print('MAPE:', mape_knn)
        
        return mae_knn, mape_knn, knn_model

    ## Decision_tree regression ML model
    def decision_tree_regression(X, y):
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling the dataset
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Hyperparameter tuning
        param_grid = {
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        best_params = grid_search.best_params_
        print("Best Parameters:", best_params)
        
        # Training the model with the best parameters
        decision_tree_model = grid_search.best_estimator_
        decision_tree_model.fit(X_train_scaled, y_train)
        
        # Making predictions
        y_pred_train = decision_tree_model.predict(X_train_scaled)
        y_pred_test = decision_tree_model.predict(X_test_scaled)
        
        # Evaluating the model
        mae_dt = mean_absolute_error(y_test, y_pred_test)
        mape_dt = mean_absolute_percentage_error(y_test, y_pred_test)
        
        print('MAE:', mae_dt)
        print('MAPE:', mape_dt)
        
        return mae_dt, mape_dt, decision_tree_model

    def random_forest_regression(X, y):
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling the dataset
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [580, 600, 620],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        best_params = grid_search.best_params_
        print("Best Parameters:", best_params)
        
        # Training the model with the best parameters
        rf_model = grid_search.best_estimator_
        rf_model.fit(X_train_scaled, y_train)
        
        # Making predictions
        y_pred_train = rf_model.predict(X_train_scaled)
        y_pred_test = rf_model.predict(X_test_scaled)
        
        # Evaluating the model
        mae_rf = mean_absolute_error(y_test, y_pred_test)
        mape_rf = mean_absolute_percentage_error(y_test, y_pred_test)
        
        print('MAE:', mae_rf)
        print('MAPE:', mape_rf)
        
        return mae_rf, mape_rf, rf_model

    def gaussian_process_regression(X, y):
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling the dataset
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'alpha': [1e-5, 1e-2, 1],  # Regularization parameter
            'kernel': [Matern(length_scale=1.0, nu=1.5)], # Matern kernel with different nu values
            'normalize_y': [True, False]  # Whether to normalize the target values
        }

        # Initialize the Gaussian Process Regressor
        gpr = GaussianProcessRegressor(random_state=42)

        # Perform GridSearchCV to find the best parameters
        grid_search = GridSearchCV(gpr, param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        best_params = grid_search.best_params_
        print("Best Parameters:", best_params)
        
        # Training the model with the best parameters
        gp_model = grid_search.best_estimator_
        gp_model.fit(X_train_scaled, y_train)
        
        # Making predictions
        y_pred_train = gp_model.predict(X_train_scaled)
        y_pred_test = gp_model.predict(X_test_scaled)
        
        # Evaluating the model
        mae_gp = mean_absolute_error(y_test, y_pred_test)
        mape_gp = mean_absolute_percentage_error(y_test, y_pred_test)
        
        print('MAE:', mae_gp)
        print('MAPE:', mape_gp)
        
        return mae_gp,  mape_gp, gp_model

    def extra_trees_regression(X, y):
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling the dataset
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [400, 500, 600],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(ExtraTreesRegressor(random_state=42), param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        extra_trees_model = grid_search.best_estimator_
        extra_trees_model.fit(X_train_scaled, y_train)
        
        # Making predictions
        y_pred_test = extra_trees_model.predict(X_test_scaled)
        
        # Evaluating the model
        mae_et = mean_absolute_error(y_test, y_pred_test)
        mape_et = mean_absolute_percentage_error(y_test, y_pred_test)
        
        print('Extremely Randomized Trees Regression MAPE:', mape_et)
        
        
        return mae_et, mape_et, extra_trees_model

    def plot_mae(y_arr_mae):
        min_val= min(y_arr_mae)
        min_ind= y_arr_mae.index(min_val)
        fig= plt.figure(figsize=(10, 8))
        ml_mod= ['Linear Regression', 'Knn-Regression', 'Decision Tree', 'Random Forest', 'Gaussian Process Regression', 'Extra Trees Regressor']
        for iii, values in enumerate(y_arr_mae):
            plt.bar(ml_mod[iii], values, edgecolor= 'black', label= ml_mod[iii])
        plt.xlabel('Models', fontsize= 14)
        plt.ylabel('MAE')
        plt.title('Plot showing the Mean Absolute error for different Regression model', fontsize= 16)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig)
        st.info(f"The Best Model is {ml_mod[min_ind]}, with MAE {min_val}")

    def plot_mape(y_arr_mape):
        min_val= min(y_arr_mape)
        min_ind= y_arr_mape.index(min_val)
        fig= plt.figure(figsize=(10, 8))
        ml_mod= ['Linear Regression', 'Knn-Regression', 'Decision Tree', 'Random Forest', 'Gaussian Process Regression', 'Extra Trees Regressor']
        for iii, values in enumerate(y_arr_mape):
            plt.bar(ml_mod[iii], values, edgecolor= 'black', label= ml_mod[iii])
        plt.xlabel('Models', fontsize= 14)
        plt.ylabel('MAPE')
        plt.title('Plot showing the Mean Absolute Percentage error for different Regression model', fontsize= 16)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig)
        st.info(f"The Best Model is {ml_mod[min_ind]}, with MAPE {min_val}")

    def run_file(X, y, i): 
        ##main_df, output_parameter
        ##X= main_df
        ##y= output_parameter
        if i==0:
            mae, mape, model_c= linear_regression(X, y)
        if i==1:
            mae, mape, model_c = knn_regression(X, y)
        if i==2:
            mae, mape, model_c = decision_tree_regression(X, y)
        if i==3:    
            mae, mape, model_c = random_forest_regression(X, y)
        if i == 4:
            mae, mape, model_c = gaussian_process_regression(X, y)
        if i==5:
            mae, mape, model_c = extra_trees_regression(X, y)

        
        return model_c

    def parity_plot(y, y_pred, names):
        fig_pp= plt.figure()
        plt.scatter(y_pred, y)
        plt.plot([min(y), max(y)], [min(y), max(y)], color='C2')
        plt.xlabel(f'Predicted Values of {names}')
        plt.ylabel(f'Real Values of {names}')
        plt.title('Parity Plot to check real and predicted target variable')
        st.pyplot(fig_pp)


    def find_index(arr, models):
        for i in range (len(models)):
            if models[i] == arr:
                ind= i
                break
        return ind

    def test_page():
        st.title('Test your models here!')
        model_name= ['Linear Regression', 'Knn-Regression', 'Decision Tree', 'Random Forest', 'Gaussian Process Regression', 'Extra Trees Regressor']
        test_file = st.file_uploader("Upload Test CSV file (Make sure similar targets and features are present)", type=['csv'])
        st.session_state.test_data = pd.DataFrame()
        if test_file is not None:
            st.session_state.test_data = pd.read_csv(test_file)
            st.success(f'Thank you! Your File has been uploaded!')
            columns_target= st.selectbox("Choose the target variable", options= st.session_state.test_data.columns)
            st.session_state.new_data = st.session_state.test_data.drop(columns= columns_target)
            options_feat = ['Select All'] + st.session_state.new_data.columns.tolist()
            st.session_state.main_cols= st.multiselect("Choose features", options= options_feat)

            if columns_target:
                st.session_state.y_val= st.session_state.test_data[columns_target]
            if st.session_state.main_cols:
                if 'Select All' in st.session_state.main_cols:
                        st.session_state.main_cols= st.session_state.new_data.columns.to_list()
                st.session_state.x_val = st.session_state.test_data[st.session_state.main_cols]
                st.session_state.choose_model = st.selectbox("Please Choose ML Model", options=model_name)
                st.write("Please wait a little after you choose! Sorry for the delay!")
                if st.session_state.choose_model:
                    ind_mod= find_index(st.session_state.choose_model, model_name)
                    all_mod= run_file(st.session_state.x_val, st.session_state.y_val, ind_mod)
                    st.session_state.selected_model = all_mod
            
            if 'run_button' not in st.session_state:
                st.session_state.run_button= False
            
            if st.button("Run Model"):
                st.session_state.run_button = True

            if st.session_state.run_button:
                print (st.session_state.x_val)
                print (st.session_state.y_val)

                pred_values=  st.session_state.selected_model.predict(st.session_state.x_val)
                print (pred_values)
                mae = mean_absolute_error(st.session_state.y_val, pred_values)
                mape= mean_absolute_percentage_error(st.session_state.y_val, pred_values)
                ##y_given= st.session_state.y_val.values
                ##y_pred = pred_values
                parity_plot(st.session_state.y_val, pred_values, columns_target)
                st.write(f"Mean Absolute Error: {mae}")
                st.write(f"Mean Absolute Percentage Error: {mape}")

    test_page()

