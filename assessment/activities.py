import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from MLExpression import ridgeRegression, elasticRegression, lassoRegression, selection
from MLExpression import linearRegression, polyRegression, showFigure

houseDict = json.load(open("houseDict.json"))

df = pd.read_csv('../dataSet/kc_house_data.csv')

x_dataDrop_key = ['price', 'id', 'date']

y_dataDrop_key = ['price']

size = 0.3

random = 10

process_running = True

print("Here is the House Sales in King Country, USA.\n")

while(process_running):

    print("We have a lot of ML excrise such as .........\n")

    print("The first look at the data, the processing parameter:hea\n")

    print("The information of the data, the processing parameter:inf\n")

    print("The describle of the data, the processing parameter:des\n")

    print("The histogram of the data, the processing parameter:fig\n")
          
    print("please add second input with space to show which key you want to show?\n")

    print("The Linear Regression of the data, the processing parameter:lin\n")

    print("The PolynomialFeatures and Linear Regression of the data, the processing parameter:pol\n")

    print("The Ridge of the data, the processing parameter:rid\n")

    print("The Lasso of the data, the processing parameter:las\n")

    print("The Elastic Net of the data, the processing parameter:net\n")

    print("The Regression of the data, the processing parameter:sns\n")

    print("The Feature Selection of the data, the processing parameter:sel\n")

    print("Which ML excrise you want to show?\n")

    str = input()
    
    strSplit = str.split(" ") if str else None

    processing_name = strSplit[0] if str else None
    
    key = strSplit[1] if str and len(strSplit) > 1 else None

    print()

    if str == None: process_running = False

    elif processing_name == 'hea': df.head()

    elif processing_name == 'inf': df.info()

    elif processing_name == 'des': df.describe()

    elif processing_name == 'sns':

        sns.pairplot(
            df[
                [
                    'price', 
                    'sqft_living', 
                    'sqft_basement', 
                    'yr_built', 
                    'zipcode'
                ]
            ]
        )

        plt.show()

    elif processing_name == 'fig':

        figure = showFigure(
            df, 
            key, 
            houseDict
        )

        figure.show()

        if key == "bedrooms": df.loc[df[key] == 33]

        print(df[key].describe(), "\n")

    elif processing_name == 'lin':

        expression = linearRegression(
            df, 
            x_dataDrop_key, 
            y_dataDrop_key, 
            size, 
            random,
            fit_intercept=True
        )

        print("mreg intercept_:", expression["intercept"], "\n")

        print("mreg coef_:", expression["coef"], "\n")

        print("Train Set R-square Val: {:.2f}".format(expression["train_score"]), "\n")
        
        print("Test Set R-square Val: {:.2f}".format(expression["test_score"]), "\n")
        
        ols_m = sm.OLS(expression["y_train"], sm.add_constant(expression["x_train"])).fit()
        
        ols_m.summary()

    elif processing_name == 'pol':
        
        expression = polyRegression(
            df, 
            x_dataDrop_key, 
            y_dataDrop_key, 
            size, 
            random,
            2, 
            include_bias=False,
            fit_intercept=True
        )
        
        print("mreg intercept_:", expression["intercept"], "\n")
        
        print("mreg coef_:", expression["coef"], "\n")
        
        print("Train Set R-square Val: {:.2f}".format(expression["train_score"]), "\n")
        
        print("Test Set R-square Val: {:.2f}".format(expression["test_score"]), "\n")

    elif processing_name == 'rid':
        
        expression = ridgeRegression(
            df, 
            x_dataDrop_key, 
            y_dataDrop_key, 
            size, 
            random, 
            alpha=0.5
        )
        
        print("Train Set R-square Val: {:.2f}".format(expression["train_score"]), "\n")
        
        print("Test Set R-square Val: {:.2f}".format(expression["test_score"]), "\n")

    elif processing_name == 'las':

        expression = lassoRegression(
            df, 
            x_dataDrop_key, 
            y_dataDrop_key, 
            size, 
            random, 
            alpha=0.5
        )

        print("Train Set R-square Val: {:.3f}".format(expression["train_score"]), "\n")

        print("Test Set R-square Val: {:.3f}".format(expression["train_score"]), "\n")

        print("num_of_IV:", np.sum(expression["coef"] != 0), "\n")

    elif processing_name == 'net':

        expression = elasticRegression(
            df, 
            x_dataDrop_key, 
            y_dataDrop_key, 
            size, 
            random, 
            alpha=0.5, 
            l1_ratio=0.1
        )
        
        print("Train Set R-square Val: {:.3f}".format(expression["train_score"]), "\n")
        
        print("Test Set R-square Val: {:.3f}".format(expression["train_score"]), "\n")
        
        print("num_of_IV:", np.sum(expression["coef"] != 0), "\n")

    elif processing_name == 'sel':

        id_vars = 'sqft_living'

        value_name = 'value'

        var_name = 'price'

        corr = df.corr()

        sns.heatmap(
            corr, 
            xticklabels=corr.columns, 
            yticklabels=corr.columns
        )
        
        expression = selection(
            df, 
            ['sqft_living', 'grade', 'bathrooms'], 
            ['price'], ['sqft_living', 'actual_price', 'predicted_price']
        ) 
        
        print("model score:", expression["model_score"], "\n")
        
        print("mean absolute error:", expression["mean_absolute_error"], "\n")
        
        x_test_transp = expression["x_test_transp"]
        
        df_test = pd.DataFrame({
            'sqft_living': x_test_transp[0],
            'grade': x_test_transp[1],
            'bathrooms': x_test_transp[2],
            'actual_price': expression["y_test"],
            'predicted_price': expression["y_pred"]
        })
        
        print(df_test.head(), "\n")
        
        df_plot_test = df_test['sqft_living'].melt(
            id_vars, 
            var_name=var_name,  
            value_name=value_name
        )
        
        #<AxesSubplot:xlabel='sqft_living', ylabel='price'>
        sns.scatterplot(
            data=df_plot_test, 
            x=id_vars, 
            y=value_name, 
            hue=var_name
        )

    else: sys.exit('0')
