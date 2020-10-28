# Evaluation of the 2008 Global Financial Crisis Impact on Corporate Capital Structure Determinants of Publicly Listed Latin American Firms Through the Use of Machine Learning Tools

*Luciano Luca Fabbri Soto-Aguilar*



### **Abstract**

This study analyzes the impact of the global financial crisis of 2008 on the determinants of the capital structure of non-financial Latin American companies listed on the stock market to evaluate different financial theories associated with the capital structure. Specifically, a database covering the period 2002-2018 with financial information of companies in Brazil, Chile, Mexico and Peru is used to compare the effect of the determinants of leverage in the pre-crisis, crisis and post-crisis period with linear and machine learning models. The results show that the financial indicators of liquidity, size and tangibility have a significant impact on the total leverage of companies in all periods. Furthermore, it is concluded that the Trade-of theory better explains the capital structure of large unprofitable firms and that the Pecking Order theory better explains the debt structure of smaller, profitable companies, less tangible and more liquid firms.


### **Introduction**

The global financial crisis was caused by the collapse of the housing bubble in the United States due to defaults on subprime mortgages. The effects of the subprime crisis could be seen in early 2008, spreading initially to the North American financial system, and later to the global financial system. The global financial crisis caused a generalized economic downturn and the recession of several of the world’s largest economies such as the United States, Japan, Germany, and the United Kingdom, along with the largest bankruptcy in the history of the United States. The crisis exposed the failure of several highly leveraged firms, which casts doubt on the robustness of some of the major theories of capital structure, especially in periods of crisis. In addition, many of the studies on capital structure of firms present mixed results despite being carried out in periods of economic stability. It should be noted that the analysis of the capital structure could contribute in future crises to the elaboration of preventive financial containment plans at a firm specific level and to the development of public policies.

The capital structure of a firm refers to its configuration in terms of financing, which can come from equity or debt. The capital structure determines to a certain extent the ability of a company to comply with its obligations both in operational terms and in terms of the interests of its stakeholders. This said, capital structure theories are a set of principles that attempt to explain the mechanisms that determine decisions of corporate financing and their effect on the firm’s value. Modern theories incorporate influential elements of firm value such as information asymmetries, agency costs, corporate taxes and transaction costs into the analysis. The most dominant theories about capital structure are the Trade-off theory and the Pecking order theory, but  there are no empirical results that allow to conclude that one capital structure theory explains in a robust way the mechanisms and fundamentals that guide corporate financing decisions [1, 2, 3].

On one hand the Trade-off theory establishes that all firms have an optimal capital structure, which is determined by evaluating the trade-off between the benefits and the costs of financing through the issuance of debt or equity. This way, the 'optimal' structure or optimal debt ratio is the point where the marginal costs and benefits of issuing debt and equity are balanced. This theory explains why firms follow a moderate and cautious approach to debt problems, despite the tax shields granted by debt. It also predicts that firms focus on a certain ‘target’ capital structure, avoiding deviations from this structure over time [4]. On the other hand the pecking order theory [5, 4] establishes that firms do not have an optimal capital structure and that they follow a pecking order with respect to financing sources to minimize the problem of asymmetry of information between internal and external firm stakeholders, which is based on the empirical fact that firms show different preferences for the use of internal over external financing. This way, the pecking order theory establishes that firms first use internal sources of financing (retained earnings or liquid assets), then debt, and finally, financing through debt issuance.

The way in which the effect of different variables on the capital structure of firms has been explored in the literature normally consists of estimating some type of linear regression (OLS regressions [6], Weighted least squares [2], Gaussian mixture models [7, 8], fixed effects models and / or random effects models [9, 10, 3, 11, 12]) between observed leverage and a series of macroeconomic or firm-level explanatory variables. However, the linear nature of the commonly used models prevents the detection of non-linear, unspecified or unknown relationships between the variables, for which some authors used different modeling strategies to control temporal changes or non-linear relationships [10, 12, 9]. 

The machine learning approach is relatively new and is characterized by a high level of complexity that makes it difficult to interpret its results. However, there is evidence that machine learning algorithms can be adjusted significantly better than traditional statistical models to the same data, which ultimately translates into better predictions and interpretations that are closer to reality. Although machine learning models have been widely used in the financial area for forecasting stock fluctuations, these models have not been used for the specific purpose of interpreting the effect of different variables on the capital structure of companies. Therefore, it is expected that the present research will be a contribution to the theoretical understanding of the relationship between different financial indicators and conditions of companies with leverage, as well as a contribution to the empirical evidence of the impact of the 2008 global financial crisis in the determinants of the capital structure.




## **Methodology**

To develop this study, quarterly financial statements of Latin American companies were collected covering the period 2002-2018 through the Thompson Reuters Database representing a total of 57.475 entries and 97 columns. The initial exploration of the database showed that null entries became more predominant for periods before the year 2002 and in certain columns. In reference to country composition of the database, Brazil, Chile, Mexico and Peru accumulate 78% of the samples of the database. Regarding industry information present in the database, 54% of specific industries contain only one firm and 89% of specific industries contain less than 5 firms which makes this feature very sparse and difficult to use for many models. In addition, the manufacturing industry corresponds to the most predominant macro-industry in the database, where more than 300 firms are coded as belonging to this area. 

Considering the factors commonly used in the literature and the availability of information, the financial indicators of Total-debt-to-total-assets leverage ratio, growth, tangibility, profitability, liquidity and size of the companies were used to study the determinants of the capital structure of firms and its variation based on the 2008 global financial crisis. The composition and effect of this indicators on firm leverage according to Zeitun, et. al (2017) is summarized in the following table:

| Variable | Abbreviation | Composition | *Trade-off* | *Pecking order* |
|----------|-------------|------------- |:-----------:|:---------------:|
|Leverage  |LEV|Total debt to total assets|N/A|	N/A|
Growth	|GROW	|Quarterly percentage variation of total assets|	-|	+|
Tangibility	|TANG|	Fixed net Assets to total assets|	+/-|	-|
Profitability|	PROF|	Net income to total assets|	+	|-|
Liquidity|	LIQ	|Current assets to current liabilities|	+/-|	-|
Firm size|	SIZE|	Natural logarithm of total assets|	+|	-|



Two approaches to determine the impact of different variables on leverage and their variation in time were proposed. The first approach was formulated rigorously with the primary goal of avoiding any type of data leakage. In this context, data leakage could come from two sources; the explanatory variables, because the way the way the financial indicators are constructed (see Table 1), and temporal leakage because the panel nature of the data; which is produced when training a model with samples from future temporal data. The second approach consisted in fitting three static models which were trained to three macro time periods: pre-crisis, crisis and post-crisis, as if only cross-sectional data were used. It is important to remark that this last approach, more than explaining firm leverage, can result in explaining internal variable correlation because of data leakage from the explanatory variables. The principal motivation for this proposal is to trade-off validity for interpretability of the results.

The first approach consisted in training different sequential Mixed effect XGBoost regression models with lagged information of the up to one year of the financial explanatory variables for each quarter. Because of computing time, the hyperparameters of each model were optimized without considering the mixed effect computation. Specifically the hyperparameters of each XGBoost were determined by a grid search in which an XGBoost was trained in a two year period and cross-validated with half of the samples in the next quarter to determine the best set of hyperparameters according to the predefined grid. Then a Mixed Effects XGBoost was trained with the optimized hyperparameters on the training data and finally tested with the other half of the unseen data of the following period. The second approach, referred to as a static approach, estimated a total of three Mixed Effects XGBoost models and used the same logic for optimizing the XGBoost hyperparameters, but in this case the cross-validation and testing sets corresponded to the last four quarters of each macro-period. For both approaches, each model considered the impact of the different explanatory variables as fixed and the firm-specific effect as random.

Sequential approach:









Static approach:




Where f(X) is the unknown function estimated using an XGBoost, Br_it,Ch_it,Mx_it,Pe_it are dichotomic variables that indicate the country of headquarters of each firm, θ is the corresponding hyperparameter vector that parameterizes XGBoost models, Z_i is the covariate matrix of random effects, b_i is the vector of random effects for company i ,  and ϵ_it is the random error term. SHAP Values were used to interpret the results of each model. The computing of this values does not consider firm-specific effects on leverage, leaving only the fixed effects of the corresponding explanatory variables which facilitates the exploration of  ‘pure’ effects of capital structure determinants within the framework of the objectives of this work.

The sequential formulation enabled to examine a more continuous variation of the effects of the different variables on leverage, however, a drawback in terms of interpretability is introduced because of the repetition of each variable caused by the lagged effect. The effects on firm leverage of each financial explanatory variable were plotted in a 3d graph with dimensions of variable value, year of the corresponding model and impact on firm leverage. On the other hand, formulation two only enables to plot 2d plots for each macro time period. Lastly the effects of each variable and formulation were compared according to the estimated effect on leverage according to the different capital structure theories and between the two formulations.


## **Results**

An increase in model performance is achieved by including random firm effects which indicates the existence of non-negligible random effects. In other words, individual/internal firm factors have strong effects on leverage, which could be due to internal company policies or stakeholder management preferences. In terms of training and testing accuracy the sequential formulation shows satisfactory performance (see Figure 2), achieving an R2 testing score over 0.77 for 75% of the models and a minimum R2 training score of 0.96. Similarly, the static formulation also achieved satisfactory performance, with an average training and testing R2 score of 0.94 and 0.77, respectively. Respect to the performance of the models, it is possible to observe a positive trend between number of training data on the performance of the model in unobserved data (see Figure 2). It is possible to assume that obtaining new or more observations could result in an increase in performance. However, considering that the only way that the author of this project has to increase the volume of data is by imputing missing data under some criteria, and given the complex nature of the data used (panel data - time series for each variable per company) it is estimated that the use of any imputation technique could result in the addition of unnecessary noise instead of rich and meaningful information to explain firm leverage.




 *Figure 1.  Joint distribution: R2 train/test – Sequential Formulation.*             |  *Figure 2. Training samples and test score – Sequential Formulation.*
:-------------------------:|:-------------------------:
!![Figure 1](https://github.com/lutifabbri/Capital-Structure-Determinants/blob/main/Dataset%20Images/R2%20test-train-kde.png)  |  ![Figure 2](https://github.com/lutifabbri/Capital-Structure-Determinants/blob/main/Dataset%20Images/R2%20test%20vs%20training%20samples.png)



![FIG3](https://github.com/lutifabbri/Capital-Structure-Determinants/blob/main/Dataset%20Images/Combined.png)










In aggregate terms the static and sequential formulations achieved practically equal performance. However, because changes in the effects of different variables on total firm leverage might occur in short time spans (few samples with altered behavior), this could possibly have non-significant influence in the performance metric (R2 score). This phenomenon of short time span changes in effect of variables could not possibly be detected by the static model formulation and would not be reflected in the aggregate performance of the model. Because of this, is encouraged to use both model formulations with these considerations in mind to get accurate inferences about the impact and change of different firm characteristics on total leverage. 
In terms of the effects of the different variables on firm leverage: for the sequential formulation, independent of the lag of the variable, the effect on leverage is the same, and both formulations show the same trend effects of each variable on firm leverage. It is encouraged to the reader to check the SHAP plots in this repository and to use the plotting tool in the notebook to explore interactively the effects of the different variables and how they vary across time.

From both model formulations, the following results are obtained; liquidity and firm size indicators are those that explain, on average, the highest proportion of firm leverage.  On the other hand, all variables presented both positive and negative effects on leverage depending on the value of the indicator and none presented an effect that could have been considered linear. The trends presented are sufficiently similar between periods to consider that the global financial crisis did not have a significant effect on the effect of the different financial indicators on the leverage of the companies.
Considering the expected effects of each variable according to Zeitun et. al. (2017), the results suggest that one theory is not predominant over another because the effect of the different determinants of the capital structure can be positive or negative depending on the values they take. However, it is possible to suggest that one theory better explains the capital structure for certain types of firms better than the other. In this way,  and in general terms, for larger and unprofitable companies, the Trade-off theory explains in a better way the debt structure of firms and the Pecking Order theory better explains the capital structure of smaller, more profitable, less tangible and more liquid firms.



