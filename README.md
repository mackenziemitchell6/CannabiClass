# CannbiClass
By Mackenzie Mitchell

## Overview
The purpose of this project was to classify strains of cannabis as indica, sativa, or hybrid using a classification model. The data was obtained from The Strain API and from web scraping WikiLeaf. I was able to classify cannabis strains based on their effects with 69.48% accuracy and an average F-1 score across all classes of 0.65 using a Scalar Vector Machine model.
### Process
Dummy Baseline Model   
Classification models with all features  
Classification models with hand selected features  
Classification models with PCA prinicpal component variables  
Grid Search 
### Goal
The goal is to predict all indica strains as either indica or hybrid, and to predict all sativa strains as either sativa or hybrid. This is because of the significant difference in effects and results from using these two different types of strains. While indica is better for night use and to relax, sativa is better for day use. Therefore, if someone wanted to consume sativa but instead consumed indica, they would be very upset as they would become tired and lazy during the time they may have wanted to be creative and uplifted. Additionally, of course, the overall goal is to predict as many of the observations as the correct strain as possible. 

  #### Venn Diagram
![Venn Diagram](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Visualizations/VennDiagram_Indica_Sativa.png "Venn Diagram")

### Obtaining Data and Feature Engineering
The goal of this project was intially to classifiy a cannabis strain as indica, sativa, or hybrid based on it's chemcial components. After webscraping, I was only able to get about 500 observations. This was insufficient for my objective. 
Changing gears, I found The Strain API (http://strains.evanbusse.com/). Using this tool, I was able to gather almost 2000 observations with 33 features scraped from this API.

The features included the various effects of the cannabis strain as well as the thc content. The thc content of each strain was obtained from webscraping https://www.wikileaf.com/strains. Once joining the dataframes, I was left with about 1300 observations and 37 features, as I also engineered features that represented scores of positive effects, negative effects, and medical effects for each cannabis variant observation. 

## EDA
  ### Class Distribution
![Class Distributions](https://github.com/mackenziemitchell6/CannabiClass/blob/master/TargetDistplots.png "Class Distribution")
There did not seem to be class imbalance at first as no one class contained half the data, however, if I were to go back I might SMOTE or use another method in order to treat the fact that there were 1004 observations of hybrid strains and only 966 observations split between indica and sativa strains within my dataset.
  ### THC Content
![THC Content](https://github.com/mackenziemitchell6/CannabiClass/blob/master/thc.png "THC Content")

The THC content seemed to be pretty stable across all classes, with no strain type having significantly more or less THC than the others. There are some outliers in the indica class with higher levels of THC while the other two classes do not experience these outliers that have these excess levels of THC. 

![THC Violin Plot](https://github.com/mackenziemitchell6/CannabiClass/blob/master/ViolinPlots.png "THC plot")
  ### Feature Selection
 In looking at how features change across the different classes, I was able to hand select certain features that seemed to distinctly change across the three classes. The selected features are shown as follows:
  #### THC
  ![Class Distributions](https://github.com/mackenziemitchell6/CannabiClass/blob/master/thc.png "Class Distribution")
  #### Relaxed
  ![Relaxed](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Relaxed.png "Relaxed Distribution")
  #### Hungry
  ![Hungry](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Hungry.png "Hungry Distribution")
  #### Sleepy
  ![Sleepy](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Sleepy.png "Sleepy Distribution")
  #### Depression
  ![Depression](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Depression.png "Depression Distribution")
  #### Insomnia
  ![Insomnia](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Insomnia.png "Insomnia Distribution")
  #### Pain
  ![Pain](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Pain.png "Pain Distribution")
  #### Euphoric
  ![Euphoric](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Euphoric.png "Euphoric Distribution")
  #### Creative
  ![Creative](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Creative.png "Creative Distribution")
  #### Energetic
  ![Energetic](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Energetic.png "Energetic Distribution")
  #### Dry Mouth
  ![Dry Mouth](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Dry%20Mouth.png "Dry Mouth Distribution")
  #### Nausea
  ![Nausea](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Nausea.png "Nausea Distribtuion")
  #### Uplifted
  ![Uplifted](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Uplifted.png "Uplifted Distribution")
  #### Fatigue
  ![Fatigue](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Fatigue.png "Fatigue Distribution")
  #### Focused
  ![Focused](https://github.com/mackenziemitchell6/CannabiClass/blob/master/Focused.png "Focused Distribution")
  ### PCA 
  After implementing PCA and running models, the results actually seemed to decrease in their performance. However, this was great experience for me to work with PCA as I was confused about this concept and how to implement it prior. 
  ![PCA Plot](https://github.com/mackenziemitchell6/CannabiClass/blob/master/PCAPlot.png "PCA")
  
  ## Modeling
  ### Baseline Dummy Model
  The most populated class is hybrid, so predict all strains as hybrid. 
  Accuracy: 57.85%
  Average F-1 Score: 0.7
  
  ![Baseline Matrix](https://github.com/mackenziemitchell6/CannabiClass/blob/master/BaselineConfuseMatrix.png "Baseline Matrix")
  
  ### Logistic Regression Model
  This model improved accuracy from the baseline, but was not the best model. Logistic Regression with PCA components reduced accuracy to 57.85% 
  Accuracy: 66.28%
  Average F-1 Score: 0.6
  
  ![Logistic Matrix](https://github.com/mackenziemitchell6/CannabiClass/blob/master/LogisticConfuseMatrix.png "Logistic Matrix")
  
  ### Best KNN Model (K=19)
  This model improved accuracy from the baseline, but was not the best model. 
  Accuracy: 66.86%
  Average F-1 Score: 0.6
  
  ![Best KNN Matrix](https://github.com/mackenziemitchell6/CannabiClass/blob/master/BestKKNNConfuseMatrix.png "Best KNN Matrix")

  ### Best Random Forest
  Accuracy: 68.31%
  Average F-1 Score: 0.6
  
  ![Best Random Forest](https://github.com/mackenziemitchell6/CannabiClass/blob/master/ForrestAllDataMtx.png "Best RF Matrix")
  
  ### Scalar Vector Machine
  Accuracy: 69.48%
  Average F-1 Score: 0.63
  
  ![SVM mtx](https://github.com/mackenziemitchell6/CannabiClass/blob/master/SVMmtx.png "SVM mtx")
  
