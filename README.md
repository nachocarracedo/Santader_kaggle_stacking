# Santader_kaggle_stacking
Santander classification Kaggle competiton: https://www.kaggle.com/c/santander-customer-satisfaction

I used this competition to create model ensembles which usually improve the loss score. Here is a nice explanation: http://mlwave.com/kaggle-ensembling-guide/

I practice with **stacking**, here is the procedure I followed:

2-fold stacking:

- Split the train set in 2 parts: train_a and train_b
- Fit a first-stage model on train_a and create predictions for train_b
- Fit the same model on train_b and create predictions for train_a
- Finally fit the model on the entire train set and create predictions for the test set.
- Now train a second-stage stacker model on the probabilities from the first-stage model(s).

More detailed info:

- First, I do some basic feature engineering and thn I do feature selection to remove noise (using best features from Gradient Boosting).
- For first-stage models I used RF Gini, RF Entropy, 2 x Gradient Boosting, and AdaBoost.
- For second-stage model I tried 3: 1) Logistic regression 2) RF 3) Weights on 1st stage models.

Conclusion - next steps.

- I got a little improvement with stacking but not a big one. Throwing more models into the mix will probably help.
- Use skitlearn to find weights for 2nd stage. I did it manually and it is very expensive.
- Save ensambles for the last step of you analysis, after getting you best score and having tried different features. Evaluate if the gain is worth the effort.

TODO:

- Add comments !!!! I'm sorry for not doing this before hand; I know it makes the project unreadable. Lack of time is responsible but I’ll introduce them as soon as I can.
