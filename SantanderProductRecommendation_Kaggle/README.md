One of my most recent and similar to real-world data competitions, with many milestones to be achieved, problems to be solved, and with a relatively big amount of data, more than what my computer could process. This made every step a challenge, from data loading to cleaning and model building.  I was able to build a model that scored 0.241, roughly 80% of the maximum score, using a 2GB RAM laptop and a code that takes around 3h to run.

The model consists of separate classification XGBoost models for every product, trained on three months June 2015, December 2015 and May 2016. This was decided so we could capture both seasonality (the test data is June 2016) and some trend.

We filtered the raw data only staying with the entries where a customer purchased a product. Since this is a relatively rare event, this reduced the data from ~17.000.000 to ~500.000.

All cleaning was performed in the most possible sensible way after having done the appropriate data exploration and visualization.

Edit: A couple of months after I finished the code I learned you could run your scripts on Kaggle Kernels, which means I wouldn't have had so much trouble with RAM space (ouch).
However, I know the challenge I faced of dealing with an amount of data that forced me to optimize my code and rethink my whole data manipulation process taught me many things. I still believe it is meaningful to try and tackle a huge amount of data with quite limited resources rather than just relying in sheer power of calculation, as this also happens in real world situations.
