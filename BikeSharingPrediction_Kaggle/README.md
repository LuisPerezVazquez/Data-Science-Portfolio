Code for Kaggle's competition: Bike Sharing Demand. Solution based on an Analytics Vidyha's tutorial (yeah I know it is based on a tutorial but don't kick me out yet, there is a reason for it!) 

Tutorial Link: https://www.analyticsvidhya.com/blog/2015/06/solution-kaggle-competition-bike-sharing-demand/

Motivation: I had two main goals when I started this code.

                     1. Practice with timestamped data.
                     
                     2. The tutorial is done in R studio, while I use Python. Ever since I chose this language, I always wondered about it's 
differences with R, especially how big would be the difference in the final accuracy obtained between both languages if the same steps were performed. Also as a bonus I thought it would be interesting too to learn the 'translations' from one language to another.


Results: It turns out in this case R is about a 10% more accurate than Python when predicting the bike sharing demand (having performed the same steps). Difference is thought to be in how decission tree regressor algorithms are built in each language. Not only that, but also it was discovered that using Decission Trees to bin the time and temperature data, while seeming effective on R (as seen on the tutorial), turned out to be counterproductive on Python, the best score being achieved when this process was not performed at all.


Random Forest hyperparameter fine tuning was not performed since that was not the point of the project, more focused on Feature Engineering.
