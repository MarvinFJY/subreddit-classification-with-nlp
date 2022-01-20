# Project 3: Subreddit Classification with NLP

---

## Introduction and Problem Statement

A marketing and advertising firm that specializes in the Food & Beverage (F&B) industry has been engaged by both a sizeable tea company (think T2 Tea or TWG) and a sizeable coffee chain (think Starbuks) to run a digital advertising campaign (think Facebook and Instagram ads) over the Christmas festive period, that targets tea-lovers and tea collectors or coffee-lovers and coffee roasters respectively.

One issue that the advertising team faced was the association of tea with coffee, in which tea and coffee would have similar properties which might cause the marketing algorithm to target coffee drinkers instead and vice versa. Examples of these similar properties are: caffeine content, brewing method, tastes profiles and crockery used.

Our in-house data team has been tasked to run models to find some of the commonly used words among coffee and tea drinkers respectively and recommend words that the marketing algorithm should pick up when people type these words on the internet, allowing targeted facebook ads to show up on their feed. 

Ideally, our final model would result in tea advertisements being shown to only tea drinkers and coffee advertisements being shown to only coffee drinkers, thereby reaching a larger and more specific audience, hence making both advertising campaigns a success.


## Executive Summary

Reddit is a social news website and forum where content is socially curated and promoted by site members through voting. The site name is a play on the words "I read it." The site is composed of hundreds of subcommunities, known as subreddits. Each subreddit has a specific topic, such as technology, politics or music. Reddit's homepage, or the front page, as it is often called, is composed of the most popular posts from each default subreddit. The default list is predetermined and includes subreddits such as "pics," "funny," "videos," "news" and "gaming."

Reddit site members, also known as redditors, submit content which is then voted upon by other members. The goal is to send well-regarded content to the top of the site's front page. Content is voted on via upvotes and downvotes. The more upvotes a post gets, the more popular it becomes, and the higher up it appears on its respective subreddit or the front page. To access a subreddit via the address bar, simply type "reddit.com/r/subreddit name."<sup>[(source)](https://searchcio.techtarget.com/definition/Reddit)</sup>

To address our problem statement at hand, text data will be extracted from two different subreddits : `r/tea` and `r/Coffee` via the **Pushshift Reddit API**. The most recent 1500 posts from each subreddit have been scrapped and subsequently, null values, URL links, HTML special entities, spamming and moderator posts were removed to ensure the quality of vectorized words that we will be training our models with.

We have tested out a number of combinations of vectorizers, classifiers and text normalization methods:

**Vectorizers used:**

`CountVectorizer` and `Tfidfvectorizer`

**Models used:**

`Random Forest`, `Multinomial Naïve Bayes`, `Logistic Regression`

**Text Normalization used:**

`Lemmatization` and `Snowball Stemmer`


and found that a combination of CountVectorizer and Multinomial Naive Bayes was able to accurately classify 88.3% of posts correctly after engaging Snowball Stemmer. Our primary evaluation metric used was `accuracy` - a measure of the proportion of true predictions over all predictions because our classes were balanced ('tea' = 50.4%, 'coffee' = 49.6%). Furthermore, the main aim is not to minimize either False Positives or False Negatives, because ideally, both should be as low as possible (i.e. improper classifying of posts is equally bad). We will also be visualizing the best categorization method to be our final model by plotting the `ROC-AUC` curve.

To interpret our Multinomial Naive Bayes model, we will be calculating the empirical log probability of features for a given class. In fact, to convert `log_probability` into an actual probability score, we would have to exponentiate the log_probability (i.e `np.exp(feature_log_prob)`). However, in this case, as what we are really interested in is to get the top 15 predictor words that are most important when classifying subreddit posts into either 'Coffee' or 'Tea'. just doing `log_probability` alone would suffice. 

Now, looking at the results below, some of the top predictive words for 'Tea' included matcha, loose leaf, teapot, herbal and green. This makes sense as these are usually words most talked about even by the average tea drinkers. Hence, these are also the words that we would be recommending for the marketing algorithm to pick up when deciding on which users to show the digital tea advertisements to.

Similar for 'Coffee', the top predictive words for Multinomial NB included grinder, burr, v60, drip and aeropress 

Overall, we did not observe much overlap in the top predictive words. This is also likely the reason why our test accuracy score was relatively high (88%). 


![](/pictures/mnb_combine.png)



Next, we have also calculated the coefficients of our Logistic Regression model, which was out 2nd best performing model. By looking at the coefficients, we can determine the words that are most related to r/Coffee and r/tea. 

In the case of 'Tea' for the Logistic Regression model, the top predictive words also included matcha, loose leaf and tea pot while also included some additional words such as japanese and chai. While that of Logistic Refression included ginder, v60, pour and maker. We were also able to observe consistent predictive words between the models. Overall, I am satisfied the the top predictive words for both the Multinomial NB and Logistic Regression model are consistent with one another. This would further reaffirm the words that would get recommended in the marketing algorithm to pick up to show tea advertisements for tea drinkers. 


![](/pictures/lr_combine.png)

Our final model's limitations would mainly comprise of the misclassified posts (i.e. false predictions) made. Since the primary objective of the model was to accurately predict which subreddit should the post belong to. Therefore, if the model was unable to achieve this primary objective, it will be considered a limitation. After analysing some of the misclassified posts, there were indeed some true misclassifications, however we did also note that some misclassifications were due to the posts not being a quality one where it does not relate to either topics of 'Coffee' and 'Tea. Overall, I am satisfied with the low false positive/negative score of 108.

Some recommendations to improve on our model in future includes:

- a bigger corpus that incorporates a larger set of vocabulary on the topics of coffee and tea. This could also be taken from other sites such as food review blogs and related Facebook groups
- as mentioned earlier, preferences in the Food & Beverage scene are everchanging. In order for our model to maintain a comparatively high accuracy, it should ideally be re-trained at regular periods so that it does not contain out-of-date information and trends from coffee/tea drinkers, for example the 'Dalgona coffee' craze that took place at the start of COVID-19
- use word similarities (e.g. word2vec) to classify posts instead of frequency
- try other estimators like AdaBoost / GradientBoosting and try other vectorizers like lancaster Stemmer
- explore relationship between post content, number of comments, and upvote ratio
- use VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon to analyze the sentiments of posts