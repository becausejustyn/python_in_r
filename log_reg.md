## Title

The `python` vs `R` debate is one that will probably never end. I started with R and avoiding python because it can be overwhelming for individuals who do not come from a computer science background. Overtime I began to appreciate python more and while I am more comfortable with `R`, the language I used essentially comes down to these two things:

- If I am spending more time wrangling or visualising data, I use `R`
- If I want to focus more on the math, `python` is my choice since I find it easier to do things from scratch.

I am going to do a logistic regression model using NFL data so I am going to use `R` to wrangle it and then use `tidymodels`, however, I will also run a similar analysis in `python` just to show the differences. 

```r
library(reticulate)
```

```r
#play by play data
pbp <- nflreadr::load_pbp(2019:2020, file_type = 'rds')
```

```r
library(tidyverse)
library(zoo)
```

Basic summary of each game, the teams, winner, and expected points added stats.

```r
pbp %>% 
  filter(week <= 16) %>% 
  mutate(
    home_off_epa = ifelse(posteam == home_team & play == 1, epa, NA),
    away_off_epa = ifelse(posteam == away_team & play == 1, epa, NA),
    home_def_epa = ifelse(defteam == home_team & play == 1, epa, NA),
    away_def_epa = ifelse(defteam == away_team & play == 1, epa, NA)
  ) %>% 
  group_by(home_team, away_team, game_date) %>%
  summarise(
    home_score = max(total_home_score, na.rm = T),
    away_score = max(total_away_score, na.rm = T),
    home_off_epa = mean(home_off_epa, na.rm = T),
    away_off_epa = mean(away_off_epa, na.rm = T),
    home_def_epa = mean(home_def_epa, na.rm = T),
    away_def_epa = mean(away_def_epa, na.rm = T),
    .groups = 'drop'
  ) %>% 
  mutate(
    game_date = lubridate::ymd(game_date)
  ) %>% 
  arrange(game_date) %>% 
  mutate(outcome = case_when(
    home_score > away_score ~ 'home',
    away_score > home_score ~ 'away',
    TRUE ~ 'Tie'
  )) %>% 
  filter(outcome != 'Tie') -> games
```

```r
games %>% 
  mutate(outcome = outcome == 'home') %>% 
  rename_with(~ str_replace(., "home", "team")) %>% 
  rename_with(~ str_replace(., "away", "opp")) -> home_teams
```

```r
games %>% 
  mutate(outcome = outcome == 'home') %>% 
  rename_with(~ str_replace(., "away", "team")) %>% 
  rename_with(~ str_replace(., "home", "opp")) %>% 
  mutate(outcome = outcome == FALSE) -> away_teams
```


```r
df <- bind_rows(home_teams, away_teams)

df <- df %>% 
  arrange(game_date) %>% 
  group_by(team_team) %>% 
  mutate(
    across(starts_with("team_"), ~rollmean(., 4, align = 'right', fill = NA))
    ) %>% 
  ungroup() %>% 
  group_by(opp_team) %>% 
  arrange(game_date) %>% 
  mutate(
    across(starts_with("opp_"), ~rollmean(., 4, align = 'right', fill = NA))
  ) %>% 
  filter(lubridate::year(game_date) == 2020) %>% 
  ungroup()

df %>% 
  select(contains('epa'), outcome) %>% 
  mutate(outcome = ifelse(outcome == TRUE, 1, 0)) -> glm_data


mylogit <- glm(outcome ~ ., data = glm_data, family = 'binomial')

glm_data$pred <-predict(mylogit, newdata = df %>% select(-outcome))

glm_data %>% 
  mutate(pred = exp(pred)/(1+exp(pred))) %>% 
  mutate(pred = ifelse(pred >= .5, 1, 0)) %>% 
  count(outcome == pred) %>% 
  mutate(acc = n / sum(n))
```

```r
summary(mylogit)
```

```r
data <- reticulate::r_to_py(glm_data)
```

## Python

By default, sklearn already imposes a regularization penalty. I’ll be diving more into that in a later post.


```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import load_diabetes
```

The same data, without the R predictions.

```python
data = r.data.drop(['pred'], axis=1)
data['outcome'] = data['outcome'].astype(int)
print(data.head())
```

```python
predictors = ['team_off_epa', 'opp_off_epa', 'team_def_epa', 'opp_def_epa']

X_train, X_test, y_train, y_test = train_test_split(data[predictors], data['outcome'], test_size = 0.2)

zscore = StandardScaler()
zscore.fit(X_train)
```

```python
Xz_train = zscore.transform(X_train)
Xz_test = zscore.transform(X_test)
```

```python
myLogit = LogisticRegression()
myLogit.fit(Xz_train, y_train)
```

```python
predictedVals = myLogit.predict(Xz_test)
accuracy_score(y_test, predictedVals)
```

```python
confusion_matrix(y_test, predictedVals)
```

### Cross Validation

```python
X = data[predictors]
y = data['outcome']

kf = KFold(n_splits = 5)
lr = LogisticRegression()

acc = []
```

```python
for train_indicies, test_indicies in kf.split(X):
  
  X_train = X.iloc[train_indicies]
  X_test = X.iloc[test_indicies]
  y_train = y[train_indicies]
  y_test = y[test_indicies]
  
  z = StandardScaler()
  z.fit(X_train)
  
  Xz_train = zscore.transform(X_train)
  Xz_test = z.transform(X_test)
  
  model = lr.fit(Xz_train, y_train)
  acc.append(accuracy_score(y_test, model.predict(Xz_test)))
```

```python
print(acc)
```

```python
print(np.mean(acc))
```

```python
coef = pd.DataFrame({'Coefs': myLogit.coef_[0], 'Names':predictors})
coef = coef.append({'Coefs': myLogit.intercept_[0], 'Names':'intercept'}, ignore_index=True)
coef
```

### Odds
Odds show the exponentiated log odds provided from the logit model. Odds means that for every 1 standard deviation increase in X will increase by a factor of the odds.

Odds are usually more intuitive for stakeholders.

In this instance, an increase in opponents offensive EPA weighs more heavily than an an increase in the individual teams offensive EPA.

```python
coef['Odds Coef'] = np.exp(coef['Coefs'])
coef
```

## Adjust the probability for classification of a win

Threshold modifications won’t do anything to improve performance with this data. Depending on what you are classifying, you can adjust the risk tolerance by changing the threshold of the probabilities.

For marketing, 20% might be enough to justify sending an ad. In healthcare, a 90% probability might be necessary.

For our data, it absolutely doesn’t matter… I’ll try to build a better NFL dataset and drop in here!

```python
X_new = data.iloc[:, 0:4].copy()
Xnewz = zscore.transform(X_new)

Ypred_prob = myLogit.predict_proba(Xnewz)
Ypred_prob[1:5]
```

```python
Ypred_prob1 = Ypred_prob[:,1]
thresh = 0.75
Ypred_prob1_thresh = (Ypred_prob1 > thresh) * 1
Ypred_prob1_thresh
```

```python
accuracy_score(data['outcome'], Ypred_prob1_thresh)
```



























