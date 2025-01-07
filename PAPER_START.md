Get brazilians movies sentiments from imdb top 10 movies

# Phase 1: Model initialization

initialize base BERT model

load imdb dataset

tokenize dataset

train model

push model to hub

# Phase 2: Data collection for tuning

Crawl reddit and other sources for movie reviews

store reviews in database

# Phase 3: SFT (Supervised Fine Tuning)

streamlit app to audit reviews from database and pick the best ones to train model on them .csv

push model to hub

# Phase 4: RLHF (Reinforcement Learning with Human Feedback)

Run DBOS to connect with slackbot

The user will rate the model predictions for the movie the user input

The interactions are stored in the database as reactions

Train model with the reactions

push model to hub

# Phase 5: Application usage

Use the model to suggest movies to the user based on the user input for the brazilian public target

---

it should start conversation after a greeting from the user, then answer with the top 10 trendy movies:
1 - Movie 1
2 - Movie 2
...
after that the user choose one of the movies and receives a resume from imdb of the movie, he asked if we watched that movie, if yes he can rate if not skip that movie.

his intereactions will be saved as reactions in the database

after answer the first 5 movies, he will be presented with the movies he might like.

Run DBOS to connect with slackbot
The user will rate the model predictions for the movie the user input
The interactions are stored in the database as reactions
A worker will be activate every 10minutes to Train model with the reactions and push model to hub
