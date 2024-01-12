# https://github.com/twitterdev/Twitter-API-v2-sample-code

import tweepy

twitter_Client_ID = 'b3Zfbi05T2hpZEl2ZkJ1VU1mX2M6MTpjaQ'

twitter_Client_Secret = 'dSnq2SmhHboTd-x1wrzpcGOcsKRMsoPp3EuRUR4jdn_4nSOvkK'

twitter_Consumer_Key = 'BtFWfrtfFKHUz9ZxjMjHIfolZ'

twitter_Consumer_Secret = 'ZjmiAtlnvxwSIYraVlE3YQ046zg1asFLoigD5MiasTNCO0Dws2'

twitter_Access_Token = '2249935225-rOZInLXuIlkk1lcDHaAX9KhekcNChnuBoRInAFm'

twitter_Token_Secret = '6i9rsCXbkUMMQR51TPviMqhD87VNNBaMaCjUes4HQeV2J'

twiter_API_Bearer_Token = 'AAAAAAAAAAAAAAAAAAAAAPFkrwEAAAAAH%2F5vVRRALtcr2BSZgB2%2BcokSIiE%3DHmGY0j2OsI7ZszkNyDfNKK0m1jl7dqZYh11dKxijgVllAymv1Q'

action = None

###################################### Outh 1.0 disable ############################################################################################

# # update action
# action = 'home_1.0'

# # Creating the authentication object
# auth = tweepy.OAuthHandler(twitter_Consumer_Key, twitter_Consumer_Secret)

# # Setting your access token and secret
# auth.set_access_token(twitter_Access_Token, twitter_Token_Secret)

# # Creating the API object while passing in auth information
# api = tweepy.API(auth, wait_on_rate_limit=True)


####################################################################################################################################################

###################################### Outh 2.0 Enable  ############################################################################################

# # update action
action = 'home_2.0'

api = tweepy.Client(
   bearer_token=twiter_API_Bearer_Token,
   consumer_key=twitter_Consumer_Key,
   consumer_secret=twitter_Consumer_Secret,
   access_token=twitter_Access_Token,
   access_token_secret=twitter_Token_Secret,
   wait_on_rate_limit=True
)

####################################################################################################################################################

######################### Display user with get ####################################################################################################
# The Twitter user who we want to get tweets from

# # update action
action = 'user'

name = "nytimes"
# Number of tweets to pull
tweetCount = 20
####################################################################################################################################################

######################### search user with query ###################################################################################################
# The search term you want to find

# # update action
action = 'search'
# query with key pair value
query = "Toptal"
# Language code (follows ISO 639-1 standards)
language = "en"

####################################################################################################################################################

#Using the API object to get tweets from your timeline, and storing it in a variable called public_tweets
public_tweets = None

if action == 'home_2.0':
    public_tweets = api.get_home_timeline()
elif action == 'home_1.0':
    public_tweets = api.home_timeline()
elif action == 'user':
    public_tweets = api.user_timeline(
        id=name, 
        count=tweetCount
    )
elif action == 'search':
    public_tweets = api.search(
        q=query, 
        lang=language
    )


# foreach through all tweets pulled
for tweet in public_tweets:
    # printing the text stored inside the tweet object
    print(tweet.text)
    print(tweet.created_at)
    print(tweet.user.screen_name)
    print(tweet.user.location)