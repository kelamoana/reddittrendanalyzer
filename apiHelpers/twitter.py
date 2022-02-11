import tweepy

# Your app's bearer token can be found under the Authentication Tokens section
# of the Keys and Tokens tab of your app, under the
# Twitter Developer Portal Projects & Apps page at
# https://developer.twitter.com/en/portal/projects-and-apps
bearer_token = "AAAAAAAAAAAAAAAAAAAAAERoZAEAAAAA%2FfL%2BhjXnWcr9uLId27U%2BQTXJYuo%3DRxZnPXcOCTXiP5Ik4NUr4cBY4J8qnjeNPaphUxsVaXMA5lUELG"

# Your app's API/consumer key and secret can be found under the Consumer Keys
# section of the Keys and Tokens tab of your app, under the
# Twitter Developer Portal Projects & Apps page at
# https://developer.twitter.com/en/portal/projects-and-apps
consumer_key = "1tvw4bXrjWa2Zd1Hy1D780jGa"
consumer_secret = "bm2JxmHHBoZK2bqu3TukHbsrTT5rLCX56CPkkZ5r2zbmuImAZw"

# Your account's (the app owner's account's) access token and secret for your
# app can be found under the Authentication Tokens section of the
# Keys and Tokens tab of your app, under the
# Twitter Developer Portal Projects & Apps page at
# https://developer.twitter.com/en/portal/projects-and-apps
access_token = "1051683853176168448-ch9xPUTgzvm1g9QSpzdjq5dpM2qA2C"
access_token_secret = "NOLEsWQDzcKWrtN5L6qIGkz5mpmNFWWNJ9VJr63kctkRK"

def get_tweets(username):
          
        # Authorization to consumer key and consumer secret
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  
        # Access to user's access key and access secret
        auth.set_access_token(access_token, access_token_secret)
  
        # Calling api
        api = tweepy.API(auth)
  
        # 200 tweets to be extracted
        number_of_tweets=200
        tweets = api.user_timeline(screen_name=username)
  
        # Empty Array
        tmp=[] 
  
        # create array of tweet information: username, 
        # tweet id, date/time, text
        tweets_for_csv = [tweet.text for tweet in tweets] # CSV file created 
        for j in tweets_for_csv:
  
            # Appending tweets to the empty array tmp
            tmp.append(j) 
  
        # Printing the tweets
        print(tmp)
  
  
# Driver code
if __name__ == '__main__':
  
    # Here goes the twitter handle for the user
    # whose tweets are to be extracted.
    get_tweets("ucirvine") 