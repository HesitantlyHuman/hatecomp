import os
import json

try:
    TWITTER_KEY_PATH = os.environ['TWITTER_KEYS']
except KeyError:
    TWITTER_KEY_PATH = 'hatecomp/twitter/keys.json'
    print(f"TWITTER_KEYS environment variable not set, using default location of {TWITTER_KEY_PATH}")

class TweetDownloader():
    def __init__(self, twitter_keys_path = TWITTER_KEY_PATH):
        with open(twitter_keys_path) as twitter_keys:
            self.keys = json.loads(twitter_keys.read())
    
    

if __name__ == '__main__':
    t = TweetDownloader()