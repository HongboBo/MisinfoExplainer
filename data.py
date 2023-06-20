from mumin import MuminDataset

twitter_bearer_token='AAAAAAAAAAAAAAAAAAAAADLZbAEAAAAA3E3ttnsNl%2FKyDP%2B29JTRSciTY7I%3DVoh9ZGq89R3ctrKwt8nYqW2DsOIWG49WVO1i13XgUlmRlK8mIY'

def load_mumin_graph(size: str = 'small'):
    dataset = MuminDataset(twitter_bearer_token=twitter_bearer_token, size=size)
    dataset.compile()
    dataset.add_embeddings()
    return dataset