import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def item_desc(df, id):
    return df.loc[df['id'] == id]['description'].tolist()[0].split(' - ')[0]

def recommend(df, results, item_id, num):
    print("\nRecommending " + str(num) + " products similar to " +
          item_desc(df, item_id) + "...")
    print("-" * 50)
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + item_desc(df, rec[1]) + " (score:" + str(rec[0]) + ")")

if __name__ == '__main__':
    items_i_like = [10, 50, 130, 450]
    num_rec = 5

    df = pd.read_csv("../data/skus_and_descriptions.csv")

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english',
                     max_features=500)
    tfidf_matrix = tf.fit_transform(df['description'])

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    results = {}

    for idx, row in df.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-num_rec-2:-1]
        similar_items = [(round(cosine_similarities[idx][i], 3), df['id'][i])
                         for i in similar_indices]
        # First item is the item itself, so remove it.
        results[row['id']] = similar_items[1:]

    for item in items_i_like:
        recommend(df, results, item_id=item, num=num_rec)
