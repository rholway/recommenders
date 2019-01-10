# Recommenders

We were tasked with creating recommender systems for a couple of our assignments.  Some details on the assignments are below.

## Content Based Recommender

**Content Based Recommender** - in content-based recommenders, item attributes and user-item preferences are used to recommend similar items.

For this assignment, we were asked to make recommendations of products that were similar to a specified product.

We received a data frame with a product id as one attribute, and a product name/description as the other attribute.  There were approximately 500 products in the data frame.  Here is an example of one row of the data frame:

id  | description
------------- | -------------
10  | 'Baby sun bucket hat - This hat goes on when the sun rises above the horizon, and stays on when raindrops start falling...'

I made a term frequency-inverse document frequency (tf-idf) matrix to represent the product name/descriptions numerically.  From the tf-idf matrix, I was able to use cosine similarity to compare  products to one another.  Products with the highest cosine similarity to the specific product were recommended.  The output result looked like this:

```python
Recommending 5 products similar to Baby sun bucket hat...            

Recommended: Trim brim hat (score:0.77)                             
Recommended: Surf brim (score:0.639)                                
Recommended: Bucket hat (score:0.61)                                
Recommended: Beach bucket (score:0.567)                             
Recommended: Baby baggies shorts (score:0.528)
```

From this recommender, we can specify any product in the data frame, and find the *n* most similar products (in this case n=5)

## Collaborative Filtering Recommender

**Collaborative filtering recommender** - in collaborative filtering recommenders, user-item interactions (like movie ratings) are used to identify similarities between users and ratings to recommend items.

For this assignment, we were asked to find movie recommendations for a user based on how that user rated other movies.

We received a data frame with approximately 100,000 observations of users, movies, and ratings.  Here is a snippet of the data frame:

user  | movie | rating
----- | ----- | -----
196  | 242  | 3
186  | 302  | 3

The same user can be repeated in the 'user' column, because that user may have rated more than one movie.  Likewise, the same movie can be repeated in the 'movie' column because that movie may have been rated by more than one user.  The 'ratings' are between 1-5.

By representing the table as a matrix and utilizing cosine similarity and neighborhood methods, we can find out what users are similar to each other. Based on what users are similar to each other, we can recommend certain movies to one user, based on the movies other users (similar to that user) have rated highly. The output result looked like this:

```python
The top 10 recommended movies are:
[1285, 226, 228, 229, 153, 186, 213, 198, 1425, 233]
```

Unfortunately, we did not have the actual movie titles that the numbers correspond to.  If we did have the actual movie titles, then we could map the titles to their associated number, and get the top 10 recommended movies for a user, just based on how they rated other movies.
