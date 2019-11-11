# Goodreads_Book_Recommender
Book Recommendation System Using Neural Network Embeddings

## Motivation
Recommendation systems have become increasingly popular to help business succeed. (e.g. 40% of app installs on Google Play come from recommendations, 60% of watch time on YouTube comes from recommendation). Recommendation systems help users find compelling content in a large corpora to increase the users stickness to the business. In addition to the business value of recommendation system, personal interest is the main reason I picked up the topic. I like reading books with applications in my phone and watching Youtube video. There are always some good recommendations that I can't help to check out. I'm curious how the systems 'read my mind' and give me recommendations I like.

## Objective
The objective of this project is to build an effective book recommendation system by learning embeddings of books and users using a neural network. 

## Data
Data from Goodreads about mystery, thriller and crime books
* Raw data: 

    goodreads reviews - 1,849,236 reviews with features of user id, book id, review id, rating, review text, date added; 
    
    meta book data - 219,235 books with features of book id, title, text reviews count, country code, language code, genre, description, format, authors, publisher, ratings count, etc
* Cleaned data: 

    reviews excluding users or books only have one rating - 1,551,765 reviews;  
    
    meta book data - 105,365 books

reference: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home?authuser=0


## Data Analysis


Below is the data struction:
<p align="center">
  <image src=Visualization/.png />
</p>

## Models
### Baseline model - average rating adjusted by user/book deviation
### Collaborative filtering model - matrix factorization

The algorithm applied is alternative least squares. By using embedding, we can uncover the latent features for both users and books and predict the rate by a given user to a given book. We used embeddings rather than the method of singular value decomposition. Since the dataset is very sparse, using embeddings with tensorflow sparse tensor will improve the computation efficiency. (SVD is based on algebra calculation to get the eighenvalues and eighen vectors, which is expensive)
### Hybrid model - neural network with user_id, book_id and book_feature (e.g. author_id) embeddings

Process: 
1. for a given user and book pair, create lists for his/her previously liked (rating is greater than 3) and disliked (rating is less than 4) book_ids;
2. create book_id embeddings for the book lists and create book feature embedding (we only used author_id here) for the given book;
3. use the embeddings created above as input layer
4. create a certain number of hidden layers
5. the output of the last hidden layers is the user's embedding
6. multiple the user embedding and learned book embedding

With the implementation, we can predict the rate for this given user and book pair. 




## Reference
* Data source: Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19 https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home?authuser=0
* Google recommendation tutorial: https://developers.google.com/machine-learning/recommendation/overview
* Book recommendation algorithm for the company Instafreebie https://github.com/declausen/capstone_project
* Neural network embeddings explained https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526
* Building a recommendation system using neural network embeddings https://towardsdatascience.com/building-a-recommendation-system-using-neural-network-embeddings-1ef92e5c80c9
* Netflix recommendations: beyond the 5 stars https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-2-d9b96aa399f5


