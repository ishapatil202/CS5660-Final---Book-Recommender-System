# CS5660 Final - Book Recommender System

## Current Code

### Our code is divided into parts:
1. Download - small script to download and unpack the text files for all gutenberg books
2. Prepare ([prepare.py](prepare.py)) - This uses [gutenbergpy](https://github.com/raduangelescu/gutenbergpy) to remove the license headers to reduce noise in the text dataset.
3. Catalog ([catalog.py](catalog.py)) - This adds the metadata of authors, date issued, title, as well as categorizational attributes such as Library of Congress Categorization as well as Project Gutenberg's subjects and bookshelves for each book
4. Summarize ([summarize.py](summarize.py)) - This uses Google's X-Large Pegasus LLM fine tuned on the Booksum dataset to create summaries, in order to create more suscinct embeddings
5. Embeddings ([embedding.py](embedding.py)) - This creates embeddings for all the summaries, as well as the text based metadata, using BAAI's bge-3 LLM based on LLAMA.
6. Make Clusterable ([make_clusterable.py](make_clusterable.py)) - This flattens the embeddings, as well as other metadata into a single 2D matrix.
7. Cluster ([cluster.py](cluster.py)) - This uses a minibatch K-means clustering approach to cluster data for faster processing.
8. Recommendation ([Final.ipynb](Final.ipynb)) - This loads the data generated in the previous steps to do the actual recommendation.

### The recomendation algorithm:
Our variables: $b_s$ is the array of scored books, $B$ is the set of all books, $C$ is the set of all clusters, $k$ is the number of clusters.

Cluster selection: $\forall c \in C (score = \sum_{i=0}^{|b_s|} \text{cosine\\\_similarity}(b_{s,i}, \text{centroid}(c)))$, take the top $n * \frac{k}{|B|}$ scoring clusters' books as potential recommendations, $R_p$.

Book Selection: $\forall r_p \in R_p (score = \sum_{i=0}^{|b_s|} \text{cosine\\\_similarity}(b_{s,i}, \text{embedding}(b)))$, take the top $n$ scoring books as recommendations.

#### Algorithmic Complexity
A naive approach would compare every scored book to every book in the dataset, thus being $O(|b_s| * |B|)$.

Our clustered approach divides this into $O(|b_s| * k + \frac{|B| * |b_s|}{k})$, and using $k=\sqrt{|b_s|}$ we achieve an algorithmic complexity of $O(|b_s| * \sqrt{|b_s|} + |B| * \sqrt{|b_s|}) = O(\sqrt{b_s} * (|b_s| + |B|))$

## Potential Extras
- [] Adding dimentionality reduction with HDBScan and adding a bonus to books in the same HDBScan clusters - this would likely reward books in the same genres.
- [] Adding dimentionality reduction with KMeans and similarity - could boost processing, though would likely break cosine\_similarity.