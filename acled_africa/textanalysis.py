import pandas as pd
import numpy as np
import sklearn
import time
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import cluster
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
from wordcloud import WordCloud
import itertools
import warnings
import re
import matplotlib
from sklearn.feature_extraction.text  import TfidfVectorizer
warnings.filterwarnings('ignore')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text

stop_words_protests = text.ENGLISH_STOP_WORDS.union(['protest', 'protests', 'protesters', 'protested','demonstrate','demonstrated','demonstrators','demonstrations','demonstrator','demonstration','january', 'february', 'march','april', 'may', 'june', 'july','august','september','october','november','december'])

stop_words_battles = text.ENGLISH_STOP_WORDS.union(['january', 'february', 'march','april', 'may', 'june', 'july','august','september','october','november','december', 'killed','kill', 'attack', 'attacked', 'kills', 'killing', 'soldiers','soldier', 'gun', 'gunmen','dead', 'death', 'died'])

def max_sum_sim(doc_embedding, word_embeddings, words, top_n, nr_candidates):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, word_embeddings)
    distances_candidates = cosine_similarity(word_embeddings, 
                                            word_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [words[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]

def mmr(doc_embedding, word_embeddings, words, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


def keyword_extraction(data,event_type,n_gram_range=(1,3),keyword_n=5,mss_n=5,mss_nr=10,mmr_n=5,mmr_diversity=0.7):
    """
    Performs keyword extraction for the given country and event type
    
    Parameters
    ----------
    data : pandas dataframe that contains the following columns ['event_type', 'notes']
        
    event_type : the event type for which keywords are being extracted (important as list of stop words used changes by event type)
        
    n_gram_range : the length of keywords/phrases to be extracted
         (Default value = (1,3))
        
    keyword_n : top number of keywords to extract
         (Default value = 5)
    mss_n : top number of keywords to extract from mss algorithm
         (Default value = 5)
    mss_nr : top number of keywords/keyphrases to select from 
         (Default value = 10)
    mmr_n : top number of keywords to extract from mmr algorithm
         (Default value = 5)
    mmr_diversity : level of diversity between keywords (higher value = keywords are more different from each other)
         (Default value = 0.7)
         
    References
    -------
    Towards Data Science Article by Maarten Grootendorst: Keyword Extraction with BERT
    https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea
    

    Returns
    -------
    Dataframe with added column for keywords for each algorithm
    
    """
    if event_type == 'Protests':
            stop_word_list = stop_words_protests
    elif event_type == 'Battles':
            stop_word_list = stop_words_battles
    initial_start = time.time()
    initial_start_readable = time.ctime(int(initial_start))
    print("Start time =", initial_start_readable)
    for row in range(len(data)):
        doc = data.iloc[row]['notes']
        #Extract candidate words/phrases
        count = CountVectorizer(preprocessor=preprocess_text, ngram_range=n_gram_range, stop_words=stop_word_list).fit([doc])
        candidates = count.get_feature_names()
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        doc_embedding = model.encode([doc])
        candidate_embeddings = model.encode(candidates)
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][-keyword_n:]]
        max_sum_sim_out = max_sum_sim(doc_embedding = doc_embedding, word_embeddings=candidate_embeddings,
                                      words=candidates, top_n=mss_n, nr_candidates=mss_nr)
        mmr_out = mmr(doc_embedding=doc_embedding, word_embeddings=candidate_embeddings,
                      words=candidates, top_n=mmr_n, diversity=mmr_diversity)
        keywords = ', '.join(keywords)
        data.loc[data.index[row], 'keywords'] = keywords
        max_sum_sim_out = ', '.join(max_sum_sim_out)
        data.loc[data.index[row], 'max_sum_sim_out'] = max_sum_sim_out
        mmr_out = ', '.join(mmr_out)
        data.loc[data.index[row], 'mmr_out'] = mmr_out
    finish = time.time()
    finish_readable = time.ctime(int(finish))
    print("Total time elapsed:", (finish-initial_start), "seconds")
    print("End time =", finish_readable)



def run_KMeans(max_k, data):
    max_k += 1
    kmeans_results = dict()
    for k in range(2 , max_k):
        kmeans = cluster.KMeans(n_clusters = k
                               , init = 'k-means++'
                               , n_init = 10
                               , tol = 0.0001
                               , n_jobs = -1
                               , random_state = 1
                               , algorithm = 'full')

        kmeans_results.update( {k : kmeans.fit(data)} )
        
    return kmeans_results

def printAvg(avg_dict):
    for avg in sorted(avg_dict.keys(), reverse=True):
        print("Avg: {}\tK:{}".format(avg.round(4), avg_dict[avg]))
        
def plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg):
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(8, 6)
    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])
    
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--") # The vertical line for average silhouette score of all the values
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.title(("Silhouette analysis for K = %d" % n_clusters), fontsize=10, fontweight='bold')
    
    y_lower = 10
    sample_silhouette_values = silhouette_samples(df, kmeans_labels) # Compute the silhouette scores for each sample
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = matplotlib.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i)) # Label the silhouette plots with their cluster numbers at the middle
        y_lower = y_upper + 10  # Compute the new y_lower for next plot. 10 for the 0 samples
    plt.show()
    
        
def silhouette(kmeans_dict, df, plot=False):
    df = df.to_numpy()
    avg_dict = dict()
    for n_clusters, kmeans in kmeans_dict.items():      
        kmeans_labels = kmeans.predict(df)
        silhouette_avg = silhouette_score(df, kmeans_labels) # Average Score for all Samples
        avg_dict.update( {silhouette_avg : n_clusters} )
        printAvg(avg_dict)
        if(plot): plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg)
            
def centroidsDict(centroids, index):
    a = centroids.T[index].sort_values(ascending = False).reset_index().values
    centroid_dict = dict()

    for i in range(0, len(a)):
        centroid_dict.update( {a[i,0] : a[i,1]} )

    return centroid_dict

def generateWordClouds(centroids):
    wordcloud = WordCloud(max_font_size=100, background_color = 'white')
    for i in range(0, len(centroids)):
        centroid_dict = centroidsDict(centroids, i)        
        wordcloud.generate_from_frequencies(centroid_dict)

        plt.figure(figsize=(10,20))
        plt.title('Cluster {}'.format(i))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()

def cluster_analysis(data, country, word_type, k, plot_silhouette=True, word_cloud=True, save=True):
    """
    This function performs cluster_analysis on the extracted keywords
    
    Parameters
    ----------
    data : pandas data frame containing acled data with the following columns: 
            ['country', 'event_type'] and a column containing keywords 
        
    country : country for which analysis is being performed in the data
        
    event_type : event type for which cluster analysis is being performed
        
    word_type : column in pandas dataframe that contains keywords to perform cluster analysis on 
        
    k : max number of clusters 'k' to perform kMeans
        
    plot_silhouette :
         (Default value = True)
    word_cloud :
         (Default value = True)
    save :
         (Default value = True)
    
    References
    -------
    Medium article by Lucas de Sá: Text Clustering with K-Means
    https://medium.com/@lucasdesa/text-clustering-with-k-means-a039d84a941b
    https://github.com/lucas-de-sa/national-anthems-clustering/blob/master/Cluster_Anthems.ipynb

    Returns
    -------
    Silhouette Score plots and Word Clouds generated using kMeans cluster analysis
    """
    data = data[data['country']==country]
    data = data[word_type]
    #tf-idf analysis
    corpus = data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    tf_idf = pd.DataFrame(data = X.toarray(), columns=vectorizer.get_feature_names())
    final_df = tf_idf
    
    # Running Kmeans
    kmeans_results = run_KMeans(k, final_df)

    # Plotting Silhouette Analysis
    if plot_silhouette==True:
        silhouette(kmeans_results, final_df, plot=True)
    
    selection=input("Best k: ")
    
    while not selection.isdigit:
        selection=input("Must be an integer. Try again: ")
    
    selection = int(selection)
    kmeans = kmeans_results.get(selection)
    
    if word_cloud==True:
        centroids = pd.DataFrame(kmeans.cluster_centers_)
        centroids.columns = final_df.columns
        generateWordClouds(centroids)
    
    if save==True:
        # Assigning the cluster labels to each row of data
        labels = kmeans.labels_ 
        data['keyword_label'] = labels