
# coding: utf-8

# In[16]:
from __future__ import print_function
import re
from nltk.corpus import stopwords
import nltk
import collections
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import entity2
import numpy as np
import rbm


# In[17]:

stemmer = nltk.stem.porter.PorterStemmer()
WORD = re.compile(r'\w+')


# In[18]:

caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


# In[25]:

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


# In[20]:

text = "An army soldier was injured in fierce gun battle with a group of infiltrating terrorists from across the Line of Control in Gali Maidan area of Sawjian sector, while a BSF jawan was injured in unprovoked firing by Pakistani rangers in Hiranagar sector along the international border in Kathua district on Friday.Identifying the injured army soldier as Launce Naik Vinod Kumar and the BSF jawan as Gurnam Singh, sources said that former got injured during an encounter with a group of terrorists who sneaked into Sawjian sector on the Indian side from the Pakistan occupied Kashmir during wee hours of Friday. The encounter was in progress, sources added.Significantly, the infiltration attempt from across the LoC in Sawjian sector of Poonch district came nearly 24 hours after half a dozen heavily armed terrorists attacked a BSF naka along the international border in Kathua district with small arms fire and rocket propelled grenades so as to cross over to the Indian side. The infiltration attempt was foiled by alert BSF personnel who retaliated killing one of them as during illumation of the area with the help of tracer bomb, terrorists fleeing back to Pakistan side were seen carrying a body with them, sources added.Meanwhile, a BSF jawan was injured as Pakistani Rangers continued resorting to mortar shelling and small arms fire on two forward Indian positions at Bobiyan in Hiranagar sector of Kathua district. Sources said that the fire from across the international border first came around 9.35 am and it continued for nearly 40 minutes.Thereafter, the Pakistani Rangers again resumed firing on Indian side around 12.15 noon, sources said, adding that it was continuing till reports last came in. The Indian side was also retaliating.Ever since, India carried out surgical strikes across the Line of Control causing sufficient damage to terrorists and those shielding them last month, Pakistan has been resorting to mortar shelling and small arms fire at one or the other place along the borders in Jammu region. It continued lobbing mortar shells, besides resorting to automatics and small arms fire along the LoC in Rajouri district’s Manjakote area of Bhimber Gali sector throughout Wednesday night.The Indian troops retaliated appropriately. There had been no casualty or damage on the Indian side. Pakistani troops have resorted to firing in Rajouri sector also this afternoon."


# In[27]:

sentences = split_into_sentences(text)

stop = set(stopwords.words('english'))


# In[ ]:




# In[33]:

def remove_stop_words(sentences) :
    tokenized_sentences = []
    for sentence in sentences :
        tokens = []
        split = sentence.lower().split()
        for word in split :
            if word not in stop :
                try :
                    stemmer.stem(word)
                    tokens.append(word)
                except :
                    tokens.append(word)
        
        tokenized_sentences.append(tokens)
    return tokenized_sentences
        


# In[34]:

tokenized_sentences = remove_stop_words(sentences)


# In[35]:

def posTagger(tokenized_sentences) :
    tagged = []
    for sentence in tokenized_sentences :
        tag = nltk.pos_tag(sentence)
        tagged.append(tag)
    return tagged


# In[36]:

tagged = posTagger(remove_stop_words(sentences))


# In[37]:

def tfIsf(tokenized_sentences):
    scores = []
    COUNTS = []
    for sentence in tokenized_sentences :
        counts = collections.Counter(sentence)
        isf = []
        score = 0
        for word in counts.keys() :
            count_word = 1
            for sen in tokenized_sentences :
                for w in sen :
                    if word == w :
                        count_word += 1
            score = score + counts[word]*math.log(count_word-1)
        scores.append(score/len(sentence))
    return scores


# In[38]:

tfIsfScore = tfIsf(tokenized_sentences)


# In[39]:

def similar(tokens_a, tokens_b) :
    #Using Jaccard similarity to calculate if two sentences are similar
    ratio = len(set(tokens_a).intersection(tokens_b))/ float(len(set(tokens_a).union(tokens_b)))
    return ratio


# In[40]:

def similarityScores(tokenized_sentences) :
    scores = []
    for sentence in tokenized_sentences :
        score = 0;
        for sen in tokenized_sentences :
            if sen != sentence :
                score += similar(sentence,sen)
        scores.append(score)
    return scores


# In[41]:

similarityScore = similarityScores(tokenized_sentences)


# In[42]:

def properNounScores(tagged) :
    scores = []
    for i in range(len(tagged)) :
        score = 0
        for j in range(len(tagged[i])) :
            if(tagged[i][j][1]== 'NNP' or tagged[i][j][1]=='NNPS') :
                score += 1
        scores.append(score/float(len(tagged[i])))
    return scores
        


# In[43]:

properNounScore = properNounScores(tagged)

def text_to_vector(text):
    words = WORD.findall(text)
    return collections.Counter(words)


# In[46]:

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

# In[44]:

def centroidSimilarity(sentences) :
    centroidIndex = tfIsfScore.index(max(tfIsfScore))
    scores = []
    for sentence in sentences :
        vec1 = text_to_vector(sentences[centroidIndex])
        vec2 = text_to_vector(sentence)
        
        score = get_cosine(vec1,vec2)
        scores.append(score)
    return scores


# In[45]:




# In[47]:

centroidSimilarityScore = centroidSimilarity(sentences)


# In[50]:

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
# In[48]:

def numericToken(tokenized_sentences):
    scores = []
    for sentence in tokenized_sentences :
        score = 0
        for word in sentence :
            if is_number(word) :
                score +=1 
        scores.append(score/float(len(sentence)))
    return scores


# In[51]:

numericTokenScore = numericToken(tokenized_sentences)

def namedEntityRecog(sentences) :
    counts = []
    for sentence in sentences :
        count = entity2.ner(sentence)
        counts.append(count)
    return counts



namedEntityRecogScore = namedEntityRecog(sentences)

# In[2]:

class RBM:
  
    def __init__(self, num_visible, num_hidden, learning_rate = 0.1):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate

        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a Gaussian distribution with mean 0 and standard deviation 0.1.
        self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)    
        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)

    def train(self, data, max_epochs = 1000):
        """
        Train the machine.

        Parameters
        ----------
        data: A matrix where each row is a training example consisting of the states of visible units.    
        """

        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis = 1)

        for epoch in range(max_epochs):      
            # Clamp to the data and sample from the hidden units. 
            # (This is the "positive CD phase", aka the reality phase.)
            pos_hidden_activations = np.dot(data, self.weights)      
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
            # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
            # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
            pos_associations = np.dot(data.T, pos_hidden_probs)

            # Reconstruct the visible units and sample again from the hidden units.
            # (This is the "negative CD phase", aka the daydreaming phase.)
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:,0] = 1 # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
            # themselves.
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # Update weights.
            self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_examples)

            error = np.sum((data - neg_visible_probs) ** 2)
            print("Epoch %s: error is %s" % (epoch, error))

    def run_visible(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get a sample of the hidden units.
    
        Parameters
        ----------
        data: A matrix where each row consists of the states of the visible units.
    
        Returns
        -------
        hidden_states: A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.
        """
    
        num_examples = data.shape[0]
    
        # Create a matrix, where each row is to be the hidden units (plus a bias unit)
        # sampled from a training example.
        hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis = 1)

        # Calculate the activations of the hidden units.
        hidden_activations = np.dot(data, self.weights)
        # Calculate the probabilities of turning the hidden units on.
        hidden_probs = self._logistic(hidden_activations)
        # Turn the hidden units on with their specified probabilities.
        hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        # Always fix the bias unit to 1.
        # hidden_states[:,0] = 1
  
        # Ignore the bias units.
        hidden_states = hidden_states[:,1:]
        return hidden_states
    
      # TODO: Remove the code duplication between this method and `run_visible`?
    def run_hidden(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of hidden units, to get a sample of the visible units.

        Parameters
        ----------
        data: A matrix where each row consists of the states of the hidden units.

        Returns
        -------
        visible_states: A matrix where each row consists of the visible units activated from the hidden
        units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the visible units (plus a bias unit)
        # sampled from a training example.
        visible_states = np.ones((num_examples, self.num_visible + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis = 1)

        # Calculate the activations of the visible units.
        visible_activations = np.dot(data, self.weights.T)
        # Calculate the probabilities of turning the visible units on.
        visible_probs = self._logistic(visible_activations)
        # Turn the visible units on with their specified probabilities.
        visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
        # Always fix the bias unit to 1.
        # visible_states[:,0] = 1

        # Ignore the bias units.
        visible_states = visible_states[:,1:]
        return visible_states
    
    def daydream(self, num_samples):
        """
        Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
        (where each step consists of updating all the hidden units, and then updating all of the visible units),
        taking a sample of the visible units at each step.
        Note that we only initialize the network *once*, so these samples are correlated.

        Returns
        -------
        samples: A matrix, where each row is a sample of the visible units produced while the network was
        daydreaming.
        """

        # Create a matrix, where each row is to be a sample of of the visible units 
        # (with an extra bias unit), initialized to all ones.
        samples = np.ones((num_samples, self.num_visible + 1))

        # Take the first sample from a uniform distribution.
        samples[0,1:] = np.random.rand(self.num_visible)

        # Start the alternating Gibbs sampling.
        # Note that we keep the hidden units binary states, but leave the
        # visible units as real probabilities. See section 3 of Hinton's
        # "A Practical Guide to Training Restricted Boltzmann Machines"
        # for more on why.
        for i in range(1, num_samples):
            visible = samples[i-1,:]

            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.weights)      
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self._logistic(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i,:] = visible_states

        # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:,1:]        
      
    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))

#if __name__ == '__main__':
  



# In[53]:

featureMatrix = []
featureMatrix.append(tfIsfScore)
featureMatrix.append(similarityScore)
featureMatrix.append(properNounScore)
featureMatrix.append(centroidSimilarityScore)
featureMatrix.append(numericTokenScore)
featureMatrix.append(namedEntityRecogScore)


# In[60]:

featureMat = np.zeros((len(sentences),6))
for i in range(6) :
    for j in range(len(sentences)):
        featureMat[j][i] = featureMatrix[i][j]


# In[61]:

print(featureMat)
featureMat_normed = featureMat / featureMat.max(axis=0)
print(featureMat_normed)
for i in range(len(sentences)):
    print(featureMat_normed[i])
# In[65]:

r = RBM(num_visible = 6, num_hidden = 6)
#training_data = np.array([[0.5,0.9,0.7,0,0.8],[0.3,0,1,0.4,0],[0.8,1,1,0,0.6],[0,0.5,1,1,1], [0,0,1,1,0],[0,0,1,1,0.6]])
training_data = featureMat_normed
r.train(training_data, max_epochs = 5000)
print(r.weights)
enhanced_featureMat = r.run_visible(featureMat_normed)
print(enhanced_featureMat)


for i in range(len(sentences)):
    user = np.array([featureMat_normed[i]])
    enhanced_feature = r.run_visible(user)
    print(enhanced_feature)


rbm.test_rbm(dataset = featureMat_normed,learning_rate=0.1, training_epochs=10, batch_size=4,n_chains=4,
             n_hidden=6)

# In[54]:

#featureMatrix


# In[ ]:




# In[ ]:



