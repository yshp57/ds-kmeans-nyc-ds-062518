
# Unsupervised Learning, Clustering and Kmeans

Thus far we have investigated supervised learning techniques; with these, we have been creating mathematical mappings between two spaces, X and y. Unsupervised learning is different by nature in that we are looking to uncover patterns and structures within the overall data itself rather then mappings between two sets of variables for predictive modelling. A classic and easy to understand unsupervised learning technique is the K-means algorithm. This algorithm takes a set of data and groups the individaul datapoints based on which points are the most similar to one another. Clustering techniques like this can be used for a variety of application purposes such as grouping products, people, places, pictures or time periods together.

# Kmeans via scikit learn
With that lets look at how to implement the Kmeans algorithm from a practitioner's point of view.


```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import string
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
#Load in some sample data from sklearn
digits = load_digits()
df = pd.DataFrame(digits.data)
print('Length of dataset:', len(df))
df.head()
```

    Length of dataset: 1797





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
      <th>61</th>
      <th>62</th>
      <th>63</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>9.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>...</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 64 columns</p>
</div>




```python
print('8x8 Pixel Grayscale Handwritten Digits preview')
for idx, image in enumerate(digits.images[:10]):
    plt.subplot(2, 5, idx+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
```

    8x8 Pixel Grayscale Handwritten Digits preview



![png](index_files/index_4_1.png)



```python
kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
kmeans.fit(df)
df['Cluster'] = kmeans.predict(df)
#Change Cluster Labels to Letters to Avoid Confusion
num_to_letters = dict(zip(list(range(10)), list(string.ascii_lowercase)[:10]))
df['Cluster'] = df['Cluster'].map(num_to_letters)
print(df.Cluster.value_counts())
df.head()
```

    i    370
    g    194
    h    181
    e    179
    b    175
    f    166
    c    164
    a    150
    j    121
    d     97
    Name: Cluster, dtype: int64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
      <th>61</th>
      <th>62</th>
      <th>63</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>e</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>j</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>i</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>



## Cluster Sizes
Notice above that the various clusters are not equally sized. This is normal behavior; groupings are made to be homogenous, not equallly distributed.


```python
df['Number'] = digits.target
temp = pd.DataFrame(df.groupby('Cluster')['Number'].value_counts(normalize=True))
temp.columns= ['Percent of Cluster']
temp = temp.reset_index()
temp = temp.pivot(index='Cluster', columns='Number', values='Percent of Cluster')
temp = temp.sort_index(ascending=False)

ax = temp.plot(kind='barh', figsize=(10,6))
plt.title('Actual Image Digit by Cluster')
ax.legend(bbox_to_anchor=(1.15,1))

rects = ax.patches

labels = temp.columns
# For each bar: Place a label
for n, rect in enumerate(rects):
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2
    
    label = labels[n//10]
    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'
    try:
        percent = int(round(temp.iloc[n%10][n//10],4)*10000)/100
    except:
        continue
    final_label = 'Digit {},\n {}%'.format(label, percent)
    if x_value > 0.5:
        # Create annotation
        plt.annotate(
            final_label,                # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(space, 0),          # Horizontally shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            va='center',                # Vertically center label
            ha=ha)                      # Horizontally align label differently for
                                        # positive and negative values.

# plt.savefig("image.png")
```


![png](index_files/index_7_0.png)



```python
temp.fillna(value=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Number</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>j</th>
      <td>0.000000</td>
      <td>0.834711</td>
      <td>0.008264</td>
      <td>0.000000</td>
      <td>0.041322</td>
      <td>0.000000</td>
      <td>0.024793</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>i</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.032432</td>
      <td>0.435135</td>
      <td>0.000000</td>
      <td>0.108108</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.032432</td>
      <td>0.391892</td>
    </tr>
    <tr>
      <th>h</th>
      <td>0.000000</td>
      <td>0.011050</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011050</td>
      <td>0.972376</td>
      <td>0.000000</td>
      <td>0.005525</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>g</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.005155</td>
      <td>0.041237</td>
      <td>0.041237</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.860825</td>
      <td>0.010309</td>
      <td>0.041237</td>
    </tr>
    <tr>
      <th>f</th>
      <td>0.006024</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.981928</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>e</th>
      <td>0.988827</td>
      <td>0.000000</td>
      <td>0.005587</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.005587</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.000000</td>
      <td>0.546392</td>
      <td>0.030928</td>
      <td>0.000000</td>
      <td>0.030928</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.103093</td>
      <td>0.082474</td>
      <td>0.206186</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.079268</td>
      <td>0.067073</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.006098</td>
      <td>0.012195</td>
      <td>0.817073</td>
      <td>0.006098</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.834286</td>
      <td>0.005714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017143</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>a</th>
      <td>0.000000</td>
      <td>0.006667</td>
      <td>0.000000</td>
      <td>0.013333</td>
      <td>0.000000</td>
      <td>0.920000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.040000</td>
    </tr>
  </tbody>
</table>
</div>



If we already new the desired labels (as in this case), then applying our previous knowledge of classification algorithms would be more appropriate. (We can also see that these clusters have not created any class where 3 is prominent.) However, if we don't know what labels or groupings to apply to the data beforehand, unsupervised clustering methods such as this can help us bin cases into homogeneous groupings.

# Understanding the algorithm

Let's talk about the Kmeans algorithm in a bit more depth. In particular, we will examine 4 important initialization parameters that influence the results of the algorithm. Those five are as follows.

* Number of Clusters
* Initialization
* Precision
* Max Iterations

The way Kmeans starts is by first selecting n-starting points to act as the initial group centers. From there, the distance from each observation point is calculated from each of the cluster centers. The point is then assigned the cluster center to which it is closest. Once this has been done for all points, the centroid is then calculated for all those points assigned to a certain group. This then becomes the new group center points and the process is repeated; most points will remain in the same group, but certain edge points will shift from one group to another as the cluster centers themselves change. (Remember our initial cluster points were arbitrary or random, as we didn't know much about the data.) This process continues until either the max number of iterations is reached, or no points shift between groups (this leads to a steady state of the algorithm converging). Alternatively, a precision parameter could be passed specifying algorithm termination if a certain number of points were to not change cluster groupings.

Due to the iterative nature of the Kmeans algorithm, our initialization points are extremely important as to the final cluster results that will be returned. For this reason, Kmeans is often run multiple times on a dataset using different initialization points and the resulting clusters are then compared by how tightly grouped the resulting clusters are. Afterwards, the clusters with the minimal average variance within a cluster are selected as the optimal grouping.

# Practice and Compare

Look at the doc string for the sklearn KMeans method. As we've discussed above, the initialization parameter is extremely important. By default, we have used an intelligent initial guess for cluster centers, which is outside the scope of our current discussion. Try changing this parameter to 'random' as described in the docstring. Use this method to perform 4 different rounds of clustering and plot the distribution of the actual digit images (like we did above) on 4 subplots to compare them.


```python
#Your code here
```

# Normalization
When calculating the distance between points, the scale of our various features is extremely important. As with linear regression and other algorithms, if one feature is on a larger scale then other features, it will play a disproportionate weight on the convergence of the algorithm. For this reason, normalizing all features to a single scale (or intentionally weighting certain features which you believe to be more meaningful or impactful) is an important preprocessing step to the Kmeans algorithm.

Fortunately, our data has already been normalized, but this is an important fact to remember. 

# Choosing Appropriate K
Another question that naturally arises when discussing the Kmeans algorithm is how to choose an optimal value for K, the number of clusters. Sometimes this can be chosen by practical application purposes, such as wanting 3-5 audience or product groupings. At other times, there might be no obvious answer and as such a hearustic measurement is needed. One measurement which can further shed light on the homogeneity of groupings is inter-cluster variance.

This is easily implemented with the inertia method built into the KMeans instance used to cluster the data. The inertia of clustering is the sum of the distances between points and their groups centroid. At one extreme, this will be greatest when we have one group, and at the other extreme, this value will be zero if we have as many groups as we have data points, since each point would form its own group and the distance bewteen something and itself is zero.


```python
#Retrieving the inertia (total sum of point distances from their cluster center)
kmeans.inertia_
```




    1172470.2722752562



# Graphing Inertia for Various Values of K

Iterate over the range 1 to 100, training a new Kmeans algorithm with that amount of clusters. For each, retrieve the inertia of the resulting clusters. Then plot this data on a graph. The x-axis will be the number of clusters (from 1 to 100) and the y-axis will be the inertia associated with the clusters produced from that value of K.


```python
#Starter Code
for i in range(1,101):
    #Cluster using Kmeans
    #Calculate Inertia
    #Store Data
    #Plot
```

This forms the basis for the 'elbow' method. As you should see, as the number of clusters increases, the inertia decreases. Typically, we search for an 'elbow' or corner where the rate at which the inertia decreases levels off. In this case, an appropriate number of clusters would be 10; after all, the data is associated with pictures of the digits 0-9.
