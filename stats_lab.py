import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.normal(0, 1, n)
    plt.hist(data, bins=10)
    plt.title('Histogram of Normally Distributed Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')  
    plt.show()

    return data

def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data= np.random.uniform(0, 10, n)
    plt.hist(data, bins=10)
    plt.title('Histogram of Uniformly Distributed Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    return data



def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data= np.random.binomial(1, 0.5, n)
    plt.hist(data, bins=10)
    plt.title('Histogram of Bernoulli Distributed Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    return data
    


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    data= np.array(data)
    n= len(data)
    x= np.sum(data)/n
    return x



def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    data = np.array(data)
    n = len(data)
    mean = sample_mean(data)
    variance = np.sum((data - mean) ** 2) / (n - 1)
    return variance


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - 25th percentile (Q1)
    - 75th percentile (Q3)

    Use a consistent quartile definition. The tests for the fixed
    dataset [5,1,3,2,4] expect Q1=2 and Q3=4.
    """
    data=np.array(data)
    sorted_data = np.sort(data)
    n = len(sorted_data)  
    #minmax  
    min_val = sorted_data[0]
    max_val = sorted_data[-1]
    #median
    if n % 2 == 0:
        median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        lower_half = sorted_data[:n//2]
        upper_half = sorted_data[n//2:]
    else:
        median = sorted_data[n//2]
        lower_half = sorted_data[:n//2]
        upper_half = sorted_data[n//2 + 1:]
    q1 = sorted_data[int(np.ceil(n/4)) - 1]
    q3 = sorted_data[int(np.ceil(3*n/4)) - 1]

    return min_val, max_val, median, q1, q3 

# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    mean_x = sample_mean(x)
    mean_y = sample_mean(y)
    covariance = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    return covariance 


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)
    return np.array([[var_x, cov_xy], [cov_xy, var_y]])
