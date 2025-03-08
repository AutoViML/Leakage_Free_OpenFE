<div id="top" align="center">

![new_logo](images/leakage_free_openfe.png)

Leakage Free OpenFE: An efficient automated feature generation tool without data leakage from the original
-----------------------------
<h3> |<a href="https://arxiv.org/abs/2211.12507"> Original Paper with Data Leakage </a> | 
<a href="https://openfe-document.readthedocs.io/en/latest/"> Original Documentation </a> | 
<a href="https://github.com/IIIS-Li-Group/OpenFE/tree/master/examples"> Original Examples </a> | <a href="https://github.com/IIIS-Li-Group/OpenFE"> Original Website </a> </h3>

</div>

# Original Claims from OpenFE Paper
quote:

<ul>
<li>OpenFE is a new framework for automated feature generation for tabular data. </li>
<li>OpenFE is easy-to-use, effective, and efficient with following advantages:</li>
- OpenFE can discover effective candidate features for improving the learning performance of both GBDT and neural networks.<br>
- OpenFE is efficient and supports parallel computing.<br>
- OpenFE covers 23 useful and effective operators for generating candidate features.<br>
- OpenFE supports binary-classification, multi-classification, and regression tasks.<br>
- OpenFE can automatically handle missing values and categorical features.<br>
</ul>

unquote

## In addition, the Authors claimed:
quote
<ul>
<li>Extensive comparison experiments on public datasets show that OpenFE outperforms existing feature generation methods on both effectiveness and efficiency.</li>
<li>Moreover, we validate OpenFE on the <a href="https://www.kaggle.com/competitions/ieee-fraud-detection">IEEE-CIS Fraud Detection</a></li>
<li>Kaggle competition, and show that a simple XGBoost model with features generated by OpenFE <b>beats 99.3% of 6351 data science teams</b>.</li> 
<li>The features generated by OpenFE results in <b>larger performance improvement than the features provided by the first-place team in the competition.</b></li>
</ul>
unquote

## and their paper was accepted in prestigious ML conferences such as ICML
quote
<ul>
[**2023-04-26**]: OpenFE has been accepted by ICML2023!
</ul>
unquote

# 🔥 However we found that the library's ability to beat 99% of data scientists will not work on real world datasets due to its "data leakage" flaw
- [**2025-02-25**]: The original code and datasets to reproduce the results in their paper are available at [OpenFE_reproduce](https://github.com/ZhangTP1996/OpenFE_reproduce). However, there appears to be a major flaw in their approach. This was discovered in Feb 2025 as I perused their web site and tried to "test" the library on the canonical "Titanic" dataset. Since I knew every one of the tricks employed to outperform in this very famous Kaggle competition and dataset, I was surprised to see a big lift in "Titanic" performance upon using "OpenFE" which added about 50 features. However upon examination the features seemed very similar to old approaches that been tried and discarded by most Kagglers. So I wondered how could the same features provide a huge performance lift in this test? That's when I decided to dive deeper and study their code. 

At the first outset, I found this very "curious" code snippet in the README file:

```
train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs) 
```

I wondered why one would need to send both train and test datasets together to "transform" in the same line of code? I deiced to study the "transform" function embedded in `utils.py` as well as in `openfe.py` files (it was the same code in both places). That's where I found the huge "data leakage" flaw in the OpenFE code and paper. Let me explain it here.

# Data Leakage Mechanism in OpenFE code

1. **The user calls the transform() function in utils.py**
The user sends the train and test data to `transform` for creating new features based on the fit function earlier.
```python
    train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs) 
```

2. **The transform() function combines train and test into one `data` variable in `utils.py` :**
The `transform` function combines train and test datasets into one `data` variable and writes it to a temporary feather file.
```python
    data = pd.concat([X_train, X_test], axis=0)
    data.index.name = 'openfe_index'
    data.reset_index().to_feather(self.tmp_save_path)
    n_train = len(X_train)
    ...
    for feature in new_features_list:
        results.append(ex.submit(_cal, feature, n_train))
```
3. **The transform() function then calls the `_cal() method` in `openfe.py`**
The `_cal` method picks up this data from feather.
```python
    _data = pd.read_feather('./openfe_tmp_data.feather', columns=base_features).set_index('openfe_index')
```
**Problem**: openfe_tmp_data.feather contains both train and test data (from pd.concat([X_train, X_test]) in transform).


4. **Feature Calculation on combined data in `openfe.py`** 
The `_cal()` method then calls `_calculate()` to create new features using the combined train and test data:
```python
    feature.calculate(_data, is_root=True)
```
**Critical Point**: The calculate() method operates on the entire dataset (train + test). For operators like GroupByThenMean, this means: Group statistics (e.g., mean, std) are computed using both train and test data.

For Example: GroupByThenMean(income, zipcode) calculates zipcode-level means from the full dataset, not just the training portion.

5. **Data Leakage mechanism is hidden**
Once features are created using combined train and test data, the Data Leakage Mechanism is hidden because the combined data is split back into train and test in `_trans()` method in `openfe.py`.
```python
    return ((str(feature.data.dtype) == 'category') or (str(feature.data.dtype) == 'object')), \
            feature.data.values.ravel()[:n_train], \
            feature.data.values.ravel()[n_train:], \
            tree_to_formula(feature)
```
**Critical Point**: This means that training features now contain values influenced by test data group statistics such as mean, median, min, max, etc. These "training" features are "performance boosting" for Kaggle competitions but are invalid for real-world use and production ML runs.

# What is wrong with their original approach?

- OpenFE’s transform() method explicitly merges train/test data before calculating global summary statistics which results in leakage of test data

- Features requiring global statistics (e.g., group means) leak very important "signals" about test data into training data.

**Critical Point**: This artificially "inflates" test performance metrics when model is predicted on test data since the model has already seen the "statistical distribution of test data" during training. This violates ML best practices and probably Kaggle competition rules too.

# Why This Matters

**Kaggle competition winning context**: Their Kaggle competition win was probably obtained by leaking of test data statistics into training data features, violating competition rules (even if implicitly allowed by static test sets).

**Unfair Advantage**: OpenFE’s Kaggle success likely stems from exploiting data leakage, not from superior feature engineering as they claimed.

It now becomes abundantly clear for anyone reviewing their code from their original paper (accepted by ICML2023) during [**2023-04-26**] and published later in[**2023-06-25**] that their claim of "openfe defeating 99% of data scientists in Kaggle competition" was based on flawed data and experiments from their "data leakage code". Their code reveals what is commonly known as "data leakage" in data science. 

**Unusable models in Real-World ML**: In real world production, test data (future data) is rarely the same as "test" data fed to models during training. Hence, Models trained this way will fail catastrophically during testing.

**Features Unavailable during Training**: Features derived this way will be unusable in real world ML where test data is rarely "seen" or is "never available" during training time.

**Question to ICML'23 committee**: How could an influential conference such as ICML'23 not have trained data scientists in their reviewing and selection committee to catch this data leakage flaw in their approach?

Any committee that had done a cursory review of OpenFE's published usage of the transform function in their README file (using both train and test data) would have had alarm bells ringing in their selection process.

## Most Kaggle Competitions prevent using test data to influence training. OpenFE circumvents this by:

- Deriving feature statistics (e.g., group means) from the full dataset.

- Training models on features that encode test set distributions.

## In addition, Real-World production runs rarely see test data (future data) before or during training. 

So features computed by OpenFE cannot be replicated in the real world, since they will cause disastrous failures in ML models during "production" runs on unseen test data.

In the spirit of open source collaboration, I decided to remove the inherent limitations of OpenFE's approach by starting from first principles of ML.

# Here is how to fix OpenFE's data leakage problem:

We must:
- Separate feature transform calculations for train and test

- Compute statistics (e.g., group means) during fit() that are based on training data only.

- Apply these precomputed values to test data during transform().

- Remove Global Operators: Drop operators like GroupByThenMean that inherently require combined data.

- Add Leakage Checks: Validate that no test data is used during training-phase feature engineering.

# 🏴󠁶󠁵󠁭󠁡󠁰󠁿 How I created a new OpenFE without data leakage

In the spirit of open source collaboration where we take the best ideas and approach from everyone and make it available for free, I thought that even though openfe had a "data leakage" problem, it's original purpose of creating a vast array of features and then using a powerful estimator like LightGBM to reduce/select features was a "noble purpose" and "ideal" that is very close to my heart (see my other open source libraries <a href="https://github.com/AutoViML/featurewiz">featurewiz</a> and <a href="https://github.com/AutoViML/featurewiz_polars">featurewiz_polars</a>). So I decided to modify their original offending transform() function and make it do the following:

<li><b>Verification of No Leakage:</b></li>

- Group-by features in test data will use only Training data group means (and not combined "train-test means")

<li><b>Training data global mean for unseen categories</b></li>

- No combined dataset processing

<li><b>Test data statistics never influence training features</b></li>

I created this library and decided to name it "leakage-free-openfe" to show that this implementation maintains OpenFE's original goal and functionality while adding proper "data isolation" between train/test sets. The original "openfe" feature generation logic is preserved, but critical aggregation operations now use proper training-data-only statistics when transforming test data. I hope this will be embraced by the open source community.

## Installation of the new "leakage-free-openfe"

It is recommended to use **git install** for installation.

```
pip install git+https://github.com/AutoViML/Leakage_Free_OpenFE
```

Please do not use **conda or pip install** for installation since I have not created a PyPi installation for this package yet. Once I receive some feedback from others, I will release another python package different from openfe.

## ⚡️ A Quick Example of the Leakage-Free-OpenFE

It only takes five lines of codes to generate features by Leakage-Free-OpenFE. First, we generate features by Leakage-Free-OpenFE.
Next, we augment the train and test data by the generated features.

```
from leakage-free-openfe import Leakage-Free-OpenFE
leak_free = Leakage-Free-OpenFE()

# generate new features
features = leak_free.fit(data=X_train, label=y_train, n_jobs=n_jobs)  

# transform the train and test data according to generated features.
X_train = transform(X_train, features, n_jobs=n_jobs) 

X_test = transform(X_test, features, n_jobs=n_jobs) 
```

Everything else in the code from OpenFE is preserved. 

I thank the original creators of OpenFE since I assume they embarked on this journey with the best of intentions to create a unique approach to feature generations. However their approach had a hidden flaw all too common in Kaggle competitions. Hence, I wanted to continue their journey but without the fatal flaw which would put real world ML projects at risk. I hope they will contribute to this attempt to make improvements to their library. Let's hope for the best!