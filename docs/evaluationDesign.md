# Serveral ways for evaluation

## 1. Baseline approach

This could be the foundation for many of the other approaches. It is also a good baseline for comparison with other approaches.

### 1.1 Description

Baseline is calculated on both average and median, and a standard deviation for each of the following category(Monthly average/median/std of the last 12 months or from the beginning of the data to the current month):

- Type of building(tp) in the city: Detached, Semi-detached, Townhouse, Apartment
- Features for each type of building in the city: [Feature list](#feature-list)

**Note:** Numbers to save: mean, std, count

A delta(price difference) is calculated for each of the following:

- Each feature value to the feature baseline on the type of building in city level: [Feature list](#feature-list)

**Note:** Numbers to save: delta, std, count

Every community, type of building will be assigned a numeric value based on its delta on thousand dollars. For features with discrete values, feature value will be mapped to a new numeric value based on its delta on thousand dollars. The new numeric value will be used to replace the original value in the data.
For example, if a feature value is 2 and average value is 1.5, the average property value is 800k, the feature value of 2 property value is 850k, then the delta for value 2 is 5 thousand dollars. The ratio will be the average of all the delta divided by all value difference from the average, if the descrete feature value is a number. Otherwise, there is no ratio for the feature.

For features with continues values, a mapping ratio will be calculated based on the delta of the feature value to the feature baseline.
For example, if the delta of a feature value is 5200 and average value is 4000, the feature delta for value 5200 is 30 thousand dollars, the mapping ratio will be 30/(5200-4000) = 30/1200 = 0.025. The new numeric value will be (5200-4000)\*0.025 = 30.

A standard deviation is calculated for each of the mapping or ratio for each feature. The mapping or ratio will be used to calculate the new numeric value for the test set. The new numeric value will be either map(original value) or (original value - feature baseline)\*ratio.
The standard deviation will be used to calculate the confidence interval for the new numeric value.

CI(confidence interval) = (new numeric value +/- Z \* standard deviation/sqre(sample-count))
[CI = \bar{x} \pm z \frac{s}{\sqrt{n}}](https://www.mathsisfun.com/data/confidence-interval.html)

The mappings, ratio and standard deviation(s) will be saved for each feature.

#### Feature list

- community \*1
- garage \*1
- parking total \*1
- bedrooms \*1
- bedroom plus \*1
- bathrooms \*1
- kitchen total \*1
- lot size front
- lot size total area
- room size total area
- built year
- sqft
- month(sold date) \*1: reorganized based on price baseline instead of calendar month. Month 0 is 2 years ago, and month 24 is this month. Calculate 4 years of data.

\*1 discrete values
\*2 based to the feature baseline of the city and type of building

**Note:** Numbers to save: toCity(delta, std, cnt), toType(delta, std, cnt).
For discrete values, the mapping will be saved. For continues values, the ratio will be saved.

#### Data Structure

- DB: (key) -> (fields). s for sale, r for rent.
  - (YYMMs|r)
    - ts: running date and time
    - dateSpan: date span (in days)
    - count: number of records
    - seconds: cost time in seconds
    - startOnD: start date
    - endOnD: end date
  - (YYMMs|r+Prov+City+Type) -> (mean, median, std, cnt, min, max)
  - (YYMMs|r+Prov+City+Type+Feature): for discrete values
    - (mean, median, std, cnt, min, max): feature average value
    - values: for discrete values
      - (value) -> (mean, median, std, cnt, delta, deltaM, min, max) for discrete values. delta to (Prov+City+Type), deltaM is for Median
    - ratios: for continues values
      - (mean, median, std, cnt, min, max): ratio to (Prov+City+Type), ratioM is for Median
- Cache: Dictionary. Sale and Rent are separated.
  - (ts,count,seconds,dateSpan,startOnD,endOnD)
  - Prov + City + Type
    - (mean, median, std, cnt, min, max)
    - features:
      - (Feature)
        - (mean, median, std, cnt, min, max): feature values or stats(count, unique, top, freq)
        - values: for discrete values
          - (value) -> (mean, median, std, cnt, delta, deltaM, min, max) price stats for each discrete value. delta to (Prov+City+Type), deltaM is for Median
        - ratios: for continues values
          - (mean, median, std, cnt, min, max): ratio of price/unit to (Prov+City+Type), ratioM is for Median

### 1.2 Evaluation based on features

Each property is evaluated based on the delta of each feature value to the feature baseline. The evaluation is calculated by the following formula:

    feature evaluation = ratio * (feature value - feature average) * 1000
    evaluation = type baseline + feature evaluation

### 1.3 Weight of each feature

Linear regression is used to calculate the weight of each feature. X will be the feature evaluation, and Y will be the sold price. The weight of each feature will be the coefficient of the feature evaluation. An overall evaluation is calculated by the following formula:

    evaluation = type baseline + sum(feature evaluation * feature weight)

LightGBM is used to calculate the final price prediciton. X will be all the feature and overall evaluation, and Y will be the sold price.

[coefficient calculation](https://stackoverflow.com/questions/38250707/calculate-coefficients-in-a-multivariate-linear-regression)
[sklearn](https://datatofish.com/multiple-linear-regression-python/)

## 2. Nearest neighbor approach

Based on the baseline approach, we can use the nearest neighbor approach to calculate the evaluation.

[mahalanobis distance](https://www.youtube.com/watch?v=spNpfmWZBmg)
[scikit-learn](https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html)

### 2.1 Description

We use mahalanobis distance apply to the mapped/converted feature value to find the nearest neighbor properties to calculate the evaluation.

We first get the max number of neighbors for the subject property with the least distances. Then we find the sudden raise of distance to cut off the unrelated properties. The reminding properties are the nearest neighbors. The remaining number shall bigger than a minimum number of neighbors of 3 sold and 3 on the market.

### 2.2 Evaluation

We then use the trend of both sold and unsold properties to calculate the parameters of two one-dimensional equation of the lines. A standard deviation is calculated for each of the lines to serve as a pressure to help find the optimized evaluation point, and also the confidence interval for the evaluation.
The evaluation is calculated by the following formula(To review?):

    evaluation = (sold line slope * sold line intercept + unsold line slope * unsold line intercept) / 2

## 3. LightGBM approach

### 3.1 Description

Use direct price as the target to train a LightGBM model.
