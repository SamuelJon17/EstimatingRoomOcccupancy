import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.model_selection import train_test_split
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import STL

from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import statsmodels.api as sm
from scipy.stats import chi2
import statistics
from scipy.signal import dlsim
from statsmodels.tsa.arima.model import ARIMA

## LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

sns.set(style="darkgrid")

################################################################################
# Methods 
################################################################################ 

########################################
# Via Lab 1
def ADF_Cal(x, conf=0.05):
    result = adfuller(x)
    print('p-value: %f' % result[1]) 
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    print(f'Series is {"not " if result[1] >= conf else ""}stationary')

def kpss_test(timeseries, conf=0.05):
    print ('Results of KPSS Test:')
    statistic, p_value, n_lags, critical_values = kpss(timeseries, regression='c', nlags="auto")
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'\t{key} : {value}')
    print(f'Series is {"not " if p_value < conf else ""}stationary')
    
def getRollingMeanVar(df, rnd=2):
    newDf = df.copy()
    for col in newDf.select_dtypes(include=np.number).columns:
        for row in range(len(newDf)):
            newDf.at[row, col+'_rollingMean']=newDf.iloc[0:row+1][col].mean()
            newDf.at[row, col+'_rollingVar']=newDf.iloc[0:row+1][col].var()
    newDf.loc[:, newDf.filter(regex='rolling').columns] = newDf.filter(regex='rolling').fillna(0.0)
    newDf.loc[:, newDf.filter(regex='rolling').columns] =  newDf.filter(regex='rolling').round(rnd)
    return newDf

def getRollingMeanVarPlot(df, x_label="",label=""):
    if label != x_label:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False, figsize=(10, 10))
        if x_label=="":
            ax1.plot(range(len(df)), df[label+'_rollingMean'])
            ax1.set_title(f'Rolling Mean for {label}')
            ax1.set_ylabel("Magnitude")
            ax2.plot(range(len(df)), df[label+'_rollingVar'])
            ax2.set_title(f'Rolling Variance for {label}')
            ax2.set_ylabel("Magnitude")
            plt.xlabel("Samples")
            plt.show()
        else:
            ax1.plot(df[x_label], df[label+'_rollingMean'])
            ax1.set_title(f'Rolling Mean for {label}')
            ax1.set_ylabel("Magnitude")
            ax2.plot(df[x_label], df[label+'_rollingVar'])
            ax2.set_title(f'Rolling Variance for {label}')
            ax2.set_ylabel("Magnitude")
            plt.xlabel("Samples")
            plt.show()
    
########################################
# Via Lab 2

def getACFSingle(data, timeLag=0):
    """
    :data - numpy array
    :timeLag - t2-t1
    :return - Autocorrelation value [-1,1]
    """
    mean = np.mean(data)
    return np.sum((data[timeLag:]-mean)*(data[:len(data[timeLag:])]-mean))/np.sum(np.square(data - mean))
    
def getACFAll(data, timeLagMax=10, title="White Noise", plot = False, soloPlot=True):
    """
    :data - numpy array
    :timeLagMax - maximum time lag wish to calcualte. Will calculate all time lag from 0 to timeLagMax
    :return data - Autocorrelation value for timeLag in [0:timeLagMax]
    """
    returnData = []
    for timeLag in range(timeLagMax+1):
        returnData.append(getACFSingle(data,timeLag))
    returnDataX = list(range(-timeLagMax,0))+list(range(timeLagMax+1))
    returnDataY = returnData[::-1]+returnData[1:]
    if plot:
        if soloPlot:
            fig = plt.figure(figsize=(7,7))
        ci = 1.96/np.sqrt(len(data))
        plt.stem(returnDataX,returnDataY, markerfmt="or", basefmt="black",linefmt="C0-")
        plt.fill_between(returnDataX, len(returnDataX)*[-ci], len(returnDataX)*[ci], color='b', alpha=.1)
        plt.xlabel("Lags")
        plt.ylabel("Magnitude")
        plt.title("Autocorrelation Function of "+title)
    return returnDataX, returnDataY


########################################
# Via HW 2
def trainAvgMethod(train, test):
    predTrain = [0]
    for idx in range(len(train)):
        predTrain = predTrain + [round(sum(train[:idx+1])/(idx+1),1)]
    predTest = predTrain.pop(-1)
    return predTrain, len(test)*[predTest]

def trainNaiveMethod(train,test):
    predTrain = [0] + train[:-1]
    predTest = len(test)* [train[-1]]
    return predTrain, predTest
    
def trainDriftMethod(train,test):
    predTrain = [0,0]
    predTest = []
    for idx in range(2,len(train)):
        m = (train[idx-1]-train[0])/(idx-1)
        predTrain = predTrain + [train[idx-1]+m]
    m = (train[-1]-train[0])/len(train)
    for idx in range(len(test)):
        predTest = predTest + [train[-1]+(idx+1)*m]
    return predTrain, predTest

def getError(act, pred, offset=0):
    error = [round(a-b,1) for a, b in zip(act, pred)][offset:]
    return error, [round(x**2,1) for x in error]

def getMSE(errorSq,offset = 1):
    return round(sum(errorSq[offset:])/len(errorSq[offset:]),2)

def getMethodDF(train, test, offsetDF=1, method="Average"):
    if method=="Average":
        predTrain, predTest = trainAvgMethod(train, test)
    elif method=="Naive":
        predTrain, predTest = trainNaiveMethod(train, test)
    elif method=="Drift":
        predTrain, predTest = trainDriftMethod(train, test)
    
    errorTrain, errorTrainSq = getError(train, predTrain, offset=offsetDF)
    errorTest, errorTestSq = getError(test, predTest, offset=0)
    predTrain[:offsetDF]=offsetDF*[None]
    trainDF = pd.DataFrame({'actual': train, 'predicted': predTrain, "error":offsetDF*[None]+errorTrain, "errorSq":offsetDF*[None]+errorTrainSq}, index=range(1,len(train)+1))
    testDF= pd.DataFrame({'actual': test, 'predicted': predTest, "error":errorTest, "errorSq":errorTestSq}, index=range(len(train)+1,len(train)+len(test)+1))
    return trainDF, testDF

def getGraph(dfTrain, dfTest, name="Average"):
    fig = plt.figure(figsize=(12,8))
    ## To connect train and test
        #plt.plot(list(dfTrain.index)+[len(dfTrain)+1], list(dfTrain.actual)+[dfTest.actual.iloc[0]], marker="o",label="Train Set")
    plt.plot(list(dfTrain.index), list(dfTrain.actual),label="Train Set")
    plt.plot(dfTest.index, dfTest.actual,label="Test Set")
    plt.plot(dfTest.index,dfTest.predicted,label="Predicted H-Step")
    plt.xlabel("Time")
    plt.ylabel("Dependent Variable")
    plt.title(name+" Method Prediction for Data")
    plt.grid()
    plt.legend()
    plt.show()

########################################
# Via Lab 5

def getPhiJKK(acf, j, k):
    assert(k<=len(acf))
    assert(j<=len(acf))
    assert((j+k)<=len(acf))
    acf = np.concatenate((acf[::-1],acf[1:]))
    mid = len(acf)//2
    pacfNum = []
    pacfDen = []
    for row in range(k):
        pacfNum.append(np.append(acf[mid+j+row:mid+j+row-k+1:-1],acf[mid+j+row+1]))
        pacfDen.append(np.append(acf[mid+j+row:mid+j+row-k+1:-1],acf[mid+j-k+row+1]))
    pacfNum = np.vstack(pacfNum)
    pacfDen = np.vstack(pacfDen)
#     return pacfNum, pacfDen
    ret = np.linalg.det(pacfNum)/np.linalg.det(pacfDen)
    return round(ret,2) if ret!=np.nan else np.nan


def getGPAC(acf, maxJ=7, maxK=7, plot=True):
    gpac = []
    for j in range(maxJ):
        row = []
        for k in range(1,maxK+1):
            row.append(getPhiJKK(acf,j,k))
        gpac.append(row)
    gpac = pd.DataFrame(gpac, columns=list(range(1,maxK+1)))
    if plot:    
        fig = plt.figure(figsize=(12,8))
        sns.heatmap(gpac, annot=True)
        plt.title("Generalized Partial Autocorrelation (GPAC) Table")
        plt.xlabel("n_a")
        plt.ylabel("n_b")
        plt.show()
    return gpac

########################################
# Via HW 5

def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags) 
    pacf = sm.tsa.stattools.pacf(y, nlags=lags) 
    fig = plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data') 
    plot_acf(y, ax=plt.gca(), lags=lags) 
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()
    
def arma_lma(y, ar_order, ma_order, delta=1e-6, max_iter=100, 
             tol=1e-3, lambda_init=1e-2, lambda_factor=10, lambda_max = 1e9):
    ### HELPER FUNCTIONS
    def getDLSIMParams(params):
        """
        Note that this is to solve for eT rather than yT hence the ordering of the 
        ar and ma coefficients are reversed
        """
        a = np.hstack((1, params[:ar_order])) ## ar
        b = np.hstack((1, params[ar_order:])) ## ma 
        if ma_order < ar_order:
            b = np.append(b, np.array((ar_order-ma_order)*[0]))
        if ma_order > ar_order:
            a = np.append(a, np.array((ma_order-ar_order)*[0]))     
        return (a,b,1)
    
    residual_fun = lambda params: dlsim(getDLSIMParams(params), y.squeeze())[1]
    SSE_fun = lambda e: (e.T @ e).squeeze()

    ## INITIALIZATION
    var = 0
    cov = 0 
    params = np.zeros(ar_order+ma_order)
    res = y.copy()
    SSE = SSE_fun(res).item()   
    lambda_val = lambda_init
    X = np.zeros((len(y),len(params)))
    SSE_graph = [SSE]
    error = ""
    # PERFORM Levenberg-Marquardt
    for i in range(max_iter):
        
        # CALCULATE X
        for j in range(len(params)):
            params_pert = params.copy()
            params_pert[j] += delta
            X[:, j] = ((res - residual_fun(params_pert)) / delta).squeeze()
        
        # CALCULATE A, g, DeltaTheta (dp)
        A = X.T @ X
        g = X.T @ res 
        A += lambda_val * np.identity(len(A))
        dp = np.linalg.inv(A) @ g


        # UPDATE PARAMETERS
        params = params + dp.squeeze()
        res = residual_fun(params)
        SSE_new = SSE_fun(params)
        SSE_graph.append(SSE_new)
        
        # CONVERGENCE CHECK
        if SSE_new < SSE:
            if np.linalg.norm(dp) < tol:
                var = SSE_new/(X.shape[1]-X.shape[0])
                cov = (var*np.linalg.inv(A)).squeeze()
                break    
            lambda_val /= lambda_factor
        else:
            lambda_val *= lambda_factor
            if lambda_val > lambda_max:
                print("ERROR BROKE LAMBDA_MAX")
                error = "ERROR BROKE LAMBDA_MAX"
                break
        SSE = SSE_new
    if i == max_iter:
        print("ERROR PASS ITERATION")
        error = "ERROR PASS ITERATION"
    return params, var, cov, SSE_graph, error

########################################
# New

def backward_stepwise_regression(X, y, criteria='aic'):
    p = X.shape[1]
    selected_features = np.arange(p)
    model = sm.OLS(y, X).fit()
    if criteria == 'aic':
        best_criteria_value = model.aic
    elif criteria == 'bic':
        best_criteria_value = model.bic
    else:
        best_criteria_value = -model.rsquared_adj
    for _ in range(p):
        best_feature_to_remove = None
        for feature_to_remove in selected_features:
            remaining_features = np.setdiff1d(selected_features, feature_to_remove)
            remaining_X = X[:, remaining_features]
            remaining_model = sm.OLS(y, remaining_X).fit()
            if criteria == 'aic':
                criterion_value = remaining_model.aic
            elif criteria == 'bic':
                criterion_value = remaining_model.bic
            else:
                criterion_value = -remaining_model.rsquared_adj
            if criterion_value < best_criteria_value:
                best_criteria_value = criterion_value
                best_feature_to_remove = feature_to_remove
        selected_features = np.setdiff1d(selected_features, best_feature_to_remove)
        selected_X = X[:, selected_features]
    return selected_features



################################################################################
# Results 
################################################################################ 

## Data Processing
df = pd.read_csv("Occupancy_Estimation.csv", delimiter=',', skipinitialspace = True)
df.columns = df.columns.str.replace(' ', '') 
print(df.shape)
df['time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y/%m/%d %H:%M:%S')
df.drop(columns=['Date','Time'], inplace = True)
df = df[df['time'].dt.month == 12]
df = df[df['time'].dt.day.isin([22,23,24])]
df = df[['time']+list(df.columns)[:-1]]
df['time'] = pd.date_range("2017-12-22 10:49:41", freq="30S", periods=len(df))

print("The shape of the December ONLY data is (row, column):", str(df.shape))

df.describe()
df.info()

df.drop(columns=['time'],axis=1).hist(figsize=(15,15))
plt.show()

plt.figure(figsize=(15,5))
plt.plot(df.time.values, df.Room_Occupancy_Count.values)
plt.xlabel('Time')
plt.ylabel('Room Occupancy Count')
plt.title('Time Series for Room Occupancy Count', fontsize=20)
plt.show()

## ACF and PACF
curr_fig, curr_ax = plt.subplots(figsize=(7, 7))
plot_acf(df.Room_Occupancy_Count.values, lags=20, ax=curr_ax, title="ACF of Room_Occupancy_Count")
plt.show()

getACFAll(df.Room_Occupancy_Count, timeLagMax=20, title="of Room_Occupancy_Count", plot = True, soloPlot=True)
plt.show()

curr_fig, curr_ax = plt.subplots(figsize=(7, 7))
plot_pacf(df.Room_Occupancy_Count.values, lags=20, ax=curr_ax, title="PACF of Room_Occupancy_Count")
plt.show()

## Correlation Map
plt.figure(figsize=(10,10))
sns.heatmap(df.drop(['Room_Occupancy_Count'], axis=1).corr(method='pearson'), annot=False, cmap="seismic")
plt.title(f"Correlation Matrix for All Dependent Variables",  fontsize=20)
plt.show()

## Split data
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.2, random_state=17, shuffle=False)
dfY_train = pd.DataFrame({"time":X_train['time'], "Room_Occupancy_Count":y_train})
dfY_test = pd.DataFrame({"time":X_test['time'], "Room_Occupancy_Count":y_test})

print(f"Total Training Entires: {X_train.values.shape[0]}")
print(f"Total Test Entires: {X_test.values.shape[0]}")

## Stationality

## ADF and KPSS
print("ADF Test for Dependent Variable")
ADF_Cal(y_train, conf=0.05)
print("\n")
print("KPSS Test for Dependent Variable")
kpss_test(y_train, conf=0.05)

## Rolling Average and Variance of Independent
dfRolling = getRollingMeanVar(X_train, rnd=1)
for label in X_train.columns:
    getRollingMeanVarPlot(dfRolling, x_label="time", label=label)
    
## Rolling Average and Variance of Dependent
getRollingMeanVarPlot(getRollingMeanVar(dfY_train, rnd=1), x_label="time", label="Room_Occupancy_Count")

X_train = X_train.set_index("time")
X_test = X_test.set_index("time")
dfY_train = dfY_train.set_index("time")
dfY_test = dfY_test.set_index("time")

## Time Series Decomposition

## Plot decomposed components
dfY_STL = STL(dfY_train['Room_Occupancy_Count'].values,  period=2*60*24).fit()
dfY_STL.plot().show()

## Plot detrended and seasonally adjusted
fig = plt.figure(figsize=(10,10))
plt.plot(dfY_train.index, dfY_train.Room_Occupancy_Count,label="Original")
plt.plot(dfY_train.index, (dfY_STL.resid + dfY_STL.seasonal),label=f"Detrended")
plt.plot(dfY_train.index, (dfY_STL.resid + dfY_STL.trend),label=f"Seasonally Adjusted")
plt.xlabel("Time")
plt.ylabel("Room Occupancy Count")
plt.title(f"Seasonally Adjusted vs Detrended vs. Original")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

## State trend and seasonal strength
Ft = np.maximum(0 ,1 - dfY_STL.resid.var()/(dfY_STL.resid+ dfY_STL.trend).var()) 
print(f'The strength of trend for this data set is ~{Ft*100:.2f}%')

Fs= np.maximum(0 ,1 - dfY_STL.resid.var()/(dfY_STL.resid+dfY_STL.seasonal).var()) 
print(f'The strength of seasonality for this data set is ~{Fs*100:.2f}%')

## Holt-Winters Model -- Additive Method
HWES3_ADD = ExponentialSmoothing(dfY_train["Room_Occupancy_Count"].values,trend='add',seasonal='add',seasonal_periods=24).fit()
dfY_train['HWES3_ADD'] = HWES3_ADD.fittedvalues
dfY_test['HWES3_ADD'] = HWES3_ADD.forecast(len(dfY_test))

fig = plt.figure(figsize=(10,10))
plt.plot(dfY_train.index, dfY_train.Room_Occupancy_Count,label="Train")
plt.plot(dfY_train.index, dfY_train.HWES3_ADD,label=f"Training-Fit")
plt.plot(dfY_test.index, dfY_test.Room_Occupancy_Count,label="Test")
plt.plot(dfY_test.index, dfY_test.HWES3_ADD,label=f"Predicted H-step")
plt.xlabel("Time")
plt.ylabel("Room Occupancy Count")
plt.title(f"Holt Winters Triple Exponential Smoothing: Additive Seasonality on Training Set")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

## Base Models - Average, Naive, Drift
method =["Average", "Naive", "Drift"]
results = []
for i in method:
    print("\n============================================")
    print(f"{i} Method")
    dfTrain, dfTest = getMethodDF(list(dfY_train.Room_Occupancy_Count.values), list(dfY_test.Room_Occupancy_Count.values), offsetDF=1, method=i)
    Q = sm.stats.acorr_ljungbox(dfTrain.error[2:], lags=[5], boxpierce=True, return_df=True)
    mseDF = pd.DataFrame({"MSE":[getMSE(list(dfTrain.errorSq),offset=2),getMSE(list(dfTest.errorSq),offset=0)],
                         "Variance":[round(statistics.variance(list(dfTrain.error[2:])),1),round(statistics.variance(list(dfTest.error)),1)],
                         "Q via Box-Pierce":[round(Q.bp_stat.iloc[0],2),None]},index=["Residual", "Forecast"])
    getGraph(dfTrain,dfTest, name=i)
    getACFAll(list(dfTrain.error[1:]), timeLagMax=10, title=f"{i} Method Residuals", plot = True)
    plt.show()
    print(mseDF)
    results.append(mseDF)

results = pd.DataFrame({"Q via Box-Pierce":[x["Q via Box-Pierce"].iloc[0] for x in results],
             "MSE - Residual":[x.MSE.iloc[0] for x in results],
             "MSE - Forecast":[x.MSE.iloc[1] for x in results],
             "Variance - Residual":[x.Variance.iloc[1] for x in results]},
            index=method)

## Base Models - Exponential Smoothing w/ different alphas
c = plt.cm.get_cmap("hsv", 14)
plt.figure(figsize=(12, 8))
plt.plot(dfY_train.index, dfY_train.Room_Occupancy_Count, color=c(10), label="Train")
plt.plot(dfY_test.index, dfY_test.Room_Occupancy_Count, color=c(9), label="Test")
alpha= [0.0,0.25,0.75,0.99]
for idx in range(len(alpha)):
    sse = SimpleExpSmoothing(dfY_train.Room_Occupancy_Count.values, initialization_method="known", initial_level=dfY_train.Room_Occupancy_Count[0]).fit(
        smoothing_level=alpha[idx], optimized=False)
    plt.plot(dfY_train.index, sse.fittedvalues, color=c(idx))
    plt.plot(dfY_test.index, sse.forecast(len(dfY_test)), color=c(idx),label=r"$\alpha=$"+str(alpha[idx]))
plt.xlabel("Time")
plt.ylabel("Room Occupancy Count")
plt.title("Comparing Different Alpha for SES Method")
plt.grid()
plt.legend()
plt.show()

## Feature Selection / Elimination

## SVD
X_train_np = np.c_[np.ones(len(X_train)),X_train.iloc[:,1:].to_numpy()]
s, d, v = np.linalg.svd(X_train_np)
colinearityViaSVD = np.where(d<=0.5)[0]
print("SVD Results")
print(f"Via SVD with a threshold of <= 0.5 the following fields exhibit colinearity: {', '.join(list(X_train.columns[colinearityViaSVD]))}")

## BackwardsStepwise Regression
selectedFeatures_BIC = backward_stepwise_regression(X_train.values, y_train.values, criteria='bic')
selectedFeatures_AIC = backward_stepwise_regression(X_train.values, y_train.values, criteria='aic')
selectedFeatures_R2Adj = backward_stepwise_regression(X_train.values, y_train.values, criteria='rsquared_adj')
print("\nBackwards Stepwise Regression Results")
print(f"\nSelected features via Backwards Stepwise Regression w/ BIC as Evaluation: {', '.join(list(X_train.columns[selectedFeatures_BIC]))}")
print(f"\nSelected features via Backwards Stepwise Regression w/ AIC as Evaluation: {', '.join(list(X_train.columns[selectedFeatures_AIC]))}")
print(f"\nSelected features via Backwards Stepwise Regression w/ R2Adjusted as Evaluation: {', '.join(list(X_train.columns[selectedFeatures_R2Adj]))}")
selectedFeatures =  list(set(selectedFeatures_BIC)&set(selectedFeatures_AIC)&set(selectedFeatures_R2Adj))
selectedFeatures_Dropped = list(set(range(X_train.values.shape[1])) - set(selectedFeatures))
print(f"\nOverlap amongst all: {', '.join(list(X_train.columns[selectedFeatures]))}")
print(f"\nDropped Features: {', '.join(list(X_train.columns[selectedFeatures_Dropped]))}")

## Multiple Linear Regression
linearReg = sm.OLS(y_train,sm.add_constant(X_train.iloc[:,selectedFeatures].values)).fit()

fig = plt.figure(figsize=(12,8))
plt.plot(X_train.index, y_train, label="Train Set")
plt.plot(X_test.index, y_test, label="Test Set")
plt.plot(X_test.index, linearReg.predict(sm.add_constant(X_test.iloc[:,selectedFeatures].values)),label="Predicted")
plt.xlabel("Time")
plt.ylabel("Room Occupancy Count")
plt.title("Multivariate Linear Regression")
plt.grid()
plt.legend()
plt.show()


## Summary Report
print(linearReg.summary())

## One-step ahead prediction on test set
X_test_ = sm.add_constant(X_test.iloc[:,selectedFeatures].values)
y_pred = linearReg.predict(sm.add_constant(X_test_))

# Calculate evaluation metrics
n = len(y_test)
k = X_test_.shape[1] - 1 
resid = y_test.values - y_pred
rmse = np.sqrt(np.sum((resid)** 2) / (n - k - 1))
aic, bic, rsquared, rsquared_adj= linearReg.aic, linearReg.bic, linearReg.rsquared, linearReg.rsquared_adj

print("\nEvaluation Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {rsquared:.4f}")
print(f"Adjusted R-squared: {rsquared_adj:.4f}")
print(f"AIC: {aic:.4f}")
print(f"BIC: {bic:.4f}")


## Hypothesis tests
f_stat = linearReg.fvalue
f_pval = linearReg.f_pvalue
t_stat = linearReg.tvalues
t_pval = linearReg.pvalues
print("\nHypothesis Tests:")
print(f"F-statistic: {f_stat:.4f}")
print(f"F p-value: {f_pval:.4f}")
for i in range(1, k+1):
    print(f"t({i}): {t_stat[i]:.4f}, p-value: {t_pval[i]:.4f}")


# Calculate ACF of residuals and Q-value
acf = sm.tsa.stattools.acf(resid, nlags=10, qstat=True, fft=True)
q_value = acf[2][-1]
print("\nACF of Residuals:")
print(acf[0])
print("\nQ-value:")
print(f"Q-value: {q_value:.4f}")

# Calculate variance and mean of residuals
resid_var = np.var(resid)
resid_mean = np.mean(resid)
print("\nResiduals:")
print(f"Residual Variance: {resid_var:.4f}")
print(f"Residual Mean: {resid_mean:.4f}")

## ARMA, ARIMA, SARIMA -- GPAC
yTrain_acf = sm.tsa.acf(y_train.values, nlags=20)
# use one or the other for ACF
_, a = getACFAll(y_train.values, timeLagMax=20, title="Dependent Variable", plot = True, soloPlot=True)
# pacf
ACF_PACF_Plot(y_train.values,20)
getGPAC(yTrain_acf, maxJ=10, maxK=10, plot=True)

## ARMA Model Parameters -- LMA

## ARMA - AR:1, Diff:0, MA:0

AR = 1
Diff = 0
MA =0 

params, _, _, SSE_graph, _ = arma_lma(y_train.values.reshape(-1, 1), AR, MA, delta=1e-6, max_iter=100, 
         tol=1e-3, lambda_init=1e-2, lambda_factor=10, lambda_max = 1e9)

print(f"Predicted AR Order: {[round(i,4) for i in params[:AR]]}")
print(f"Predicted MA Order: {[round(i,4) for i in params[AR:]]}")

fig = plt.figure(figsize=(7,7))
plt.plot(list(range(len(SSE_graph))), SSE_graph)
plt.title("SSE over time")
plt.xlabel("iteration")
plt.ylabel("SSE for Order AR:1 MA:0")
plt.show()

ARMA = ARIMA(y_train.values, order=(AR, Diff, MA)).fit()
print(f"Statsmodel AR Parameters: {[round(i,2) for i in ARMA.arparams]}") 
print(f"Statsmodel MA Parameters: {[round(i,2) for i in ARMA.maparams]}") 
print(ARMA.summary())

fig = plt.figure(figsize=(12,8))
plt.plot(X_train.index, y_train, label="Train Set")
plt.plot(X_train.index, ARMA.predict(start=0, end=len(y_train)-1),label="ARMA (1,0,0) Predicted")
plt.plot(X_test.index, y_test, label="Test Set")
plt.plot(X_test.index, ARMA.forecast(steps=len(y_test)) ,label="ARMA (1,0,0) Forecast")
plt.xlabel("Time")
plt.ylabel("Room Occupancy Count")
plt.title("ARMA (1,0,0) Process")
plt.grid()
plt.legend()
plt.show()

## ARMA - AR:5, Diff:0, MA:0

AR = 5
Diff = 0
MA =0 

params, _, _, SSE_graph, _ = arma_lma(y_train.values.reshape(-1, 1), AR, MA, delta=1e-6, max_iter=100, 
         tol=1e-3, lambda_init=1e-2, lambda_factor=10, lambda_max = 1e9)

print(f"Predicted AR Order: {[round(i,4) for i in params[:AR]]}")
print(f"Predicted MA Order: {[round(i,4) for i in params[AR:]]}")

fig = plt.figure(figsize=(7,7))
plt.plot(list(range(len(SSE_graph))), SSE_graph)
plt.title("SSE over time")
plt.xlabel("iteration")
plt.ylabel("SSE for Order AR:1 MA:0")
plt.show()

ARMA = ARIMA(y_train.values, order=(AR, Diff, MA)).fit()
print(f"Statsmodel AR Parameters: {[round(i,2) for i in ARMA.arparams]}") 
print(f"Statsmodel MA Parameters: {[round(i,2) for i in ARMA.maparams]}") 
print(ARMA.summary())

fig = plt.figure(figsize=(12,8))
plt.plot(X_train.index, y_train, label="Train Set")
plt.plot(X_train.index, ARMA.predict(start=0, end=len(y_train)-1),label="ARMA (5,0,0) Predicted")
plt.plot(X_test.index, y_test, label="Test Set")
plt.plot(X_test.index, ARMA.forecast(steps=len(y_test)) ,label="ARMA (5,0,0) Forecast")
plt.xlabel("Time")
plt.ylabel("Room Occupancy Count")
plt.title("ARMA (5,0,0) Process")
plt.grid()
plt.legend()
plt.show()

## Diagnostic Analysis 

ARMA = ARIMA(y_train.values, order=(1, 0, 0)).fit()
y_pred = ARMA.predict(start=0, end=len(y_train)-1)

# Confidence Interval
print("\nConfidence intervals:\n", ARMA.conf_int())

# Zero/Pole cancellation
print("\nZero/pole cancellation:\n", np.roots(np.r_[1, -ARMA.arparams]))
print("No root cancellation, only AR process")

# Calculate and display chi-square test
resid = y_train.values - y_pred
chi2, p_value = sm.stats.acorr_ljungbox(resid, lags=[len(ARMA.arparams)])
print("\nChi-square test:")
print("Test statistic:", chi2[0])
print("P-value:", p_value[0])

# Calculate variance and mean of residuals
resid_var = np.var(resid)
resid_mean = np.mean(resid)
print("\nResiduals:")
print(f"Residual Variance: {resid_var:.4f}")
print(f"Residual Mean: {resid_mean:.4f}")

# Calculate variance and mean of forecast
forecast = y_test.values - ARMA.forecast(steps=len(y_test))
forecast_var = np.var(forecast)
forecast_mean = np.mean(forecast)
print("\nForeacst:")
print(f"Foreacst Variance: {forecast_var:.4f}")
print(f"Foreacst Mean: {forecast_mean:.4f}")

## Show residuals are WN
_, _ = getACFAll(resid, timeLagMax=20, title="Residuals for ARMA (1,0,0) Process", plot = True, soloPlot=True)

fig = plt.figure(figsize=(12,8))
plt.plot(X_train.index, y_train, label="Train Set")
plt.plot(X_train.index, ARMA.predict(start=0, end=len(y_train)-1),label="ARMA (1,0,0) Predicted")
plt.plot(X_test.index, y_test, label="Test Set")
plt.plot(X_test.index, ARMA.forecast(steps=len(y_test)) ,label="ARMA (1,0,0) Forecast")
plt.xlabel("Time")
plt.ylabel("Room Occupancy Count")
plt.title("ARMA (1,0,0) Process")
plt.grid()
plt.legend()
plt.show()

## Deep Learning Model -- LSTM

X_train_LSTM = np.reshape(X_train.values, (X_train.values.shape[0], 1, X_train.values.shape[1]))
X_test_LSTM = np.reshape(X_test.values, (X_test.values.shape[0], 1, X_test.values.shape[1]))


# define the model architecture
model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(X_train_LSTM.shape[1], X_train_LSTM.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')

# train the model with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train_LSTM, y_train.values, epochs=50, batch_size=32, validation_data=(X_test_LSTM, y_test.values), callbacks=[early_stop])

# make predictions on the test set
y_pred = model.predict(X_test_LSTM)

# calculate performance metrics (e.g. mean squared error)
mse = np.mean((y_test.values - y_pred)**2)
print('MSE:', mse)

# Residual and Forecast Variance and Meann
resid = y_train.values - model.predict(X_train_LSTM).squeeze()
resid_var = np.var(resid)
resid_mean = np.mean(resid)
print("\nResiduals:")
print(f"Residual Variance: {resid_var:.4f}")
print(f"Residual Mean: {resid_mean:.4f}")

forecast = y_test.values - model.predict(X_test_LSTM).squeeze()
forecast_var = np.var(forecast)
forecast_mean = np.mean(forecast)
print("\nForeacst:")
print(f"Foreacst Variance: {forecast_var:.4f}")
print(f"Foreacst Mean: {forecast_mean:.4f}")

## Show residuals are WN
# _, _ = getACFAll(resid, timeLagMax=20, title="Residuals for LSTM", plot = True, soloPlot=True)

## Plot train, test and predicted
fig = plt.figure(figsize=(12,8))
plt.plot(X_train.index, y_train, label="Train Set")
plt.plot(X_test.index, y_test, label="Test Set")
plt.plot(X_test.index, y_pred ,label="LSTM Forecast")
plt.xlabel("Time")
plt.ylabel("Room Occupancy Count")
plt.title("LSTM")
plt.grid()
plt.legend()
plt.show()