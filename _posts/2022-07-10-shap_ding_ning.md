---
layout: post
title: Summary of ding_ning_shap_project
date: 2022-07-03 11:12:00-0400
description: write-up.
---


## doing basic preprocessing

# 

import shap
import pandas as pd



```python
import pyreadstat
path = 'ding_ning_all_origin.sav'
df, meta = pyreadstat.read_sav(path, encoding='gbk')

Y = df['group'][0:270]
```


```python
df.columns

```




    Index(['group', 'Zphonedeletion', 'Zonsetrime', 'ZSpoonerism', 'ZRANdigit',
           'ZRANpic', 'ZRANdice', 'ZRANcolor', 'ZDigitSpan', 'ZMA', 'Z假字', 'Z部件错误',
           'Z位置错误', 'Z线条图', 'ZNumberACC', 'ZSymbolACC', 'ZColorACC', 'ZChineseACC',
           'age', 'IQper', 'Gender', '父亲最高学历（问卷星）', '母亲最高学历（问卷星）', '姓名', '学号',
           'Zphonologicalaccuracy', 'Zphonologicalspeed', 'ZOaverage', '认知缺陷1.5SD',
           'Z正字法', 'SESnew', 'SES分组', 'Zacc', 'paper', 'SESQ', 'expriment2',
           'VAR00001', 'Zphonologicalskills13', 'ZPA11', '父亲', '母亲', '亚类型认知缺陷',
           'PA3', 'RAN', 'RAN数字平均时间', 'RAN筛子平均时间', 'RAN图片平均时间', 'RAN颜色平均时间',
           '语音环境交互', 'SES交互（PA）', 'ZVAS', '汉字识别任务得分', '学校', 'Grade', '语素', 'Z语素重评',
           'Spoonerism', '声母、韵母删除测验', '音位删除测验得分', '正字法', '假字', '部件错误', '位置错误',
           '线条图', 'DS', '顺序', '倒序', '表征', '智力分数', '智力等级', '障碍1.5SD', 'ZPA3_RAN',
           '认知缺陷', '词表朗读时间', 'Z词表朗读时间', 'Z词表朗读反', 'ZVAS_verbal', 'ZVAS_nonverbal',
           '词表朗读得分', '顺序得分', '倒序得分', 'PA2', 'Zrepresentation', '语素产生得分原来',
           'Z语素产生得分', 'ZPA声母韵母音位删除', '亚类型8.7', '亚类型', 'VAS_verbal',
           'VAS_nonverbal', 'VAS', 'filter_$', 'ChineseACC', 'NumberACC',
           'SymbolACC', 'ColorACC', 'RAN环境交互', '语素环境交互'],
          dtype='object')




```python
name = df['姓名'][0:270]
number = df['学号'][0:270]


```


```python
needed_col = ['Zphonedeletion','Zonsetrime','ZSpoonerism','ZRANdigit','ZRANpic','ZRANcolor','ZDigitSpan',
             'ZMA','Z假字','Z部件错误','Z位置错误','Z线条图']
```


```python
needed_col = ['Zphonedeletion','Zonsetrime','ZSpoonerism','ZRANdigit','ZRANpic','ZRANcolor','ZDigitSpan',
             'ZMA','ZOaverage']
```


```python
# needed_col = ['Zphonedeletion','Zonsetrime','ZSpoonerism','ZRANdigit','ZRANpic','ZRANdice','ZRANcolor','ZDigitSpan',
#              'ZMA','Z假字','Z部件错误','Z位置错误','Z线条图'
#              ]
```


```python
dict_to_replace = {
    'Z假字' : 'ZPseudoC',
    'Z部件错误' : 'ZIll-formed component',
    'Z位置错误' : 'ZIllegal position',
    'Z线条图': 'ZBW_drawings',
}
```


```python
df_need = df.loc[:, needed_col][0:270]
qq = df_need.interpolate(method='polynomial', order=2,axis=0)
```


```python
df2 = qq.rename(dict_to_replace, axis=1)  # new method
X = df2
```


```python
new_colums = ['Phoneme Deleltion', 'Onset rime Deletion', 'Spoonerism', 'RAN Digits', 'RAN Pictures',
       'RAN Color', 'Digit Span', 'Morphological Awareness', 'Pseudo-character', 'Ill-formed Component',
       'Illegal Position', 'Black-and-white Drawings']
```


```python
new_colums = ['Phoneme Deleltion', 'Onset And Rime Deletion', 'Spoonerism', 'RAN Digits', 'RAN Pictures',
       'RAN Color', 'Digit Span', 'Morphological Awareness','Orthographic Awareness']
```


```python

```


```python
X.columns = new_colums
```

## doing classifcation


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
         test_size = 0.2, random_state = 1234)
```


```python
xgb_model = xgb.XGBRFClassifier()
#xgb_model = xgb.XGBRegressor()
#xgb_model = xgb.XGBClassifier()

xgb_model.fit(X_train, Y_train)
```

    [16:16:51] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    




    XGBRFClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bytree=1, gamma=0, gpu_id=-1, importance_type='gain',
                    interaction_constraints='', max_delta_step=0, max_depth=6,
                    min_child_weight=1, missing=nan, monotone_constraints='()',
                    n_estimators=100, n_jobs=16, num_parallel_tree=100,
                    objective='binary:logistic', random_state=0, reg_alpha=0,
                    scale_pos_weight=1, tree_method='exact', validate_parameters=1,
                    verbosity=None)




```python
# 预测下 X
                
                
```


```python
from sklearn import metrics
y_pred = xgb_model.predict(X_train)

fpr, tpr, thresholds = metrics.roc_curve(Y_train, y_pred)
metrics.auc(fpr, tpr)
```

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\data.py:112: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
      warnings.warn(
    




    0.930839495432797




```python
from sklearn.metrics import accuracy_score


y_pred = xgb_model.predict(X_test)
print(accuracy_score(Y_test , y_pred)*100)
```

    83.33333333333334
    

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\data.py:112: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
      warnings.warn(
    


```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

# cm

y_pred = xgb_model.predict(X)
cf_matrix = confusion_matrix(Y, y_pred)

group_counts = [(value) for value in cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=labels, fmt='', cmap='Blues')
plt.savefig('all.pdf')
```

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\data.py:112: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
      warnings.warn(
    


    
![png](output_21_1.png)
    



```python

```


```python
import seaborn as sns
from sklearn.metrics import confusion_matrix


# cm

y_pred = xgb_model.predict(X_train)
cf_matrix = confusion_matrix(Y_train, y_pred)


sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
fmt='.2%', cmap='Blues')
```

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\data.py:112: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
      warnings.warn(
    




    <AxesSubplot:>




    
![png](output_23_2.png)
    



```python
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# good point 88

# set point 8 

```


```python
# cm

y_pred = xgb_model.predict(X_train)
cm = confusion_matrix(Y_train, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot(cmap='Blues')

```

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\data.py:112: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
      warnings.warn(
    


    
![png](output_25_1.png)
    



```python
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state = 8)
# 交叉验证 轮数

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# ROC_5_fold
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, Y)):
    xgb_model.fit(X.iloc[train], Y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        xgb_model,
        X.iloc[test],
        Y.iloc[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
```

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\data.py:112: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
      warnings.warn(
    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\data.py:112: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
      warnings.warn(
    

    [13:04:07] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [13:04:07] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [13:04:07] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\data.py:112: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
      warnings.warn(
    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\data.py:112: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
      warnings.warn(
    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    

    [13:04:07] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [13:04:07] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\data.py:112: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
      warnings.warn(
    


    
![png](output_26_5.png)
    



```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X)
shap.plots.bar(shap_values, max_display=20,show=False) # default is max_display=12
#plt.savefig('importance_xgb.pdf', format='pdf', dpi=1200, bbox_inches='tight')
#plt.close()
```

    ntree_limit is deprecated, use `iteration_range` or model slicing instead.
    


    
![png](output_27_1.png)
    



```python
qq = X[X.index == 80]
# 获取特殊的 index
print('model prediction is {}'.format(xgb_model.predict(qq)))
print('probability is ',xgb_model.predict_proba(qq))

shap_values = explainer(qq)
shap.plots.force(shap_values,show=False,link='logit')
# 80 
```

    model prediction is [1.]
    probability is  [[0.15268368 0.8473163 ]]
    





<div id='iDNLUEMUVOVKC6MAG6YT4'>
<div style='color: #900; text-align: center;'>
  <b>Visualization omitted, Javascript library not loaded!</b><br>
  Have you run `initjs()` in this notebook? If this notebook was from another
  user you must also trust this notebook (File -> Trust notebook). If you are viewing
  this notebook on github the Javascript has been stripped for security. If you are using
  JupyterLab this error is because a JupyterLab extension has not yet been written.
</div></div>
 <script>
   if (window.SHAP) SHAP.ReactDom.render(
    SHAP.React.createElement(SHAP.AdditiveForceVisualizer, {"outNames": ["f(x)"], "baseValue": -0.16673779487609863, "outValue": 1.7137058973312378, "link": "logit", "featureNames": ["Phoneme Deleltion", "Onset And Rime Deletion", "Spoonerism", "RAN Digits", "RAN Pictures", "RAN Color", "Digit Span", "Morphological Awareness", "Orthographic Awareness"], "features": {"0": {"effect": -0.23587824404239655, "value": 0.7749429401523535}, "1": {"effect": -0.08121000975370407, "value": 1.07153444825783}, "2": {"effect": 0.15782789885997772, "value": -0.6084258121114408}, "3": {"effect": 0.4541446268558502, "value": -0.9361724093562387}, "4": {"effect": 1.3624547719955444, "value": -1.5288467871784706}, "5": {"effect": 0.24427823722362518, "value": -1.6377341835611197}, "6": {"effect": -0.023947343230247498, "value": 0.6253291418193571}, "7": {"effect": 0.03347970172762871, "value": -0.5528771141314316}, "8": {"effect": -0.030705774202942848, "value": 0.18018825214734946}}, "plot_cmap": "RdBu", "labelMargin": 20}),
    document.getElementById('iDNLUEMUVOVKC6MAG6YT4')
  );
</script>




```python
## 双括号 获取列
```


```python
import pandas as pd

ding = pd.read_excel('longlong.xlsx')
needed = ding['ID'].tolist()
```

    Unknown extension is not supported and will be removed
    Conditional Formatting extension is not supported and will be removed
    


```python
len(needed)
```




    21




```python
X['group'] = Y
```


```python
all_dyxia = X[X['group'] == 1]

len(all_dyxia)
```




    123




```python
name = df['姓名'][df['group'] == 1]
number = df['学号'][df['group'] == 1]
```


```python
prediction = []
prediction_prob_0 = []
prediction_prob_1 = []


for i in range(len(all_dyxia)):
    qq = all_dyxia.iloc[[i]].drop(['group'],axis=1)
    # 获取特殊的 index
    prediction.append(xgb_model.predict(qq))
    prediction_prob_0.append(xgb_model.predict_proba(qq)[0][0])
    prediction_prob_1.append(xgb_model.predict_proba(qq)[0][1])
```


```python
import pandas as pd

df = pd.DataFrame(all_dyxia)

```


```python
df['prediction'] = prediction
```


```python
df['prediction_prob_0'] = prediction_prob_0
df['prediction_prob_1'] = prediction_prob_1
df['name'] = name 
df['number'] = number
```


```python
df.to_excel('预测和概率_全部.xlsx',encoding='UTF-8')
```


```python
import pdfkit
for i in range(len(all_dyxia)):
    f = shap.plots.force(shap_values[i],show=False,link='logit')
    shap.save_html("all/index{}.html".format(i), f)
    #pdfkit.from_file("index{}.html".format(i),"index{}.pdf".format(i),configuration=config,options={'javascript-delay':'5000'})
    
#shap.save_html("index.htm", f)
# default is max_display=12
#plt.savefig('importance_xgb1.pdf', format='pdf', dpi=1200, bbox_inches='tight')
#plt.close()
```


```python
import pdfkit
for i in needed:
    f = shap.plots.force(shap_values[i],show=False,link='logit')
    shap.save_html("index{}.html".format(i), f)
    #pdfkit.from_file("index{}.html".format(i),"index{}.pdf".format(i),configuration=config,options={'javascript-delay':'5000'})
    
#shap.save_html("index.htm", f)
# default is max_display=12
#plt.savefig('importance_xgb1.pdf', format='pdf', dpi=1200, bbox_inches='tight')
#plt.close()
```


```python

```


```python
shap.plots.force(shap_values[1],link='logit')
```





<div id='iDA410TJJN62F28DE1MP4'>
<div style='color: #900; text-align: center;'>
  <b>Visualization omitted, Javascript library not loaded!</b><br>
  Have you run `initjs()` in this notebook? If this notebook was from another
  user you must also trust this notebook (File -> Trust notebook). If you are viewing
  this notebook on github the Javascript has been stripped for security. If you are using
  JupyterLab this error is because a JupyterLab extension has not yet been written.
</div></div>
 <script>
   if (window.SHAP) SHAP.ReactDom.render(
    SHAP.React.createElement(SHAP.AdditiveForceVisualizer, {"outNames": ["f(x)"], "baseValue": -0.16673779487609863, "outValue": 1.465014934539795, "link": "logit", "featureNames": ["Phoneme Deleltion", "Onset And Rime Deletion", "Spoonerism", "RAN Digits", "RAN Pictures", "RAN Color", "Digit Span", "Morphological Awareness", "Orthographic Awareness"], "features": {"0": {"effect": 0.3182767331600189, "value": -1.8975079869375957}, "1": {"effect": 0.1109856516122818, "value": -0.7459929816923448}, "2": {"effect": 0.9327531456947327, "value": -2.16523337312765}, "3": {"effect": 0.18342353403568268, "value": -0.7751757748181646}, "4": {"effect": -0.4676022529602051, "value": 0.3531436664827868}, "5": {"effect": 0.20008407533168793, "value": -0.02607880949466546}, "6": {"effect": 0.05061537027359009, "value": -1.183243617792014}, "7": {"effect": 0.3350757360458374, "value": -0.7655661383888168}, "8": {"effect": -0.03185942769050598, "value": -0.09697312463962388}}, "plot_cmap": "RdBu", "labelMargin": 20}),
    document.getElementById('iDA410TJJN62F28DE1MP4')
  );
</script>




```python
shap.plots.force(shap_values[7],link='identity')
```





<div id='iUADBCST8NGXW6IEEDSW0'>
<div style='color: #900; text-align: center;'>
  <b>Visualization omitted, Javascript library not loaded!</b><br>
  Have you run `initjs()` in this notebook? If this notebook was from another
  user you must also trust this notebook (File -> Trust notebook). If you are viewing
  this notebook on github the Javascript has been stripped for security. If you are using
  JupyterLab this error is because a JupyterLab extension has not yet been written.
</div></div>
 <script>
   if (window.SHAP) SHAP.ReactDom.render(
    SHAP.React.createElement(SHAP.AdditiveForceVisualizer, {"outNames": ["f(x)"], "baseValue": -0.16673779487609863, "outValue": 1.9166638851165771, "link": "identity", "featureNames": ["Phoneme Deleltion", "Onset And Rime Deletion", "Spoonerism", "RAN Digits", "RAN Pictures", "RAN Color", "Digit Span", "Morphological Awareness", "Orthographic Awareness"], "features": {"0": {"effect": 0.32180291414260864, "value": -2.899677084596327}, "1": {"effect": 0.06779388338327408, "value": -1.1094984676823798}, "2": {"effect": 0.3874700963497162, "value": -2.16523337312765}, "3": {"effect": 0.3190848231315613, "value": -1.0394986076418686}, "4": {"effect": 0.8858121037483215, "value": -1.152174753148743}, "5": {"effect": 0.24478191137313843, "value": -1.4951098141747072}, "6": {"effect": 0.03803250938653946, "value": -0.7311004278891712}, "7": {"effect": -0.16684700548648834, "value": 1.4558008178248205}, "8": {"effect": -0.014529730193316936, "value": 0.010532123708859586}}, "plot_cmap": "RdBu", "labelMargin": 20}),
    document.getElementById('iUADBCST8NGXW6IEEDSW0')
  );
</script>




```python
import pdfkit
path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
#pdfkit.from_url("http://google.com", "out.pdf", configuration=config)

```


```python
import imgkit


pdfkit.from_file("index{}.html".format(i),"index{}.pdf".format(i),configuration=config)
```




    True




```python
import matplotlib.pyplot as plt
shap.plots.waterfall(shap_values[66],show=True) # For the first observation
#plt.savefig('case1.pdf', format='pdf', dpi=1200, bbox_inches='tight')
#plt.close()
```


    
![png](output_47_0.png)
    



```python
shap.plots.waterfall(shap_values[136],show=True) # For the first observation
#plt.savefig('case2.pdf', format='pdf', dpi=1200, bbox_inches='tight')
#plt.close()
```


    
![png](output_48_0.png)
    



```python
shap.plots.waterfall(shap_values[199]) # For the first observation
plt.savefig('case3.pdf', format='pdf', dpi=1200, bbox_inches='tight')
#plt.close()
```


    
![png](output_49_0.png)
    



    <Figure size 432x288 with 0 Axes>



```python
import pandas as pd

df = pd.DataFrame(shap_values.values)
```


```python
df.columns = new_colums
#df.to_excel('dingning_excel.xlsx')
```


```python
df.to_excel('dingning_excel.xlsx')
```


```python
df['name'] = name
df['number'] = number
```


```python
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state = 8)
# 交叉验证 轮数

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# ROC_5_fold
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, Y)):
    xgb_model.fit(X.iloc[train], Y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        xgb_model,
        X.iloc[test],
        Y.iloc[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
import shap
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X)
shap.plots.bar(shap_values, max_display=20,show=False) # default is max_display=12
```

    The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
    Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
    The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
    Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
    The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
    Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
    

    [18:14:51] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:14:51] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:14:51] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:14:51] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    

    The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
    Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
    The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
    Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
    

    [18:14:51] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    

    ntree_limit is deprecated, use `iteration_range` or model slicing instead.
    


    
![png](output_54_5.png)
    



```python
# The SHAP Values
import shap
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X)
shap.plots.bar(shap_values, max_display=20,show=False) # default is max_display=12
```


    
![png](output_55_0.png)
    



```python
dn_df = pd.DataFrame(shap_values.values)
```


```python
dn_df.columns=needed_col
```


```python
dn_df.to_csv('dingning-all.csv')
```


```python
shap.plots.waterfall(shap_values[88]) # For the first observation
```


    
![png](output_59_0.png)
    



```python
shap.summary_plot(shap_values)
```


    
![png](output_60_0.png)
    



```python
shap.force_plot(shap_values[0:10])
```





<div id='iXQ27Y3IR19KF7KNNES7X'>
<div style='color: #900; text-align: center;'>
  <b>Visualization omitted, Javascript library not loaded!</b><br>
  Have you run `initjs()` in this notebook? If this notebook was from another
  user you must also trust this notebook (File -> Trust notebook). If you are viewing
  this notebook on github the Javascript has been stripped for security. If you are using
  JupyterLab this error is because a JupyterLab extension has not yet been written.
</div></div>
 <script>
   if (window.SHAP) SHAP.ReactDom.render(
    SHAP.React.createElement(SHAP.AdditiveForceArrayVisualizer, {"outNames": ["f(x)"], "baseValue": -0.16673818230628967, "link": "identity", "featureNames": ["Zphonedeletion", "Zonsetrime", "ZSpoonerism", "ZRANdigit", "ZRANpic", "ZRANdice", "ZRANcolor", "ZDigitSpan", "ZMA", "ZPseudoC", "ZIll-formed component", "ZIllegal position", "ZBW_drawings", "ZNumberACC", "ZSymbolACC", "ZColorACC", "ZChineseACC"], "explanations": [{"outValue": 0.9751862287521362, "simIndex": 7.0, "features": {"0": {"effect": -0.17016735672950745, "value": 0.3261019471165864}, "1": {"effect": 0.028505003079771996, "value": 0.26619837761671084}, "2": {"effect": 0.633231520652771, "value": -1.5068972925127981}, "3": {"effect": 0.3299897015094757, "value": -0.6257369174402598}, "4": {"effect": 0.1007087454199791, "value": 0.006364112990777936}, "5": {"effect": 0.029408391565084457, "value": 0.15297216219123136}, "6": {"effect": -0.3118661642074585, "value": 1.093265620479921}, "7": {"effect": 0.01758134551346302, "value": -0.10492809846955936}, "8": {"effect": 0.6980631351470947, "value": -1.0871357835106705}, "9": {"effect": 0.0003657782217487693, "value": 0.052482078493819646}, "10": {"effect": 0.008312373422086239, "value": -1.0713221810995373}, "11": {"effect": 4.821363836526871e-05, "value": 0.20033367368870683}, "12": {"effect": 0.0015729444567114115, "value": 0.29697032749782926}, "13": {"effect": -0.12615853548049927, "value": 1.1725944065761877}, "14": {"effect": 0.008276348933577538, "value": 1.726096000970435}, "15": {"effect": 0.031189648434519768, "value": 2.0472012341849215}, "16": {"effect": -0.1371367871761322, "value": 1.6685711679668367}}}, {"outValue": 1.566375970840454, "simIndex": 10.0, "features": {"0": {"effect": 0.6171315908432007, "value": -1.8975079869375957}, "1": {"effect": 0.08686800301074982, "value": -0.7459929816923448}, "2": {"effect": 0.6306580305099487, "value": -2.16523337312765}, "3": {"effect": 0.18244346976280212, "value": -0.7751757748181646}, "4": {"effect": -0.16302399337291718, "value": 0.3531436664827868}, "5": {"effect": 0.05022929981350899, "value": 0.5303210861249289}, "6": {"effect": 0.037639766931533813, "value": -0.02607880949466546}, "7": {"effect": 0.057121556252241135, "value": -1.183243617792014}, "8": {"effect": 0.4417942464351654, "value": -0.7655661383888168}, "9": {"effect": 0.020379917696118355, "value": -0.9078671869557593}, "10": {"effect": -0.011763154529035091, "value": -0.0492554730169805}, "11": {"effect": -0.02361646108329296, "value": 0.2363576761573915}, "12": {"effect": -0.0033484383020550013, "value": 0.33287248525685276}, "13": {"effect": -0.07737512141466141, "value": 0.8969351130507712}, "14": {"effect": -0.002656846307218075, "value": -0.38242845530482816}, "15": {"effect": 0.00991529319435358, "value": 0.6511155787808764}, "16": {"effect": -0.1192830353975296, "value": 1.6685711679668367}}}, {"outValue": -1.0432099103927612, "simIndex": 5.0, "features": {"0": {"effect": -0.2644018530845642, "value": 0.34518}, "1": {"effect": 0.0014227493666112423, "value": 0.36447}, "2": {"effect": -0.1800813525915146, "value": 1.1120957326535137}, "3": {"effect": -0.12759561836719513, "value": 0.77241}, "4": {"effect": -0.19730225205421448, "value": 0.635}, "5": {"effect": 0.062251996248960495, "value": 0.20915}, "6": {"effect": -0.12723664939403534, "value": 1.34422}, "7": {"effect": -0.015486192889511585, "value": 0.19438745647582123}, "8": {"effect": 0.005782981403172016, "value": 0.0087877098176451}, "9": {"effect": 0.11003578454256058, "value": -0.8556836224013781}, "10": {"effect": 0.038458291441202164, "value": 0.5928156678526247}, "11": {"effect": 0.027903759852051735, "value": 0.5900174943222254}, "12": {"effect": -0.00437256321310997, "value": 0.43661917298393804}, "13": {"effect": -0.07962048053741455, "value": 1.00925}, "14": {"effect": 0.013171927072107792, "value": 0.48007}, "15": {"effect": 0.019717548042535782, "value": 0.40663}, "16": {"effect": -0.1591198593378067, "value": 1.3011}}}, {"outValue": 1.6413278579711914, "simIndex": 9.0, "features": {"0": {"effect": 0.5212088227272034, "value": -2.79784}, "1": {"effect": 0.08091815561056137, "value": -3.35853}, "2": {"effect": 0.5443305373191833, "value": -1.411864823321232}, "3": {"effect": 0.007009697612375021, "value": 0.14519}, "4": {"effect": -0.10269992798566818, "value": 0.53673}, "5": {"effect": 0.06075523793697357, "value": 0.60544}, "6": {"effect": 0.23345081508159637, "value": -1.37612}, "7": {"effect": 0.0625806376338005, "value": -0.6922652583026043}, "8": {"effect": 0.6293450593948364, "value": -2.1713585215312414}, "9": {"effect": 0.005611044354736805, "value": 0.30503152305341685}, "10": {"effect": 0.0080135278403759, "value": 0.7141176241456568}, "11": {"effect": -0.002111987676471472, "value": 0.67388031941702}, "12": {"effect": -0.0076232850551605225, "value": 0.4408441369836165}, "13": {"effect": -0.19376398622989655, "value": 1.37161}, "14": {"effect": 0.010308243334293365, "value": 1.05939}, "15": {"effect": 0.03165701776742935, "value": 2.07666}, "16": {"effect": -0.08092367649078369, "value": 0.83899}}}, {"outValue": 1.5684386491775513, "simIndex": 3.0, "features": {"0": {"effect": -0.02175315096974373, "value": -0.31539}, "1": {"effect": 0.05769079178571701, "value": -0.897}, "2": {"effect": 0.6984411478042603, "value": -1.6213172971106455}, "3": {"effect": 0.3376517593860626, "value": -0.1821}, "4": {"effect": 0.4174260199069977, "value": -0.8155}, "5": {"effect": 0.010931978933513165, "value": 0.28975}, "6": {"effect": 0.4304549992084503, "value": -0.7552}, "7": {"effect": 0.02314276620745659, "value": -0.25365444579026625}, "8": {"effect": 0.0601569227874279, "value": -0.4989538336075067}, "9": {"effect": 0.0010866050142794847, "value": 1.0104889927756688}, "10": {"effect": 0.01937810890376568, "value": 0.6409298550141405}, "11": {"effect": -0.00871013943105936, "value": 0.6409298550141405}, "12": {"effect": -0.009457609616219997, "value": 0.4408441369836165}, "13": {"effect": -0.1593945026397705, "value": 1.37161}, "14": {"effect": 0.004957747645676136, "value": -1.33127}, "15": {"effect": -0.00023575068917125463, "value": 0.27548}, "16": {"effect": -0.12659095227718353, "value": 0.83899}}}, {"outValue": 0.6033339500427246, "simIndex": 8.0, "features": {"0": {"effect": -0.13668063282966614, "value": -0.03957}, "1": {"effect": -0.05994931608438492, "value": 0.47052}, "2": {"effect": 0.4995191991329193, "value": -1.2024123495318184}, "3": {"effect": 0.020392032340168953, "value": 0.20765}, "4": {"effect": -0.035254236310720444, "value": 0.54552}, "5": {"effect": 0.030367134138941765, "value": 0.63975}, "6": {"effect": -0.20953181385993958, "value": 0.86097}, "7": {"effect": -0.149901881814003, "value": 3.693842866820776}, "8": {"effect": 0.9191365242004395, "value": -1.2422448060180555}, "9": {"effect": -0.002891330514103174, "value": 0.1720760241320885}, "10": {"effect": 0.005981753580272198, "value": 0.6409298550141405}, "11": {"effect": 0.00047678546980023384, "value": 0.6409298550141405}, "12": {"effect": -0.003928192891180515, "value": 0.4408441369836165}, "13": {"effect": -0.05846762657165527, "value": 0.78279}, "14": {"effect": 0.032455723732709885, "value": -1.33127}, "15": {"effect": 0.00767158018425107, "value": -0.08476}, "16": {"effect": -0.08932355791330338, "value": 1.40497}}}, {"outValue": 1.1296137571334839, "simIndex": 1.0, "features": {"0": {"effect": 0.9098798036575317, "value": -2.565620718710083}, "1": {"effect": -0.10064221918582916, "value": 1.07153444825783}, "2": {"effect": -0.23462314903736115, "value": 0.6370202367015267}, "3": {"effect": 0.32970762252807617, "value": -0.513255876838312}, "4": {"effect": 0.4629468321800232, "value": -0.7316717915228289}, "5": {"effect": 0.034102290868759155, "value": 0.6636838483430268}, "6": {"effect": 0.2645060122013092, "value": -1.5938497622114547}, "7": {"effect": 0.020307239145040512, "value": -0.2789572379863285}, "8": {"effect": -0.06405647844076157, "value": 0.08518995864072422}, "9": {"effect": 0.00630118977278471, "value": 0.5703268225747656}, "10": {"effect": -0.01845550164580345, "value": -0.0492554730169805}, "11": {"effect": -0.020643629133701324, "value": 0.2363576761573915}, "12": {"effect": -0.005720329470932484, "value": 0.33287248525685276}, "13": {"effect": -0.15516671538352966, "value": 0.8969351130507712}, "14": {"effect": 0.028044672682881355, "value": 1.726096000970435}, "15": {"effect": -0.013786724768579006, "value": 0.30209416492986574}, "16": {"effect": -0.14634902775287628, "value": 1.1338495279631007}}}, {"outValue": 1.5149484872817993, "simIndex": 2.0, "features": {"0": {"effect": 0.6398298740386963, "value": -2.899677084596327}, "1": {"effect": 0.03479769825935364, "value": -1.1094984676823798}, "2": {"effect": 0.46260878443717957, "value": -2.16523337312765}, "3": {"effect": 0.39253678917884827, "value": -1.0394986076418686}, "4": {"effect": 0.32930827140808105, "value": -1.152174753148743}, "5": {"effect": 0.05229026451706886, "value": -0.3028039343198932}, "6": {"effect": 0.2539529800415039, "value": -1.4951098141747072}, "7": {"effect": 0.03450145944952965, "value": -0.7311004278891712}, "8": {"effect": -0.26520901918411255, "value": 1.4558008178248205}, "9": {"effect": -0.011910642497241497, "value": 1.3094238273400278}, "10": {"effect": -0.0032754170242697, "value": -1.8365254939188338}, "11": {"effect": -0.015684176236391068, "value": 0.2363576761573915}, "12": {"effect": -0.006035727448761463, "value": 0.33287248525685276}, "13": {"effect": -0.11817865818738937, "value": 0.8969351130507712}, "14": {"effect": 0.015179494395852089, "value": 1.0232545155453472}, "15": {"effect": -0.0006980642210692167, "value": 0.6511155787808764}, "16": {"effect": -0.11232727766036987, "value": 0.8664887079612325}}}, {"outValue": 0.20973947644233704, "simIndex": 4.0, "features": {"0": {"effect": -0.20287153124809265, "value": 0.6808272626321196}, "1": {"effect": 0.009416937828063965, "value": 0.6246647187898117}, "2": {"effect": -0.26970651745796204, "value": 0.9325194007478333}, "3": {"effect": 0.41696518659591675, "value": -0.44621131604619285}, "4": {"effect": 0.042166031897068024, "value": -0.2543576238902199}, "5": {"effect": -0.004405890125781298, "value": 1.2688383842045121}, "6": {"effect": 0.6134768724441528, "value": -0.7392555648635367}, "7": {"effect": -0.07881335914134979, "value": 2.255954117095516}, "8": {"effect": -0.13638350367546082, "value": 1.0337950090860777}, "9": {"effect": 0.02538333088159561, "value": 0.45636589994625143}, "10": {"effect": 0.09705702215433121, "value": 1.120863091208189}, "11": {"effect": 0.001417485997080803, "value": 0.20033367368870683}, "12": {"effect": 0.015603321604430676, "value": 0.29697032749782926}, "13": {"effect": -0.11730673164129257, "value": 1.1725944065761877}, "14": {"effect": -0.008560393936932087, "value": 0.3204130301202595}, "15": {"effect": 0.0556349977850914, "value": 1.3491584064828994}, "16": {"effect": -0.082595594227314, "value": 0.33176706795749655}}}, {"outValue": 0.2230202853679657, "simIndex": 6.0, "features": {"0": {"effect": -0.21525755524635315, "value": 0.44088657426610983}, "1": {"effect": 0.08769620209932327, "value": -0.3824874957023099}, "2": {"effect": -0.32440948486328125, "value": 0.6370202367015267}, "3": {"effect": -0.13639159500598907, "value": 0.58728828164584}, "4": {"effect": -0.10930651426315308, "value": 0.36821054784397583}, "5": {"effect": 0.02937287464737892, "value": 0.5162003230665421}, "6": {"effect": 0.28615236282348633, "value": -0.38812528562940307}, "7": {"effect": 0.13174328207969666, "value": -0.7311004278891712}, "8": {"effect": 0.8120863437652588, "value": -1.616322235418358}, "9": {"effect": 0.007750826422125101, "value": 0.5703268225747656}, "10": {"effect": 0.017107540741562843, "value": 0.8443795374339457}, "11": {"effect": -0.01139059942215681, "value": 0.2363576761573915}, "12": {"effect": 0.03859071806073189, "value": -0.7239976554336676}, "13": {"effect": -0.03064448945224285, "value": 0.3456165259999389}, "14": {"effect": 0.02388833463191986, "value": 1.3746752582578914}, "15": {"effect": -0.006093072704970837, "value": -0.04692724892114575}, "16": {"effect": -0.21113669872283936, "value": 1.1338495279631007}}}], "plot_cmap": "RdBu", "ordering_keys": null, "ordering_keys_time_format": null}),
    document.getElementById('iXQ27Y3IR19KF7KNNES7X')
  );
</script>




```python

```
