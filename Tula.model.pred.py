#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system("cp /content/drive/MyDrive/Colab/Data/'Готовые задачи'/Калининград/participants/train/train.csv ./ ")
get_ipython().system("cp /content/drive/MyDrive/Colab/Data/'Готовые задачи'/Калининград/participants/test/test.csv ./ ")


# In[ ]:


#Установка catboost
get_ipython().system('pip install catboost')


# In[8]:


#import необходимых модулей

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[661]:


#Считывание данных в DataFrame 

train = pd.read_csv('train.csv', sep=';', index_col=None, dtype={'PATIENT_SEX':str, 'MKB_CODE':str, 'ADRES':str, 'VISIT_MONTH_YEAR':str, 'AGE_CATEGORY':str, 'PATIENT_ID_COUNT':int})
test = pd.read_csv('test.csv', sep=';', index_col=None, dtype={'PATIENT_SEX':str, 'MKB_CODE':str, 'ADRES':str, 'VISIT_MONTH_YEAR':str, 'AGE_CATEGORY':str})


# In[622]:


#Отделение меток от данных

X = train[['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY']]
y = train[['PATIENT_ID_COUNT']]


# In[623]:


#Разделение на train/test для локального тестирования

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


# In[12]:


#Создание объекта данных Pool, плюсы: возможность указать какие признаки являются категориальными

pool_train = Pool(X_train, y_train, cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])
pool_test = Pool(X_test, cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])


# In[13]:


#Объявление CatBoostRegressor и обучение

model = CatBoostRegressor(task_type='GPU')
model.fit(pool_train)


# In[14]:


#Получение ответов модели на тестовой выборке в локальном тестировании 

y_pred = model.predict(pool_test)


# In[15]:


#На локальном тестировании модель выдаёт такой результат

print("Значение метрики R2 на test: ", r2_score(y_test, y_pred))


# In[16]:


#Формируем sample_solution. В обучении используется весь train, ответы получаем на test

pool_train_solution = Pool(X, y, cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])
pool_test_solution = Pool(test, cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])

model_solution = CatBoostRegressor(task_type='GPU')
model_solution.fit(pool_train_solution)


# In[17]:


#Получение ответов

y_pred_solution = model.predict(pool_test_solution)


# In[18]:


#Вот так они выглядят

y_pred_solution.astype(int)


# In[19]:


#Формируем sample_solution для отправки на платформу

test['PATIENT_ID_COUNT'] = y_pred_solution.astype(int)


# In[20]:


#Сохраняем в csv файл
 
test.to_csv('sample_solution.csv', sep=';', index=None)


# In[23]:


# Отрицательные значения
test[test['PATIENT_ID_COUNT']<0].sort_values(by='PATIENT_ID_COUNT')


# In[28]:


# В тестовых данных нет отрицательных значений, поэтому замена их на ноль немного улучшит метрику
test_non_negative = test.copy()
test_non_negative.loc[test_non_negative['PATIENT_ID_COUNT']<0, 'PATIENT_ID_COUNT'] = 0
test_non_negative.sort_values(by='PATIENT_ID_COUNT', ascending=True)


# In[29]:


test_non_negative.to_csv('sample_solution_non_negative.csv', sep=';', index=None)


# ## Рекомендованный вариант с R2 = 0,76

# In[51]:


test


# In[57]:


(
    test.drop(['PATIENT_ID_COUNT'], axis=1).
    merge(
        (train[train['VISIT_MONTH_YEAR'].isin(['01.22', '02.22', '03.22'])]
        .groupby(['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY'], as_index=False)
        ['PATIENT_ID_COUNT'].mean()
        ),
        on=['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY'],
        how='left'
        )
    .fillna(1)
    .astype({'PATIENT_ID_COUNT': int})
    .to_csv('baseline_3m.csv', sep=';', index=None)
)

# При загрузке в систему выдало: 0.668303


# ## Данные за прошлый месяц для прогноза

# In[58]:


(
    test.drop(['PATIENT_ID_COUNT'], axis=1).
    merge(
        (train[train['VISIT_MONTH_YEAR'].isin(['03.22'])]
        .groupby(['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY'], as_index=False)
        ['PATIENT_ID_COUNT'].mean()
        ),
        on=['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY'],
        how='left'
        )
    .fillna(1)
    .astype({'PATIENT_ID_COUNT': int})
    .to_csv('baseline_last_month_pred.csv', sep=';', index=None)
)

# Это оказалось лучшим из найденных решением. В системе выдало 0.947781


# ## Сравнение R2-метрик полученных предварительных вариантов

# In[65]:


print(
    "Значение метрики R2 между средним за 3 последних месяца и исходным baseline:",
    r2_score(
        pd.read_csv('baseline_3m.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True),
        pd.read_csv('sample_solution.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True)
    ),
    sep='\n'
)


# In[66]:


print(
    "Значение метрики R2 между средним за 3 последних месяца и неотрицательным baseline:",
    r2_score(
        pd.read_csv('baseline_3m.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True),
        pd.read_csv('sample_solution_non_negative.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True)
    ),
    sep='\n'
)


# In[64]:


# TODO: можно вынести в отдельную функцию, т.к. этот вызов уже использовался 3 раза.
print(
    "Значение метрики R2 между средним за 3 последних месяца и последним месяцем:",
    r2_score(
        pd.read_csv('baseline_3m.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True),
        pd.read_csv('baseline_last_month_pred.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True)
    ),
    sep='\n'
)


# ## CatBoost данных после коронавирусного изменения тренда до предпоследнего месяца и проверка на последнем месяце данных

# Из EDA было найдено изменение тренда общего числа регистрируемых заболеваний с уменьшения на увеличение после 06.20.
# Видимо связано с появлением нового фактора - коронавируса с карантином.
# 
# Далее были предприянты попытки некоторой настройки гиперпараметров модели CatBoost с разными выборками данных. В основном выполнялась кросс-валидация данных обучения разной продолжительности с предсказанием за следующим за этой выборкой месяцем в качестве тестовой выборки.

# In[ ]:


#Отделение меток от данных

X = train[['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY']]
y = train[['PATIENT_ID_COUNT']]


# In[98]:


# Список месяцев после коронавирусного изменения тренда
after_covid_turn_month_list = [d.strftime('%m.%y') for d in pd.date_range(start='2020-06-01', end='2022-02-01', freq='MS')]
after_covid_turn_month_list


# In[108]:


# Месяц для проверки предсказаний по модели, обученной данными списка месяцев выше.
after_covid_pred_month = '03.22'


# In[109]:


#Создание объекта данных Pool, плюсы: возможность указать какие признаки являются категориальными

pool_after_covid_train = Pool(X[X['VISIT_MONTH_YEAR'].isin(after_covid_turn_month_list)], y[X['VISIT_MONTH_YEAR'].isin(after_covid_turn_month_list)], cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])
pool_after_covid_test = Pool(X[X['VISIT_MONTH_YEAR'].isin([after_covid_pred_month])], cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])


# In[110]:


#Объявление CatBoostRegressor и обучение
model_after_covid = CatBoostRegressor(task_type='GPU')
model_after_covid.fit(pool_after_covid_train)


# In[111]:


#Получение ответов модели на тестовой выборке в локальном тестировании 
y_after_covid_pred = model_after_covid.predict(pool_after_covid_test)


# In[112]:


#На локальном тестировании модель выдаёт такой результат
print("Значение метрики модели после коронавируса R2 на test: ",
      r2_score(y[X['VISIT_MONTH_YEAR'].isin([after_covid_pred_month])],
               y_after_covid_pred))


# In[115]:


y_after_covid_pred[y_after_covid_pred<0] = 0


# In[116]:


#На локальном тестировании модель выдаёт такой результат
print("Значение метрики модели после коронавируса R2 на test: ",
      r2_score(y[X['VISIT_MONTH_YEAR'].isin([after_covid_pred_month])],
               y_after_covid_pred))


# In[120]:


#На локальном тестировании модель выдаёт такой результат
print("Значение метрики модели после коронавируса R2 на test: ",
      r2_score(y[X['VISIT_MONTH_YEAR'].isin([after_covid_pred_month])],
               y_after_covid_pred.astype(int)))


# In[128]:


#На локальном тестировании модель выдаёт такой результат
print("Значение метрики модели после коронавируса R2 на test: ",
      r2_score(y[X['VISIT_MONTH_YEAR'].isin([after_covid_pred_month])],
               y_after_covid_pred.astype(int)-6))


# In[133]:


after_covid_turn_month_list[-1]


# In[134]:


# То же самое со сдвигом на месяц
pool_after_covid_train_2 = Pool(X[X['VISIT_MONTH_YEAR'].isin(after_covid_turn_month_list[:-1])], y[X['VISIT_MONTH_YEAR'].isin(after_covid_turn_month_list[:-1])], cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])
pool_after_covid_test_2 = Pool(X[X['VISIT_MONTH_YEAR'].isin([after_covid_turn_month_list[-1]])], cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])


# In[135]:


model_after_covid_2 = CatBoostRegressor(task_type='GPU')
model_after_covid_2.fit(pool_after_covid_train_2)


# In[136]:


#Получение ответов модели на тестовой выборке в локальном тестировании 
y_after_covid_pred_2 = model_after_covid_2.predict(pool_after_covid_test_2)


# In[138]:


#На локальном тестировании модель выдаёт такой результат
print("Значение метрики модели после коронавируса -1 месяц с конца R2 на test: ",
      r2_score(y[X['VISIT_MONTH_YEAR'].isin([after_covid_turn_month_list[-1]])],
               y_after_covid_pred_2.astype(int)))


# ## Обобщение в цикле попыток CatBoost для после коронавирусного тренда

# In[185]:


# Список месяцев после коронавирусного изменения тренда
after_covid_turn_month_list = [d.strftime('%m.%y') for d in pd.date_range(start='2020-06-01', end='2022-03-01', freq='MS')]
after_covid_turn_month_list


# In[230]:


"""
Обучается на последовательности месяцев.
Выдаёт предсказание по следуюущему за этой последовательностью месяцу
"""
def CatBoost_by_subsets(X, y, train_month_list, test_month, prefix_file_name='sample_subset_solution'):
    pool_month_subset_train = Pool(
        X[X['VISIT_MONTH_YEAR'].isin(train_month_list)],
        y[X['VISIT_MONTH_YEAR'].isin(train_month_list)],
        cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY']
    )
    pool_month_test = Pool(
        X[X['VISIT_MONTH_YEAR'].isin([test_month])],
        cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY']
    )
    
    #Объявление CatBoostRegressor и обучение
    model = CatBoostRegressor(task_type='GPU')
    model.fit(pool_month_subset_train)

    #Получение ответов модели на тестовой выборке в локальном тестировании 
    y_pred = model.predict(pool_month_test)

    r2_test_month = r2_score(y[X['VISIT_MONTH_YEAR'].isin([test_month])],
                             y_pred.astype(int))
        
    print("Значение метрики модели после коронавируса R2 на test: ", r2_test_month)
    
    #Формируем sample_solution для отправки на платформу
    # NOTE: здесь была ошибка y_pred_solution -> вместо y_pred. В файлах так будет записано
    test['PATIENT_ID_COUNT'] = y_pred.astype(int)
    
    #Сохраняем в csv файл
    test.to_csv(prefix_file_name + '.' + test_month + '.csv', sep=';', index=None)
    
    # NOTE: Замена отрицательных значений на 0 или среднее по категории улучшает результат
    
    return r2_test_month


# In[195]:


r2_score_subsets_by_month = {}
for month_num in range(7, 23):
    train_month_list = after_covid_turn_month_list[:month_num]
    test_month = after_covid_turn_month_list[month_num-1]
    
    r2_score_subsets_by_month[test_month] = CatBoost_by_subsets(X, y, train_month_list, test_month)
    print(r2_score_subsets_by_month[test_month])


# In[226]:


pd.DataFrame.from_dict(
    r2_score_subsets_by_month, orient="index", columns=['R2 Score']
).to_csv(
    'r2_score_subsets_by_months.csv', sep=';', index_label='VISIT_MONTH_YEAR.PRED'
)
print(r2_score_subsets_by_month)


# In[197]:


import matplotlib.pyplot as plt


# In[246]:


plt.figure(figsize=(12, 6))
plt.plot(range(7, 23), r2_score_subsets_by_month.values(), marker='o')
plt.title('Зависимость метрики $R^2$ предсказания от количества месяцев в прошлом для обучения')
plt.xlabel(r'Количество месяцев' + '\n' + 'в прошлое для обучения')
plt.ylabel(r'Значение метрики $R^2$' + '\n' + 'для полученного предсказания')
plt.grid()
plt.show()

# Хорошо видны пики локальных максимумов на 12 и 18 месяцев (1 и 1,5 года)


# ## Подбираем период обучения для предсказания последнего месяца

# In[240]:


r2_score_subsets_for_final_month = []
# Теперь ищем только для последнего месяца
test_month = after_covid_turn_month_list[-1]

for month_num in range(2, 23):
    train_month_list = after_covid_turn_month_list[-1-month_num:-1]
    
    r2_score_subsets_for_final_month.append(
        CatBoost_by_subsets(X, y, train_month_list, 
                            test_month, 'for_final_month.subsets_' + str(month_num)
                           )
    )


# In[241]:


print(r2_score_subsets_for_final_month)


# In[254]:


pd.DataFrame(
    r2_score_subsets_for_final_month, columns=['R2 Score']
).to_csv(
    'r2_score_subsets_for_final_month.csv', sep=';', index_label='VISIT_MONTH_YEAR_SHIFT_FROM_2'
)
print(r2_score_subsets_for_final_month)


# In[245]:


plt.figure(figsize=(12, 6))
plt.plot(range(2, 23), r2_score_subsets_for_final_month, marker='o')
plt.title('Зависимость метрики $R^2$ предсказания от количества месяцев в прошлом для обучения')
plt.xlabel(r'Количество месяцев' + '\n' + 'в прошлое для обучения')
plt.ylabel(r'Значение метрики $R^2$' + '\n' + 'для полученного предсказания последнего месяца')
plt.grid()
plt.show()


# ## По двум прошлым проходам обучение на 18 месяцев назад дало наибольшую точность. Сделаем такое предсказание.

# In[265]:


months_18_list = after_covid_turn_month_list[-18:]


# In[264]:


#test.drop(['PATIENT_ID_COUNT'], axis=1)
test


# In[266]:


X


# In[267]:


y


# In[269]:


pool_18_months_train = Pool(
    X[X['VISIT_MONTH_YEAR'].isin(months_18_list)],
    y[X['VISIT_MONTH_YEAR'].isin(months_18_list)],
    cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY']
)

pool_18_months_test = Pool(
    #test.drop(['PATIENT_ID_COUNT'], axis=1),
    test,
    cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY']
)

#Объявление CatBoostRegressor и обучение
model = CatBoostRegressor(task_type='GPU')
model.fit(pool_18_months_train)

#Получение ответов модели на тестовой выборке в локальном тестировании 
y_pred = model.predict(pool_18_months_test)

# Оценим точность по baseline среднего за 3 последних месяца
print(
    "Значение метрики R2 между средним за 3 последних месяца и предсказанием CatBoost по 18 месяцам:",
    r2_score(
        pd.read_csv('baseline_3m.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True),
        y_pred.astype(int)
    ),
    sep='\n'
)
    
#Формируем sample_solution для отправки на платформу
test['PATIENT_ID_COUNT'] = y_pred.astype(int)
    
#Сохраняем в csv файл
test.to_csv('out.by_18_last_months.csv', sep=';', index=None)


# In[274]:


# Отрицательные значения заменяем на 0
# NOTE: может быть 1 или среднее значение в категории?
test.loc[test.PATIENT_ID_COUNT<0, 'PATIENT_ID_COUNT'] = 0


# In[276]:


test.loc[test.PATIENT_ID_COUNT<0]


# In[277]:


test


# In[283]:


print(
    "Значение метрики R2 между средним за 3 последних месяца и неотрицательным предсказанием CatBoost по 18 месяцам:",
    r2_score(
        pd.read_csv('baseline_3m.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True),
        test['PATIENT_ID_COUNT']
    ),
    sep='\n'
)


# In[278]:


test.to_csv('out.by_18_last_months.non_negative.csv', sep=';', index=None)
# В системе выдало оценку 0.588817


# # Попытки дальнейшего обучения CatBoost с присоединёнными данными либо с расшифрованными исходными категориями.

# In[284]:


# TODO: Попробовать присоединить население по городам вместо названий (возможно в процентах от всего населения области)
# TODO: Может быть категории возрастов заменить на числовые диапазоны или перевести в долю от всего населения области
# TODO: Попробовать присоединить температуру (мин., макс., средн.) по месяцам


# In[293]:


test.ADRES.unique()


# In[ ]:


# @link https://ru.wikipedia.org/wiki/%D0%93%D0%BE%D1%80%D0%BE%D0%B4%D1%81%D0%BA%D0%B8%D0%B5_%D0%BD%D0%B0%D1%81%D0%B5%D0%BB%D1%91%D0%BD%D0%BD%D1%8B%D0%B5_%D0%BF%D1%83%D0%BD%D0%BA%D1%82%D1%8B_%D0%9A%D0%B0%D0%BB%D0%B8%D0%BD%D0%B8%D0%BD%D0%B3%D1%80%D0%B0%D0%B4%D1%81%D0%BA%D0%BE%D0%B9_%D0%BE%D0%B1%D0%BB%D0%B0%D1%81%D1%82%D0%B8
# Население основных городов Калининградской области
# Остальным можно поставить среднее население по остатку
# @link https://ru.wikipedia.org/wiki/%D0%9D%D0%B0%D1%81%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5_%D0%9A%D0%B0%D0%BB%D0%B8%D0%BD%D0%B8%D0%BD%D0%B3%D1%80%D0%B0%D0%B4%D1%81%D0%BA%D0%BE%D0%B9_%D0%BE%D0%B1%D0%BB%D0%B0%D1%81%D1%82%D0%B8
# 1027678 чел. - Население области на 2022 год


# In[445]:


main_cities_url_wiki = 'https://ru.wikipedia.org/wiki/%D0%93%D0%BE%D1%80%D0%BE%D0%B4%D1%81%D0%BA%D0%B8%D0%B5_%D0%BD%D0%B0%D1%81%D0%B5%D0%BB%D1%91%D0%BD%D0%BD%D1%8B%D0%B5_%D0%BF%D1%83%D0%BD%D0%BA%D1%82%D1%8B_%D0%9A%D0%B0%D0%BB%D0%B8%D0%BD%D0%B8%D0%BD%D0%B3%D1%80%D0%B0%D0%B4%D1%81%D0%BA%D0%BE%D0%B9_%D0%BE%D0%B1%D0%BB%D0%B0%D1%81%D1%82%D0%B8'
kal_reg_pop_url_wiki = 'https://ru.wikipedia.org/wiki/%D0%9D%D0%B0%D1%81%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5_%D0%9A%D0%B0%D0%BB%D0%B8%D0%BD%D0%B8%D0%BD%D0%B3%D1%80%D0%B0%D0%B4%D1%81%D0%BA%D0%BE%D0%B9_%D0%BE%D0%B1%D0%BB%D0%B0%D1%81%D1%82%D0%B8'

# Web Scraping
main_cities_table = pd.read_html(main_cities_url_wiki, attrs={'class': 'standard sortable'})
kal_reg_pop_table = pd.read_html(kal_reg_pop_url_wiki, attrs={'class': 'standard'})


# In[446]:


main_cities_table


# In[447]:


type(main_cities_table)


# In[448]:


type(kal_reg_pop_table)


# In[449]:


main_cities_table[0].iloc[:, 5] = main_cities_table[0].iloc[:, 5].apply(
    lambda x: ''.join(ch for ch in x[1:-3] if ch.isalnum())
)


# In[450]:


main_cities_table[0]


# In[451]:


main_cities_table[1].iloc[:, 5] = main_cities_table[1].iloc[:, 5].apply(
    lambda x: ''.join(ch for ch in x[1:-3] if ch.isalnum())
)


# In[452]:


main_cities_table[1]


# In[453]:


dict_pop = dict(zip(list(main_cities_table[0].iloc[:, 1]), list(main_cities_table[0].iloc[:, 5].astype(int))))


# In[454]:


dict_pop_2 = dict(zip(list(main_cities_table[1].iloc[:, 1]), list(main_cities_table[1].iloc[:, 5].astype(int))))


# In[455]:


dict_pop.update(dict_pop_2)


# In[405]:


# Convert to int
dict_pop.update((key, int(val)) for key, val in dict_pop.items())
# Already done above in astype(int)


# In[456]:


main_cities_total_pop = sum(dict_pop.values())


# In[457]:


dict_pop


# In[458]:


"""
По названию города Калиниградской области выдаёт его население.
Для самых больших городов население прописано в hard code - словаре.
Для отсутствующих берётся среднее население по остатку.
"""
def get_pop_by_city(city_name, dict_pop, region_pop=1027678, num_remain_city=95):
    if city_name in dict_pop:
        city_pop = dict_pop[city_name]
    else:
        main_cities_total_pop = sum(dict_pop.values())
        city_pop = round((region_pop - main_cities_total_pop) / num_remain_city)
    return city_pop


# In[459]:


for city in train.ADRES.unique():
    print(city, get_pop_by_city(city, dict_pop), sep=': ')


# In[461]:


(1027678-sum(dict_pop.values()))/95


# In[ ]:


# Присоединяем столбец данных с населением


# In[464]:


train['POPULATION'] = train['ADRES'].apply(lambda city_name: get_pop_by_city(city_name, dict_pop))


# In[466]:


train


# In[471]:


X['POPULATION'] = X.loc[:, 'ADRES'].apply(lambda city_name: get_pop_by_city(city_name, dict_pop))


# In[472]:


X


# ## Повторяем для данных с присоединенным населением

# In[549]:


pool_18_months_train = Pool(
    X[X['VISIT_MONTH_YEAR'].isin(months_18_list)],
    y[X['VISIT_MONTH_YEAR'].isin(months_18_list)],
    cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY', 'POPULATION']
)

test['POPULATION'] = test.loc[:, 'ADRES'].apply(lambda city_name: get_pop_by_city(city_name, dict_pop))

pool_18_months_test = Pool(
    test.drop(['PATIENT_ID_COUNT'], axis=1),
    #test,
    cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY', 'POPULATION']
)


# In[550]:


#Объявление CatBoostRegressor и обучение
model = CatBoostRegressor(task_type='GPU')
model.fit(pool_18_months_train)

#Получение ответов модели на тестовой выборке в локальном тестировании 
y_pred = model.predict(pool_18_months_test)

# Оценим точность по baseline среднего за 3 последних месяца
print(
    "Значение метрики R2 между средним за 3 последних месяца и предсказанием CatBoost по 18 месяцам + население:",
    r2_score(
        pd.read_csv('baseline_3m.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True),
        y_pred.astype(int)
    ),
    sep='\n'
)
    
#Формируем sample_solution для отправки на платформу
test['PATIENT_ID_COUNT'] = y_pred.astype(int)
    
#Сохраняем в csv файл
test.to_csv('out.by_18_last_months.with_popualtion.csv', sep=';', index=None)


# In[552]:


test.iloc[:, :6]


# In[553]:


# Отрицательные значения заменяем на 0
# NOTE: может быть 1 или среднее значение в категории?
test.loc[test.PATIENT_ID_COUNT<0, 'PATIENT_ID_COUNT'] = 0


# In[554]:


(test['PATIENT_ID_COUNT']<0).any()


# In[555]:


# Оценим точность по baseline среднего за 3 последних месяца
print(
    "Значение метрики R2 между средним за 3 последних месяца и предсказанием CatBoost по 18 месяцам + население, неотрицательные:",
    r2_score(
        pd.read_csv('baseline_3m.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True),
        test['PATIENT_ID_COUNT']
    ),
    sep='\n'
)


# In[556]:


#Сохраняем в csv файл
test.iloc[:, :6].to_csv('out.by_18_last_months.with_popualtion.non_negative.csv', sep=';', index=None)
# В системе выдало точность всего 0.482687. Наименьшее для всех загруженных решений


# In[488]:


# Добавление температуры в Калининграде по месяцам


# In[489]:


get_ipython().system('pip install meteostat')


# In[500]:


from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Stations, Monthly

# Time period
start = datetime(2018, 1, 15)
end = datetime(2022, 4, 15)

#data = Monthly('10637', start, end) # Это из премера для Ванкувера.
# Нам нужен Калининград. Берём его значения температуры для всей области. Скорее всего неплохое приближение.
# Kaliningrad weather station id 26702
# latitude: 54.7167
# longitude: 20.55
data = Monthly('26702', start, end)
data = data.fetch()

# Graph with average, minimum and maximum temperatures
data.plot(y=['tavg', 'tmin', 'tmax'])
plt.show()


# In[501]:


data


# In[502]:


data.tavg.astype(int)


# In[508]:


data.index.strftime('%m.%y')


# In[514]:


t_avg_kaliningrad = dict(zip(data.index.strftime('%m.%y'), data.tavg.astype(int)))


# In[516]:


# Калининград, средняя температура по месяцам
t_avg_kaliningrad


# In[526]:


# Присоединяем данные по температуре
train['TEMP_AVG'] = train['VISIT_MONTH_YEAR'].apply(lambda month_year: t_avg_kaliningrad[month_year])
test['TEMP_AVG'] = test['VISIT_MONTH_YEAR'].apply(lambda month_year: t_avg_kaliningrad[month_year])


# In[530]:


# Присоединяем данные по температуре
X['TEMP_AVG'] = X['VISIT_MONTH_YEAR'].apply(lambda month_year: t_avg_kaliningrad[month_year])
X['TEMP_AVG'] = X['VISIT_MONTH_YEAR'].apply(lambda month_year: t_avg_kaliningrad[month_year])


# In[557]:


train


# In[558]:


X


# ## Повторяем для данных с присоединенной средней температурой

# In[565]:


pool_18_months_train = Pool(
    X[X['VISIT_MONTH_YEAR'].isin(months_18_list)],
    y[X['VISIT_MONTH_YEAR'].isin(months_18_list)],
    cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY', 'POPULATION', 'TEMP_AVG']
)

pool_18_months_test = Pool(
    test.drop(['PATIENT_ID_COUNT'], axis=1),
    #test,
    cat_features = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY', 'POPULATION', 'TEMP_AVG']
)

#Объявление CatBoostRegressor и обучение
model = CatBoostRegressor(task_type='GPU')
model.fit(pool_18_months_train)

#Получение ответов модели на тестовой выборке в локальном тестировании 
y_pred = model.predict(pool_18_months_test)

# Оценим точность по baseline среднего за 3 последних месяца
print(
    "Значение метрики R2 между средним за 3 последних месяца и предсказанием CatBoost по 18 месяцам + население + средняя температура:",
    r2_score(
        pd.read_csv('baseline_3m.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True),
        y_pred.astype(int)
    ),
    sep='\n'
)
    
#Формируем sample_solution для отправки на платформу
test['PATIENT_ID_COUNT'] = y_pred.astype(int)
    
#Сохраняем в csv файл
test.to_csv('out.by_18_last_months.with_popualtion.non_negative.csv', sep=';', index=None)


# In[571]:


# Убираем отрицательные значения
test.loc[test.PATIENT_ID_COUNT < 0, 'PATIENT_ID_COUNT'] = 0


# In[572]:


test.iloc[:, :6]


# In[573]:


test


# In[574]:


#Сохраняем в csv файл
test.iloc[:, :6].to_csv('out.by_18_last_months.with_popualtion_and_temp.non_negative.csv', sep=';', index=None)


# In[575]:


print(
    "Значение метрики R2 между средним за 3 последних месяца и предсказанием CatBoost по 18 месяцам + население + средняя температура:",
    r2_score(
        pd.read_csv('baseline_3m.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True),
        test.PATIENT_ID_COUNT.astype(int)
    ),
    sep='\n'
)


# В системе выдало 0.545402. Лучше, чем до этого только с населением.
# Возможно таким путём присоедениния данных можно было ещё улучшить точность.
# Но что-то лично мне не очень верится в CatBoost для временных рядов.
# 
# К тому же надо изучить, чтобы не было коллиниарности данных, возможно какое-то из значений отбросить.
# Может быть лучшая точность могла быть достигнута при нормировании данных, ну или хотя бы приведения их к процентам.
# 
# В статье "Сезонность респираторных заболеваний": https://ru.wikipedia.org/wiki/Сезонность_респираторных_инфекций
# указываются возможные сезонные корреляции с температурой и влажностью воздуха. Особенно когда дело идёт около нуля градусов. Возможно feature engineering для комбинаций переменных температуры, влажности могли повысить точность модели.
# Например переменная изменяющая значение при прохождении температуры около нуля.

# ## Линейная экстраполяция по двум последним месяцам: (b - a)/1 + b = 2b - a
# ## b - значение в последнем месяце, a - значение в предпоследнем месяце

# In[662]:


lin_extrap = train[train['VISIT_MONTH_YEAR'].isin(['02.22', '03.22'])]


# In[663]:


lin_extrap['LIN_EXTRAP_TERM'] = 0
lin_extrap


# In[664]:


lin_extrap.loc[lin_extrap['VISIT_MONTH_YEAR'].isin(['02.22']), ['LIN_EXTRAP_TERM']] = -lin_extrap.loc[lin_extrap['VISIT_MONTH_YEAR'].isin(['02.22'])]['PATIENT_ID_COUNT'].fillna(0)


# In[665]:


lin_extrap.loc[lin_extrap['VISIT_MONTH_YEAR'].isin(['03.22']), ['LIN_EXTRAP_TERM']] = 2 * lin_extrap.loc[lin_extrap['VISIT_MONTH_YEAR'].isin(['03.22'])]['PATIENT_ID_COUNT'].fillna(0)


# In[666]:


lin_extrap


# In[667]:


Значения 2*b и -a подготовлены, осталось их просуммировать для каждой категории.
(lin_extrap
.groupby(['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY'], as_index=False)
['LIN_EXTRAP_TERM'].sum()
)


# In[668]:


lin_extrap['PATIENT_ID_COUNT'] = lin_extrap['LIN_EXTRAP_TERM']


# In[669]:


lin_extrap


# In[674]:


# Линейная экстраполяция по двум последним месяцам: (b - a)/1 + b = 2b - a
# b - значение в последнем месяце, a - значение в предпоследнем месяце
lin_extrap_solution = (
    #test.drop(['PATIENT_ID_COUNT'], axis=1).
    test.
    merge(
        (lin_extrap
        .groupby(['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY'], as_index=False)
        ['PATIENT_ID_COUNT'].mean()
        ),
        on=['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY'],
        how='left'
        )
    .fillna(0)
    .astype({'PATIENT_ID_COUNT': int})
)


# In[675]:


# Заменяем отрицательные значения на ноль для небольшого улучшения точности.
lin_extrap_solution.loc[lin_extrap_solution['PATIENT_ID_COUNT']<0, 'PATIENT_ID_COUNT'] = 0


# In[676]:


(lin_extrap_solution['PATIENT_ID_COUNT']<0).any()


# In[677]:


lin_extrap_solution.to_csv('lin_extrap_by_last_2_months.csv', sep=';', index=None)


# In[678]:


lin_extrap_solution


# In[679]:


print(
    "Значение метрики R2 между последним месяцем и линейной интерполяцией по последним двум месяцам:",
    r2_score(
        pd.read_csv('baseline_last_month_pred.csv', sep=';', usecols=['PATIENT_ID_COUNT'], squeeze=True),
        lin_extrap_solution.PATIENT_ID_COUNT.astype(int)
    ),
    sep='\n'
)
В системе выдало точность 0.748746

