import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


DATA_COUNT = 300  # количество случайных наблюдений
AXES = 3  # количество осей пространства
AXES_NAMES = ["x", "y", "z"]  # название осей
MAX_POSITION = 100  # максимальное значение координаты
CLUSTER_NUM = 9  # явное количество задаваемых кластеров
TEST_CASE_CLUSTER_NUM = 50  # максимальное количество кластеров при тестировании

# TODO Найти решение вопроса: Оптимальное количество кластеров необходимо определить из 
#  данных и дать обоснование выбранному количеству. 
#  Протестировать идею скачкообразных значение максимального удаления точки от центроиды кластера
#  (наименьшее значение и есть оптимальное количество кластеров), либо обратить внимание на средне статистическое
#  отдаление между ближайшими центроидами (наибольшее отдаление при наименьшем текс- кейсе и есть решение)
def test_case(test_num=TEST_CASE_CLUSTER_NUM):
    """
    Принятие решения о числе кластеров.
    Когда мера расстояния между двумя кластерами увеличивается скачнообразно, процесс объединения в новые кластеры
    необходимо остановить. Иначе будут объеденены кластеры, находящиеся на большом расстоянии друг от друга.
    Оптимальным считается число кластеров равное разности количества наблюдений и количества шагов,
    после которого коэффициент увеличивается скачкообразно.
    """
    test = list()  # буфер значение максимального расстояния до центроиды каждого кластера
    for i in range(1, test_num):
        cluster = StandardScaler().fit_transform(dataframe)
        cluster_num = i
        kmeans = KMeans(init='k-means++', n_clusters=cluster_num, n_init=12)
        kmeans.fit(cluster)
        labels = kmeans.labels_
        dataframe['label'] = labels
        for j in range(1, i):
            cond = dataframe.label == j
            subset = dataframe[cond].dropna()
            test.append(sum(kmeans.transform(subset)[-1]))  # максимальное отдаление точки от центроиды в кластере

# создание массива на основе рандомизированных значений Pandas
array = np.random.randint(MAX_POSITION, size=(DATA_COUNT, AXES))
columns = AXES_NAMES  # названия колонок
dataframe = pd.DataFrame(data=array, columns=columns)# перевод в Pandas DataFrame

cluster = StandardScaler().fit_transform(dataframe)  # подготовка кластеризации
kmeans = KMeans(init='k-means++', n_clusters=CLUSTER_NUM, n_init=12)  # инициализация метода к- средних
kmeans.fit(cluster)  # кластеризация дата- сета

labels = kmeans.labels_  # получение индекса кластера
dataframe['label'] = labels

figure = plt.figure()  # инициализация области визуализации
ax = figure.add_subplot(111, projection='3d')
# покадровая отрисовка: один кластер-- один цвет
for i in range(CLUSTER_NUM):
    cond = dataframe.label == i
    subset = dataframe[cond].dropna()
    ax.scatter(subset.x, subset.y, subset.z, cmap='rainbow', label=str(i+1))

plt.legend(bbox_to_anchor=(1.3, 0.9), bbox_transform=ax.transAxes)
plt.show()  # вывод 3D модели
