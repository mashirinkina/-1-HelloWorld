# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Ширинкина Мария Алексанровна
- НМТ-213929
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
- Для Python в отчете привести скриншоты с демонстрацией сохранения документа google.colab на свой диск с запуском программы, выводящей сообщение Hello World.
- Для Unity в отчете привести скришноты вывода сообщения Hello World в консоль.
### Вывод Hello World на Python
```py

print('Hello World')

```
![image](https://user-images.githubusercontent.com/114253132/192147841-efcdb60d-4db1-499f-9d83-324139c102bc.png)
![image](https://user-images.githubusercontent.com/114253132/192147872-3f1f5552-099e-4b9a-b5a3-b512eeef462b.png)


### Вывод Hello World в консоль на Unity

```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HelloWorld : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Hello World");
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
```
![image](https://user-images.githubusercontent.com/114601006/192829371-6b256c77-b2a7-4787-9dc9-b98549fd9e30.png)





## Задание 2
В разделе «ход работы» пошагово выполнить каждый пункт с описанием и примером реализации задачи по теме лабораторной работы.

1. Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

plt.scatter(x,y)
```
![image](https://user-images.githubusercontent.com/114253132/192267013-a0f27ee4-75fd-4e96-aa4e-9e2ca4a42678.png)

-Ввод данных и преобразование их в формат массива для использования операций умножений сложения и умножения 

2. Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.

```py
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

plt.scatter(x,y)

def model(a, b, x):
  return a*x + b

def loss_function(a, b, x, y):
  num = len(x)
  prediction = model(a, b, x)
  return (0.5/num) * (np.square(prediction-y)).sum()

def optimize(a, b, x, y):
  num = len(x)
  prediction = model(a, b, x)

  da = (1.0 / num) * ((prediction - y)*x).sum()
  db = (1.0 / num) * ((prediction - y).sum())
  a = a - Lr * da
  b = b - Lr * db
  return a, b

def iterate(a, b, x, y, times):
  for i in range (times):
    a, b = optimize(a, b, x, y)
  return a, b
```
-Создание функций и применение еще не описанных переменных. Базовой моделью регрессии является wx + b, в нашем случае это ax + b. Функция оптимизации используется для обновления двух параметров a и b.

3. Начать итерацию

```py
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

plt.scatter(x,y)

def model(a, b, x):
  return a*x + b

def loss_function(a, b, x, y):
  num = len(x)
  prediction = model(a, b, x)
  return (0.5/num) * (np.square(prediction-y)).sum()

def optimize(a, b, x, y):
  num = len(x)
  prediction = model(a, b, x)

  da = (1.0 / num) * ((prediction - y)*x).sum()
  db = (1.0 / num) * ((prediction - y).sum())
  a = a - Lr * da
  b = b - Lr * db
  return a, b

def iterate(a, b, x, y, times):
  for i in range (times):
    a, b = optimize(a, b, x, y)
  return a, b

a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

a,b = iterate(a, b, x, y, 1)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)
```
-Задаем случайным способом значения переменным a и b. Задание значения для переменной Lr. Создание модели итеративной оптимимзации.

-На каждой итерации отображаются значения параметров, значения потерь и эффекты визуализации после итерации.

![image](https://user-images.githubusercontent.com/114253132/192813408-0148452a-db79-4ee6-b1fd-e6655d480022.png)

## Задание 3
### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.

- Перечисленные в этом туториале действия могут быть выполнены запуском на исполнение скрипт-файла, доступного [в репозитории](https://github.com/Den1sovDm1triy/hfss-scripting/blob/main/ScreatingSphereInAEDT.py).
- Для запуска скрипт-файла откройте Ansys Electronics Desktop. Перейдите во вкладку [Automation] - [Run Script] - [Выберите файл с именем ScreatingSphereInAEDT.py из репозитория].

```py

import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.NewProject()
oProject.Rename("C:/Users/denisov.dv/Documents/Ansoft/SphereDIffraction.aedt", True)
oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Terminal Network", "")
oDesign = oProject.SetActiveDesign("HFSSDesign1")
oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.CreateSphere(
	[
		"NAME:SphereParameters",
		"XCenter:="		, "0mm",
		"YCenter:="		, "0mm",
		"ZCenter:="		, "0mm",
		"Radius:="		, "1.0770329614269mm"
	], 
)

```

## Выводы

Абзац умных слов о том, что было сделано и что было узнано.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
