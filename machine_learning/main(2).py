import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Загрузка и предобработка данных
df = pd.read_csv('titanic.csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Обработка пропусков в 'Embarked'
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Заполнение пропусков в 'Age' с использованием медианы
age_imputer = SimpleImputer(strategy='median')
df['Age'] = age_imputer.fit_transform(df[['Age']])

# Кодирование пола (мужчина = 1, женщина = 0)
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# Определение признаков и целевой переменной
X = df.drop('Survived', axis=1)
y = df['Survived']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Масштабирование данных
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Использование Random Forest Classifier для обучения
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Предсказание и оценка модели
y_pred = classifier.predict(X_test)

# Вывод результатов
print('Предсказания:', y_pred)
print('Процент правильно предсказанных исходов:', accuracy_score(y_test, y_pred) * 100)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))