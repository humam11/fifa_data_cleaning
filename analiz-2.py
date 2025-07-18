import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Настройка для отображения русских символов
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


print("исследовательский анализ данных fifa 2021")


# Загрузка данных
print("\n1. ЗАГРУЗКА ДАННЫХ")

df = pd.read_csv('players_cleaned.csv')
print(f"Размер датасета: {df.shape}")
print(f"Количество игроков: {len(df)}")
print(f"Количество признаков: {len(df.columns)}")

print("\nПервые 5 строк датасета:")
print(df.head())

print("\nИнформация о датасете:")
print(df.info())

print("\nСтатистическое описание числовых признаков:")
print(df.describe())

# 1. индексация по координаторам и логическая индексация
print("\n")
print("2. индексация по координаторам и логическая индексация")


# индексация по координаторам (не менее 5 различных условий)
print("\n2.1. индексация по координаторам:")


# 1. Выбор первых 10 игроков
print("1. Первые 10 игроков:")
print(df.iloc[0:10, [1, 5, 6, 7, 8]])  # longname, name, age, ova, pot

# 2. Выбор игроков с индексами 100-110
print("\n2. Игроки с индексами 100-110:")
print(df.iloc[100:111, [1, 5, 6, 7, 8]])

# 3. Выбор последних 5 игроков
print("\n3. Последние 5 игроков:")
print(df.iloc[-5:, [1, 5, 6, 7, 8]])

# 4. Выбор игроков с шагом 1000 (каждый 1000-й игрок)
print("\n4. Каждый 1000-й игрок:")
print(df.iloc[::1000, [1, 5, 6, 7, 8]])

# 5. Выбор игроков с индексами 5000-5010
print("\n5. Игроки с индексами 5000-5010:")
print(df.iloc[5000:5011, [1, 5, 6, 7, 8]])

# 6. Выбор игроков с индексами 10000-10005
print("\n6. Игроки с индексами 10000-10005:")
print(df.iloc[10000:10006, [1, 5, 6, 7, 8]])

# 7. Выбор игроков с индексами 15000-15010
print("\n7. Игроки с индексами 15000-15010:")
print(df.iloc[15000:15011, [1, 5, 6, 7, 8]])

# 8. Выбор игроков с индексами 18000-18005
print("\n8. Игроки с индексами 18000-18005:")
print(df.iloc[18000:18006, [1, 5, 6, 7, 8]])

# 9. Выбор игроков с индексами 500-510
print("\n9. Игроки с индексами 500-510:")
print(df.iloc[500:511, [1, 5, 6, 7, 8]])

# 10. Выбор игроков с индексами 1000-1010
print("\n10. Игроки с индексами 1000-1010:")
print(df.iloc[1000:1011, [1, 5, 6, 7, 8]])

# Логическая индексация (не менее 5 различных условий)
print("\n2.2. ЛОГИЧЕСКАЯ ИНДЕКСАЦИЯ:")


# 1. Игроки с общим рейтингом выше 90
print("1. Игроки с общим рейтингом выше 90:")
high_rated = df[df['ova'] > 90]
print(f"Количество игроков с рейтингом > 90: {len(high_rated)}")
print(high_rated[['longname', 'age', 'ova', 'pot', 'nationality']].head(10))

# 2. Игроки в возрасте до 20 лет с потенциалом выше 85
print("\n2. Молодые игроки (до 20 лет) с потенциалом выше 85:")
young_talents = df[(df['age'] < 20) & (df['pot'] > 85)]
print(f"Количество молодых талантов: {len(young_talents)}")
print(young_talents[['longname', 'age', 'ova', 'pot', 'nationality']].head(10))

# 3. Игроки с ростом выше 190 см и весом больше 80 кг
print("\n3. Высокие и тяжелые игроки (рост > 190 см, вес > 80 кг):")
tall_heavy = df[(df['height_cm'] > 190) & (df['weight_kg'] > 80)]
print(f"Количество высоких и тяжелых игроков: {len(tall_heavy)}")
print(tall_heavy[['longname', 'height_cm', 'weight_kg', 'positions_formatted']].head(10))

# 4. Игроки с высокой скоростью (sprint_speed > 90)
print("\n4. Быстрые игроки (sprint_speed > 90):")
fast_players = df[df['sprint_speed'] > 90]
print(f"Количество быстрых игроков: {len(fast_players)}")
print(fast_players[['longname', 'sprint_speed', 'acceleration', 'positions_formatted']].head(10))

# 5. Игроки с высокими навыками владения мячом (dribbling > 85)
print("\n5. Игроки с высокими навыками владения мячом (dribbling > 85):")
skilled_players = df[df['dribbling'] > 85]
print(f"Количество техничных игроков: {len(skilled_players)}")
print(skilled_players[['longname', 'dribbling', 'ball_control', 'positions_formatted']].head(10))

# 2. СОРТИРОВКА ДАННЫХ
print("\n")
print("3. СОРТИРОВКА ДАННЫХ")


# Сортировка по различным столбцам
print("\n3.1. Топ-10 игроков по общему рейтингу:")
top_rated = df.sort_values('ova', ascending=False).head(10)
print(top_rated[['longname', 'age', 'ova', 'pot', 'nationality']])

print("\n3.2. Топ-10 молодых игроков по потенциалу:")
young_potential = df.sort_values('pot', ascending=False).head(10)
print(young_potential[['longname', 'age', 'ova', 'pot', 'nationality']])

print("\n3.3. Самые высокие игроки:")
tallest = df.sort_values('height_cm', ascending=False).head(10)
print(tallest[['longname', 'height_cm', 'weight_kg', 'positions_formatted']])

print("\n3.4. Самые быстрые игроки:")
fastest = df.sort_values('sprint_speed', ascending=False).head(10)
print(fastest[['longname', 'sprint_speed', 'acceleration', 'positions_formatted']])

print("\n3.5. Самые техничные игроки:")
most_skilled = df.sort_values('dribbling', ascending=False).head(10)
print(most_skilled[['longname', 'dribbling', 'ball_control', 'positions_formatted']])

# Анализ наибольших и наименьших значений
print("\n3.6. АНАЛИЗ ЭКСТРЕМАЛЬНЫХ ЗНАЧЕНИЙ:")


print(f"Максимальный общий рейтинг: {df['ova'].max()}")
print(f"Минимальный общий рейтинг: {df['ova'].min()}")
print(f"Средний общий рейтинг: {df['ova'].mean():.2f}")

print(f"\nМаксимальный возраст: {df['age'].max()}")
print(f"Минимальный возраст: {df['age'].min()}")
print(f"Средний возраст: {df['age'].mean():.2f}")

print(f"\nМаксимальный рост: {df['height_cm'].max():.2f} см")
print(f"Минимальный рост: {df['height_cm'].min():.2f} см")
print(f"Средний рост: {df['height_cm'].mean():.2f} см")

# 3. фильтрация данных
print("\n")
print("4. фильтрация данных")


# фильтрация с помощью метода query (не менее 5 различных фильтров)
print("\n4.1. фильтрация с помощью метода query:")


# 1. Игроки из топ-5 стран
print("1. Игроки из топ-5 стран по количеству игроков:")
top_countries = df['nationality'].value_counts().head(5).index.tolist()
top_countries_players = df.query('nationality in @top_countries')
print(f"Количество игроков из топ-5 стран: {len(top_countries_players)}")
print(top_countries_players[['longname', 'nationality', 'ova']].head(10))

# 2. Игроки с высокими физическими показателями
print("\n2. Игроки с высокими физическими показателями (stamina > 80 и strength > 80):")
physical_players = df.query('stamina > 80 and strength > 80')
print(f"Количество физически сильных игроков: {len(physical_players)}")
print(physical_players[['longname', 'stamina', 'strength', 'positions_formatted']].head(10))

# 3. Игроки с высокими навыками удара
print("\n3. Игроки с высокими навыками удара (shot_power > 85 и finishing > 85):")
shooting_players = df.query('shot_power > 85 and finishing > 85')
print(f"Количество игроков с высокими навыками удара: {len(shooting_players)}")
print(shooting_players[['longname', 'shot_power', 'finishing', 'positions_formatted']].head(10))

# 4. Игроки с высокими защитными навыками
print("\n4. Игроки с высокими защитными навыками (marking > 80 and standing_tackle > 80):")
defensive_players = df.query('marking > 80 and standing_tackle > 80')
print(f"Количество защитников с высокими навыками: {len(defensive_players)}")
print(defensive_players[['longname', 'marking', 'standing_tackle', 'positions_formatted']].head(10))

# 5. Игроки с высокими навыками передачи
print("\n5. Игроки с высокими навыками передачи (short_passing > 85 and vision > 85):")
passing_players = df.query('short_passing > 85 and vision > 85')
print(f"Количество игроков с высокими навыками передачи: {len(passing_players)}")
print(passing_players[['longname', 'short_passing', 'vision', 'positions_formatted']].head(10))

# фильтрация с помощью оператора where (не менее 5 различных фильтров)
print("\n4.2. фильтрация с помощью оператора where:")


# 1. Игроки с высоким рейтингом и молодым возрастом
print("1. Игроки с рейтингом > 80 и возрастом < 25:")
young_stars = df.where((df['ova'] > 80) & (df['age'] < 25)).dropna()
print(f"Количество молодых звезд: {len(young_stars)}")
print(young_stars[['longname', 'age', 'ova', 'pot', 'nationality']].head(10))

# 2. Игроки с высокими навыками вратаря
print("\n2. Игроки с высокими навыками вратаря (gk_diving > 70 and gk_handling > 70):")
good_goalkeepers = df.where((df['gk_diving'] > 70) & (df['gk_handling'] > 70)).dropna()
print(f"Количество хороших вратарей: {len(good_goalkeepers)}")
print(good_goalkeepers[['longname', 'gk_diving', 'gk_handling', 'gk_reflexes']].head(10))

# 3. Игроки с высокими ментальными качествами
print("\n3. Игроки с высокими ментальными качествами (composure > 75 and positioning > 75):")
mental_players = df.where((df['composure'] > 75) & (df['positioning'] > 75)).dropna()
print(f"Количество игроков с высокими ментальными качествами: {len(mental_players)}")
print(mental_players[['longname', 'composure', 'positioning', 'positions_formatted']].head(10))

# 4. Игроки с высокими навыками контроля мяча
print("\n4. Игроки с высокими навыками контроля мяча (ball_control > 75 and dribbling > 75):")
control_players = df.where((df['ball_control'] > 75) & (df['dribbling'] > 75)).dropna()
print(f"Количество игроков с высоким контролем мяча: {len(control_players)}")
print(control_players[['longname', 'ball_control', 'dribbling', 'positions_formatted']].head(10))

# 5. Игроки с высокими навыками удара головой
print("\n5. Игроки с высокими навыками удара головой (heading_accuracy > 75 and jumping > 75):")
heading_players = df.where((df['heading_accuracy'] > 75) & (df['jumping'] > 75)).dropna()
print(f"Количество игроков с высокими навыками удара головой: {len(heading_players)}")
print(heading_players[['longname', 'heading_accuracy', 'jumping', 'positions_formatted']].head(10))


# 3. ФИЛЬТРАЦИЯ ДАННЫХ С ПОМОЩЬЮ МЕТОДА QUERY (10 различных фильтров)
print("\n3. ФИЛЬТРАЦИЯ ДАННЫХ С ПОМОЩЬЮ МЕТОДА QUERY (10 различных фильтров)")

# 1. Игроки старше 30 лет
older_players = df.query('age > 30')
print("1. Игроки старше 30 лет:")
print(older_players[['longname', 'age', 'ova', 'nationality']].head(10))

# 2. Игроки с рейтингом между 80 и 85
mid_rated = df.query('ova >= 80 and ova <= 85')
print("\n2. Игроки с рейтингом между 80 и 85:")
print(mid_rated[['longname', 'age', 'ova', 'nationality']].head(10))

# 3. Игроки из Германии с потенциалом выше 80
brazil_talents = df.query('nationality == "Germany" and pot > 80')
print("\n3. Игроки из Германии с потенциалом выше 80:")
print(brazil_talents[['longname', 'age', 'ova', 'pot']].head(10))

# 4. Вратари с ростом выше 185 см
high_gks = df.query('positions_formatted == "GK" and height_cm > 185')
print("\n4. Вратари с ростом выше 185 см:")
print(high_gks[['longname', 'height_cm', 'weight_kg', 'ova']].head(10))

# 5. Игроки с выносливостью менее 60 и возрастом до 23 лет
low_stamina_young = df.query('stamina < 60 and age < 23')
print("\n5. Игроки с выносливостью < 60 и возрастом < 23:")
print(low_stamina_young[['longname', 'age', 'stamina', 'ova']].head(10))

# 6. CB или LB с силой удара > 60
cb_lb_shot = df.query('(positions_formatted == "CB" or positions_formatted == "LB") and shot_power > 60')
print("\n6. CB или LB с силой удара > 60:")
print(cb_lb_shot[['longname', 'positions_formatted', 'shot_power', 'ova']].head(10))

# 7. Игроки с ростом между 175 и 180 см и весом менее 70 кг
medium_height_light = df.query('height_cm >= 175 and height_cm <= 180 and weight_kg < 70')
print("\n7. Игроки с ростом 175-180 см и весом < 70 кг:")
print(medium_height_light[['longname', 'height_cm', 'weight_kg', 'ova']].head(10))

# 8. Игроки с навыком дриблинга > 80 и завершением > 75
skill_finish = df.query('dribbling > 80 and finishing > 75')
print("\n8. Игроки с дриблингом > 80 и завершением > 75:")
print(skill_finish[['longname', 'dribbling', 'finishing', 'ova']].head(10))

# 9. Молодые игроки из Франции
argentina_young = df.query('nationality == "France" and age < 25')
print("\n9. Молодые игроки из Франции:")
print(argentina_young[['longname', 'age', 'ova', 'pot']].head(10))

# 10. CM с потенциалом > 80
high_pot_st = df.query('pot > 80 and positions_formatted == "CM"')
print("\n10. CM с потенциалом > 80:")
print(high_pot_st[['longname', 'age', 'ova', 'pot']].head(10))

# 4. СВОДНЫЕ ТАБЛИЦЫ
print("\n")
print("5. сводные таблицы")


print("\n5.1. сводная таблица 1: средний рейтинг по национальностям (топ-10):")
print("-" * 70)
pivot1 = df.pivot_table(
    values='ova',
    index='nationality',
    aggfunc=['mean', 'count', 'max']
).round(2)
pivot1.columns = ['Средний рейтинг', 'Количество игроков', 'Максимальный рейтинг']
pivot1 = pivot1.sort_values('Средний рейтинг', ascending=False).head(10)
print(pivot1)

print("\n5.2. СВОДНАЯ ТАБЛИЦА 2: Средние показатели по позициям:")
print("-" * 70)
pivot2 = df.pivot_table(
    values=['ova', 'age', 'height_cm', 'weight_kg'],
    index='positions_formatted',
    aggfunc='mean'
).round(2)
pivot2.columns = ['Средний рейтинг', 'Средний возраст', 'Средний рост (см)', 'Средний вес (кг)']
print(pivot2)

print("\n5.3. СВОДНАЯ ТАБЛИЦА 3: Распределение игроков по возрастным группам и рейтингу:")
print("-" * 70)
# Создаем возрастные группы
df['age_group'] = pd.cut(df['age'], bins=[0, 20, 25, 30, 35, 100], 
                        labels=['До 20', '20-25', '25-30', '30-35', '35+'])
pivot3 = df.pivot_table(
    values='ova',
    index='age_group',
    columns=pd.cut(df['ova'], bins=[0, 70, 80, 85, 90, 100], 
                  labels=['До 70', '70-80', '80-85', '85-90', '90+']),
    aggfunc='count',
    fill_value=0
)
pivot3.columns.name = 'Рейтинг'
pivot3.index.name = 'Возрастная группа'
print(pivot3)

# 5. группировка данных и агрегатные функции
print("\n")
print("6. группировка данных и агрегатные функции")


print("\n6.1. группировка по национальностям:")

nationality_stats = df.groupby('nationality').agg({
    'ova': ['mean', 'max', 'min', 'count'],
    'age': ['mean', 'min', 'max'],
    'height_cm': ['mean', 'min', 'max'],
    'weight_kg': ['mean', 'min', 'max']
}).round(2)

nationality_stats.columns = [
    'Средний рейтинг', 'Макс рейтинг', 'Мин рейтинг', 'Количество игроков',
    'Средний возраст', 'Мин возраст', 'Макс возраст',
    'Средний рост', 'Мин рост', 'Макс рост',
    'Средний вес', 'Мин вес', 'Макс вес'
]

print("Топ-10 стран по среднему рейтингу:")
print(nationality_stats.sort_values('Средний рейтинг', ascending=False).head(10))

print("\n6.2. ГРУППИРОВКА ПО ПОЗИЦИЯМ:")

position_stats = df.groupby('positions_formatted').agg({
    'ova': ['mean', 'max', 'min', 'count'],
    'age': ['mean', 'min', 'max'],
    'height_cm': ['mean', 'min', 'max'],
    'weight_kg': ['mean', 'min', 'max'],
    'sprint_speed': ['mean', 'max'],
    'dribbling': ['mean', 'max'],
    'shot_power': ['mean', 'max'],
    'short_passing': ['mean', 'max'],
    'marking': ['mean', 'max'],
    'stamina': ['mean', 'max']
}).round(2)

position_stats.columns = [
    'Средний рейтинг', 'Макс рейтинг', 'Мин рейтинг', 'Количество игроков',
    'Средний возраст', 'Мин возраст', 'Макс возраст',
    'Средний рост', 'Мин рост', 'Макс рост',
    'Средний вес', 'Мин вес', 'Макс вес',
    'Средняя скорость', 'Макс скорость',
    'Среднее владение', 'Макс владение',
    'Средняя сила удара', 'Макс сила удара',
    'Средняя передача', 'Макс передача',
    'Средняя опека', 'Макс опека',
    'Средняя выносливость', 'Макс выносливость'
]

print(position_stats)

print("\n6.3. ГРУППИРОВКА ПО ВОЗРАСТНЫМ ГРУППАМ:")

age_group_stats = df.groupby('age_group').agg({
    'ova': ['mean', 'max', 'min', 'count'],
    'pot': ['mean', 'max', 'min'],
    'height_cm': ['mean', 'min', 'max'],
    'weight_kg': ['mean', 'min', 'max']
}).round(2)

age_group_stats.columns = [
    'Средний рейтинг', 'Макс рейтинг', 'Мин рейтинг', 'Количество игроков',
    'Средний потенциал', 'Макс потенциал', 'Мин потенциал',
    'Средний рост', 'Мин рост', 'Макс рост',
    'Средний вес', 'Мин вес', 'Макс вес'
]

print(age_group_stats)

# 6. исследовательский анализ данных
print("\n")
print("7. исследовательский анализ данных")


print("\n7.1. корреляционный анализ:")

# Выбираем числовые колонки для корреляционного анализа
numeric_columns = [
    'ova', 'pot', 'age', 'height_cm', 'weight_kg',
    'sprint_speed', 'acceleration', 'agility', 'reactions', 'balance',
    'shot_power', 'finishing', 'short_passing', 'ball_control', 'dribbling',
    'stamina', 'strength', 'marking', 'standing_tackle', 'jumping',
    'vision', 'composure', 'heading_accuracy'
]
numeric_columns = [col for col in numeric_columns if col in df.columns]
correlation_matrix = df[numeric_columns].corr()
print("Корреляционная матрица (топ-5 корреляций с общим рейтингом):")
ova_correlations = correlation_matrix['ova'].sort_values(ascending=False)
print(ova_correlations.head(6))  # Включая саму переменную

print("\n7.2. анализ распределения рейтингов:")

print(f"Распределение игроков по рейтингу:")
print(f"90+: {len(df[df['ova'] >= 90])} игроков")
print(f"85-89: {len(df[(df['ova'] >= 85) & (df['ova'] < 90)])} игроков")
print(f"80-84: {len(df[(df['ova'] >= 80) & (df['ova'] < 85)])} игроков")
print(f"75-79: {len(df[(df['ova'] >= 75) & (df['ova'] < 80)])} игроков")
print(f"70-74: {len(df[(df['ova'] >= 70) & (df['ova'] < 75)])} игроков")
print(f"До 70: {len(df[df['ova'] < 70])} игроков")

print("\n7.3. АНАЛИЗ ВОЗРАСТНОГО РАСПРЕДЕЛЕНИЯ:")

age_distribution = df['age'].value_counts().sort_index()
print("Распределение игроков по возрасту (топ-10 возрастов):")
print(age_distribution.head(10))

print("\n7.4. АНАЛИЗ ПОЗИЦИЙ:")

position_counts = df['positions_formatted'].value_counts()
print("Распределение игроков по позициям:")
print(position_counts)

print("\n7.5. АНАЛИЗ НАЦИОНАЛЬНОСТЕЙ:")

nationality_counts = df['nationality'].value_counts()
print("Топ-15 стран по количеству игроков:")
print(nationality_counts.head(15))

print("\n7.6. АНАЛИЗ ФИЗИЧЕСКИХ ХАРАКТЕРИСТИК:")

print(f"Средний рост игроков: {df['height_cm'].mean():.2f} см")
print(f"Средний вес игроков: {df['weight_kg'].mean():.2f} кг")
print(f"Средний ИМТ игроков: {(df['weight_kg'] / ((df['height_cm']/100)**2)).mean():.2f}")

print("\n7.7. АНАЛИЗ НАВЫКОВ:")

skills = [
    'sprint_speed', 'dribbling', 'shot_power', 'short_passing', 'stamina',
    'strength', 'marking', 'finishing', 'ball_control', 'acceleration',
    'agility', 'reactions', 'vision', 'composure', 'heading_accuracy'
]
skills = [col for col in skills if col in df.columns]
skill_means = df[skills].mean().sort_values(ascending=False)
print("Средние значения навыков:")
for skill, mean_value in skill_means.items():
    print(f"{skill}: {mean_value:.2f}")

print("\n7.8. АНАЛИЗ ПОТЕНЦИАЛА:")

df['potential_growth'] = df['pot'] - df['ova']
print(f"Средний рост потенциала: {df['potential_growth'].mean():.2f}")
print(f"Максимальный рост потенциала: {df['potential_growth'].max():.2f}")
print(f"Игроки с наибольшим потенциалом роста:")
growth_players = df.nlargest(10, 'potential_growth')
print(growth_players[['longname', 'age', 'ova', 'pot', 'nationality', 'potential_growth']])

# 7. промежуточные выводы
print("\n")
print("8. промежуточные выводы")


print("\nвыводы о проделанной работе:")


print("1. СТРУКТУРА ДАННЫХ:")
print("   - Датасет содержит информацию о футболистах FIFA 2021")
print(f"   - Общее количество игроков: {len(df)}")
print(f"   - Количество признаков: {len(df.columns)}")
print("   - Данные включают физические характеристики, навыки, рейтинги и личную информацию")

print("\n2. КАЧЕСТВО ДАННЫХ:")
print("   - Данные хорошо структурированы и готовы к анализу")
print("   - Отсутствуют значительные пропуски в ключевых признаках")
print("   - Разнообразие национальностей и позиций обеспечивает репрезентативность")

print("\n3. КЛЮЧЕВЫЕ ЗАКОНОМЕРНОСТИ:")
print("   - Наиболее высокий средний рейтинг у игроков из топ-футбольных стран")
print("   - Молодые игроки (до 20 лет) имеют высокий потенциал роста")
print("   - Физические характеристики коррелируют с позициями игроков")
print("   - Навыки игроков распределены неравномерно по позициям")

print("\n4. БИЗНЕС-ИНСАЙТЫ:")
print("   - Выявлены перспективные молодые игроки для скаутинга")
print("   - Определены ключевые характеристики для каждой позиции")
print("   - Установлены корреляции между различными навыками")
print("   - Выявлены национальные особенности в развитии игроков")

print("\n5. РЕКОМЕНДАЦИИ:")
print("   - Фокус на молодых игроках с высоким потенциалом")
print("   - Учет физических характеристик при подборе игроков")
print("   - Анализ национальных школ для понимания стилей игры")
print("   - Мониторинг развития навыков по возрастным группам")

print("\n")
print("АНАЛИЗ ЗАВЕРШЕН")
 