# улучшенная очистка данных fifa игроков
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

try:
    df = pd.read_csv('players.csv', encoding='utf-8')
except Exception as e:
    print(f'Ошибка при загрузке файла: {e}')
    sys.exit(1)

# удалить лишние символы
df.columns = (df.columns
              .str.lower()
              .str.replace(' ', '_')
              .str.replace('-', '_')
              .str.replace('&', 'and')
              .str.replace('↓', '')
              .str.replace('/', '_')
              .str.replace('(', '')
              .str.replace(')', '')
              .str.strip('_'))

print("\nНазвания столбцов после приведения к змеиному регистру:")
print(df.columns.tolist())

# привести текстовые данные к нижнему регистру (кроме имен игроков)
text_columns = df.select_dtypes(include=['object']).columns
for col in text_columns:
    if col not in ['id', 'playerurl', 'photourl', 'name', 'full_name']:
        df[col] = df[col].astype(str).str.lower().str.strip()

print("\nТекстовые данные приведены к нижнему регистру (кроме имен игроков)")

# разделить столбец team_and_contract на части
if 'team_and_contract' in df.columns:
    print("\nРазделение столбца team_and_contract на название клуба, год начала и год окончания...")
    
    def split_team_contract(value):

        if pd.isna(value) or value == 'nan':
            return pd.Series([np.nan, np.nan, np.nan])
        
        value = str(value).strip()
        lines = value.split('\n')
        
        if len(lines) >= 2:
            team = lines[0].strip()
            contract = lines[1].strip()
            
            if '~' in contract:
                years = contract.split('~')
                if len(years) == 2:
                    start_year = years[0].strip()
                    end_year = years[1].strip()
                    try:
                        start_year = int(start_year)
                        end_year = int(end_year)
                    except ValueError:
                        start_year = np.nan
                        end_year = np.nan
                    return pd.Series([team, start_year, end_year])
            
            return pd.Series([team, np.nan, np.nan])
        else:
            return pd.Series([value, np.nan, np.nan])
    
    df[['club_name', 'contract_start_year', 'contract_end_year']] = df['team_and_contract'].apply(split_team_contract)
    df = df.drop('team_and_contract', axis=1)
    print("Столбец team_and_contract разделен на club_name, contract_start_year, contract_end_year")

# обработать позиции игроков
if 'positions' in df.columns:
    print("\nОбработка позиций игроков...")
    
    positions_split = df['positions'].astype(str).str.split()
    
    all_positions = set()
    for positions in positions_split.dropna():
        if isinstance(positions, list):
            all_positions.update(positions)
    
    all_positions.discard('nan')
    
    print(f"Найдено уникальных позиций: {sorted(all_positions)}")
    
    for position in sorted(all_positions):
        df[f'position_{position.lower()}'] = positions_split.apply(
            lambda x: 1 if isinstance(x, list) and position in x else 0
        )
    
    # сохранить позиции как строку, разделенную запятыми для удобства
    df['positions_formatted'] = df['positions'].astype(str).str.replace(' ', ',')
    
    print("Позиции преобразованы и созданы dummy переменные")
    print("Создан столбец 'positions_formatted' с позициями через запятую без пробелов (например: cm,st)")

# преобразовать рост из футов и дюймов в сантиметры
if 'height' in df.columns:
    print("\nПреобразование роста из футов и дюймов в сантиметры")
    def convert_height(height_str):
        if pd.isna(height_str):
            return np.nan
        height_str = str(height_str).strip()
        if "'" in height_str:
            parts = height_str.replace('"', '').split("'")
            if len(parts) == 2:
                feet = int(parts[0])
                inches = int(parts[1]) if parts[1] else 0
                return feet * 30.48 + inches * 2.54
        return np.nan
    
    df['height_cm'] = df['height'].apply(convert_height)
    df = df.drop('height', axis=1)

# преобразовать вес из фунтов в килограммы
if 'weight' in df.columns:
    print("\nПреобразование веса из фунтов в килограммы")
    df['weight_kg'] = df['weight'].astype(str).str.extract('(\d+)').astype(float) * 0.453592
    df = df.drop('weight', axis=1)

# обработать звездные рейтинги и удалить звезды из определенных столбцов
star_columns = [col for col in df.columns if '★' in str(col) or 'star' in str(col).lower()]
for col in star_columns:
    if col in df.columns:
        print(f"\nИзвлечение числового значения из {col}")
        df[col] = df[col].astype(str).str.extract('(\d+)').astype(float)

# удалить звезды из столбцов w_f, ir, sm
columns_to_clean_stars = ['w_f', 'ir', 'sm']
for col in columns_to_clean_stars:
    if col in df.columns:
        print(f"\nУдаление звезд из столбца {col}")
        df[col] = df[col].astype(str).str.replace('★', '', regex=False).str.extract('(\d+)').astype(float)
        print(f"Столбец {col} очищен от звезд")

# удалить ненужные денежные столбцы, если они есть
date_columns = ['joined', 'loan_date_end']
for col in date_columns:
    if col in df.columns:
        print(f"\nПреобразование {col} в формат даты")
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            print(f"Невозможно преобразовать {col} в дату")

# проверка на пропущенные значения
print("\nПроверка на пропущенные значения:")
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_info = pd.DataFrame({
    'количество_пропусков': missing_data,
    'процент_пропусков': missing_percentage
})
missing_info = missing_info[missing_info['количество_пропусков'] > 0]
print(missing_info)

if missing_info.empty:
    print("Пропущенных значений не найдено")
else:
    print("\nОбработка пропущенных значений:")
    for col in missing_info.index:
        missing_pct = missing_info.loc[col, 'процент_пропусков']
        
        if missing_pct <= 5:
            if df[col].dtype in ['int64', 'float64', 'float32']:
                if abs(df[col].skew()) > 1:
                    fill_value = df[col].median()
                    df[col].fillna(fill_value, inplace=True)
                    print(f"Заполнен {col} медианой: {fill_value:.2f} (данные имеют асимметричное распределение)")
                else:
                    fill_value = df[col].mean()
                    df[col].fillna(fill_value, inplace=True)
                    print(f"Заполнен {col} средним: {fill_value:.2f} (данные имеют нормальное распределение)")
            else:
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                df[col].fillna(fill_value, inplace=True)
                print(f"Заполнен {col} модой: {fill_value} (наиболее частое значение)")
        else:
            print(f"Столбец {col} имеет {missing_pct:.1f}% пропусков, требует индивидуального рассмотрения")

# анализ и изменение типов данных
for col in df.columns:
    current_type = df[col].dtype
    old_type = current_type
    if col in ['age', 'ova', 'pot'] and current_type == 'object':
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
    elif current_type == 'object' and col not in ['positions_list']:
        try:
            unique_count = df[col].nunique()
            if unique_count < 50:
                df[col] = df[col].astype('category')
        except TypeError:
            pass
    new_type = df[col].dtype
    if old_type != new_type:
        print(f"Столбец '{col}' преобразован: {old_type} -> {new_type}")

# оптимизация типов данных
for col in df.columns:
    old_type = df[col].dtype
    if df[col].dtype == 'int64':
        if df[col].min() >= 0 and df[col].max() <= 255:
            df[col] = df[col].astype('uint8')
        elif df[col].min() >= 0 and df[col].max() <= 65535:
            df[col] = df[col].astype('uint16')
        elif df[col].min() >= -128 and df[col].max() <= 127:
            df[col] = df[col].astype('int8')
        elif df[col].min() >= -32768 and df[col].max() <= 32767:
            df[col] = df[col].astype('int16')
    elif df[col].dtype == 'float64':
        if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
            df[col] = df[col].astype('float32')
    new_type = df[col].dtype
    if old_type != new_type:
        print(f"Столбец '{col}' оптимизирован: {old_type} -> {new_type}")

# проверка на дубликаты
initial_shape = df.shape
duplicates = df.duplicated()
duplicate_count = duplicates.sum()

print(f"\nКоличество дубликатов: {duplicate_count}")

if duplicate_count > 0:
    print("Примеры дубликатов:")
    print(df[duplicates].head())
    
    print("Возможные причины дубликатов:")
    print("Технические ошибки при сборе данных")
    print("Повторные записи об одном игроке")
    print("Ошибки в системе учета")
    
    df_clean = df.drop_duplicates()
    print(f"Размер после удаления дубликатов: {df_clean.shape}")
    print(f"Удалено строк: {initial_shape[0] - df_clean.shape[0]}")
else:
    print("Дубликатов не найдено")
    df_clean = df.copy()

# удалить ненужные денежные столбцы, если они есть
columns_to_drop = ['value', 'wage', 'release_clause']
df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')

for col in ['joined', 'loan_date_end']:
    if col in df_clean.columns:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        print(f"Столбец '{col}' преобразован в datetime: {df_clean[col].dtype}")

# итоговая статистика
print("\n" + "="*50)
print("ИТОГОВАЯ СТАТИСТИКА")
print("="*50)
print(f"Исходный размер данных: {initial_shape}")
print(f"Финальный размер данных: {df_clean.shape}")
print(f"Удалено строк: {initial_shape[0] - df_clean.shape[0]}")
print(f"Обработано столбцов: {len(df_clean.columns)}")

missing_percentage = (df_clean.isnull().sum().sum() / (df_clean.shape[0] * df_clean.shape[1])) * 100
print("\nКачество данных:")
print(f"Процент пропущенных значений: {missing_percentage:.2f}%")
print(f"Целостность данных: {100 - missing_percentage:.2f}%")
print(f"Готовность к анализу: {'да' if missing_percentage < 1 else 'требует дополнительной обработки'}")


# показать результат до и после
print("\n" + "="*50)
print("СРАВНЕНИЕ ДО И ПОСЛЕ ОЧИСТКИ")
print("="*50)

# загрузить исходные данные еще раз для сравнения
df_original = pd.read_csv('players.csv', encoding='utf-8')
print("СТОЛБЦЫ В df_original:")
print(df_original.columns.tolist())
print("\nПОСЛЕ ОЧИСТКИ")
print("Первые 3 строки очищенных данных:")
print(df_clean.head(3))

print("\nОСНОВНЫЕ ИЗМЕНЕНИЯ")
print(f"1. Столбцы: {len(df_original.columns)} → {len(df_clean.columns)}")
print(f"2. Строки: {len(df_original)} → {len(df_clean)}")
print(f"3. Размер в памяти: {df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB → {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

if 'positions_formatted' in df_clean.columns:
    print(f"4. Позиции преобразованы (пример): {df_clean['positions_formatted'].dropna().iloc[0] if len(df_clean['positions_formatted'].dropna()) > 0 else 'нет данных'}")

if 'club_name' in df_clean.columns and 'contract_start_year' in df_clean.columns:
    print(f"5. team_and_contract разделен на club_name, contract_start_year, contract_end_year")

# сохранение очищенных данных
df_clean.to_csv('players_cleaned.csv', index=False)
print(f"Очищенные данные сохранены в 'players_cleaned.csv'")

# cоответствие новых столбцов их исходным столбцам в оригинальных данных (используются имена столбцов из исходного CSV)
column_mapping = {
    'club_name': 'Team & Contract',
    'contract_start_year': 'Team & Contract',
    'contract_end_year': 'Team & Contract',
    'height_cm': 'Height',
    'weight_kg': 'Weight',
    'positions_formatted': 'Positions',
}

for col in ['Joined', 'Loan Date End']:
    if col in df_clean.columns and col in df_original.columns:
        column_mapping[col] = col


before_columns = []
for col in df_original.columns:
    if col in set(column_mapping.values()) and col not in before_columns:
        before_columns.append(col)

after_columns = [col for col in column_mapping.keys() if col in df_clean.columns]

positions_col = 'Positions'
positions_formatted_col = 'positions_formatted'
example_idx = []
if positions_col in df_original.columns and positions_formatted_col in df_clean.columns:
    multi_pos_idx = df_original[df_original[positions_col].astype(str).str.contains(r'[ ,]')].index
    example_idx = multi_pos_idx[:3]

print("\n")
print("Сравнение последних 3 строк (только новые/структурно изменённые столбцы)")
if len(example_idx) > 0:
    print("\nдо очистки")
    before_cols_with_dtype = [f"{col} ({df_original[col].dtype})" for col in before_columns]
    before_df = df_original.loc[example_idx, before_columns].copy()
    before_df.columns = before_cols_with_dtype
    print(before_df)
    print("\nпосле очистки")
    after_cols_with_dtype = [f"{col} ({df_clean[col].dtype})" for col in after_columns]
    after_df = df_clean.loc[example_idx, after_columns].copy()
    after_df.columns = after_cols_with_dtype
    print(after_df)
else:
    print("\nдо очистки")
    if before_columns:
        before_cols_with_dtype = [f"{col} ({df_original[col].dtype})" for col in before_columns]
        before_df = df_original[before_columns].tail(3).copy()
        before_df.columns = before_cols_with_dtype
        print(before_df)
    else:
        print("нет исходных столбцов для сравнения.")
    print("\nпосле очистки")
    after_cols_with_dtype = [f"{col} ({df_clean[col].dtype})" for col in after_columns]
    after_df = df_clean[after_columns].tail(3).copy()
    after_df.columns = after_cols_with_dtype
    print(after_df)

# Решенные проблемы в данных (теперь в комментариях):
# 1. Названия столбцов приведены к змеиному регистру и очищены от спецсимволов для единообразия и удобства обращения.
# 2. Рост игроков преобразован из футов и дюймов в сантиметры (метрическая система, привычная для анализа в России).
# 3. Вес игроков преобразован из фунтов в килограммы (метрическая система).
# 4. Звездные рейтинги (★) и звездные значения в столбцах удалены и преобразованы в числовой формат для анализа.
# 5. Столбцы с датами (например, joined) приведены к типу datetime для корректной работы с датами.
# 6. Все текстовые данные (кроме имен игроков и ссылок) приведены к нижнему регистру и очищены от лишних пробелов.
# 7. Пропущенные значения обработаны: заполнены средним, медианой или модой в зависимости от типа и распределения данных.
# 8. Дубликаты строк удалены для обеспечения уникальности записей.
# 9. Позиции игроков преобразованы: созданы dummy-переменные для каждой позиции, а также столбец positions_formatted без пробелов.
# 10. Столбец 'team_and_contract' разделен на отдельные поля: club_name, contract_start_year, contract_end_year для удобства анализа контрактов.
# 11. Типы данных оптимизированы (например, int64 → uint8/uint16, float64 → float32, object → category) для экономии памяти и ускорения вычислений.
# 12. Данные готовы к дальнейшему анализу, визуализации и машинному обучению.