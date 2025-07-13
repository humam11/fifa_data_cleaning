#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
графический анализ данных fifa 2021
скрипт для создания диаграмм matplotlib и seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# настройка русского шрифта
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# настройка стиля
sns.set_style("whitegrid")
rcParams['figure.figsize'] = (12, 8)

def load_data():
    """загрузка данных из players_cleaned.csv"""
    try:
        df = pd.read_csv('players_cleaned.csv')
        print(f"данные загружены: {df.shape[0]} игроков, {df.shape[1]} признаков")
        return df
    except FileNotFoundError:
        print("ошибка: файл players_cleaned.csv не найден")
        return None

def create_age_rating_distribution(df):
    """распределение игроков по возрастным группам и рейтингу (matplotlib)"""
    print("создание диаграммы 1: распределение по возрастным группам и рейтингу...")
    age_bins = [0, 20, 25, 30, 35, 100]
    age_labels = ['16-20', '21-25', '26-30', '31-35', '35+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    rating_bins = [0, 60, 70, 80, 85, 100]
    rating_labels = ['<60', '60-70', '70-80', '80-85', '85+']
    df['rating_group'] = pd.cut(df['ova'], bins=rating_bins, labels=rating_labels)
    pivot_data = df.pivot_table(
        values='ova',
        index='age_group',
        columns='rating_group',
        aggfunc='count',
        fill_value=0
    )
    fig, ax = plt.subplots(figsize=(14, 10))
    pivot_data.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('распределение игроков по возрастным группам и рейтингу', 
                 fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('возрастные группы', fontsize=14, fontweight='bold')
    ax.set_ylabel('количество игроков', fontsize=14, fontweight='bold')
    ax.legend(title='рейтинг', title_fontsize=12, fontsize=10, bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontsize=10)
    plt.tight_layout()
    plt.savefig('age_rating_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("диаграмма сохранена как 'age_rating_distribution.png'")

def create_correlation_heatmap(df):
    """корреляционная матрица ключевых навыков (matplotlib)"""
    print("создание диаграммы 2: корреляционная матрица навыков...")
    key_skills = ['ova', 'pot', 'sprint_speed', 'dribbling', 'shot_power', 
                  'short_passing', 'stamina', 'strength', 'reactions', 'composure']
    key_skills = [col for col in key_skills if col in df.columns]
    correlation_matrix = df[key_skills].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(correlation_matrix, cmap='RdYlBu_r', aspect='auto')
    ax.set_xticks(range(len(key_skills)))
    ax.set_yticks(range(len(key_skills)))
    ax.set_xticklabels(key_skills, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(key_skills, fontsize=11)
    for i in range(len(key_skills)):
        for j in range(len(key_skills)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10,
                          fontweight='bold')
    ax.set_title('корреляционная матрица ключевых навыков игроков', 
                 fontsize=18, fontweight='bold', pad=25)
    plt.colorbar(im, ax=ax, label='коэффициент корреляции', shrink=0.8)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("диаграмма сохранена как 'correlation_heatmap.png'")

def create_nationality_pie(df):
    """распределение игроков по национальностям (matplotlib)"""
    print("создание диаграммы 3: распределение по национальностям...")
    top_countries = df['nationality'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(14, 10))
    wedges, texts, autotexts = ax.pie(top_countries.values, labels=top_countries.index, 
                                      autopct='%1.1f%%', startangle=90,
                                      colors=plt.cm.Set3(np.linspace(0, 1, len(top_countries))))
    ax.set_title('распределение игроков по топ-10 национальностям', 
                 fontsize=18, fontweight='bold', pad=25)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    plt.axis('equal')
    plt.savefig('nationality_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("диаграмма сохранена как 'nationality_distribution.png'")

def create_rating_boxplot(df):
    """распределение рейтинга по возрастным группам (seaborn)"""
    print("создание диаграммы 4: box plot рейтинга по возрастным группам...")
    if 'age_group' not in df.columns:
        age_bins = [0, 20, 25, 30, 35, 100]
        age_labels = ['16-20', '21-25', '26-30', '31-35', '35+']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    plt.figure(figsize=(14, 10))
    sns.boxplot(data=df, x='age_group', y='ova', palette='viridis')
    plt.title('распределение рейтинга по возрастным группам', 
              fontsize=18, fontweight='bold', pad=25)
    plt.xlabel('возрастные группы', fontsize=14, fontweight='bold')
    plt.ylabel('общий рейтинг', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for i, group in enumerate(df['age_group'].unique()):
        group_data = df[df['age_group'] == group]['ova']
        median = group_data.median()
        plt.text(i, median, f'медиана: {median:.1f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig('rating_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("диаграмма сохранена как 'rating_boxplot.png'")

def create_skills_comparison(df):
    """сравнение навыков по позициям (matplotlib)"""
    print("создание дополнительной диаграммы: сравнение навыков...")
    
    print(f"всего уникальных позиций: {df['positions_formatted'].nunique()}")
    print(f"топ-10 позиций: {df['positions_formatted'].value_counts().head(10).index.tolist()}")
    
    all_positions = df['positions_formatted'].value_counts().head(10).index.tolist()
    main_positions = []
    for pos in ['GK', 'CB', 'CM', 'ST', 'RW']:
        matching_positions = [p for p in all_positions if pos in str(p)]
        if matching_positions:
            main_positions.extend(matching_positions[:2])  # берем максимум 2 позиции для каждого типа
    
    if not main_positions:
        main_positions = all_positions[:5]
    
    print(f"выбранные позиции для анализа: {main_positions}")
    
    skills = ['sprint_speed', 'dribbling', 'shot_power', 'short_passing', 'stamina']
    # проверяем какие навыки есть в данных
    available_skills = [col for col in skills if col in df.columns]
    print(f"доступные навыки: {available_skills}")
    
    if not available_skills:
        print("ошибка: нет доступных навыков для анализа")
        return
    
    df_filtered = df[df['positions_formatted'].isin(main_positions)]
    print(f"игроков с выбранными позициями: {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print("ошибка: нет игроков с выбранными позициями")
        return
    
    skills_data = df_filtered.groupby('positions_formatted')[available_skills].mean()
    
    if skills_data.empty:
        print("ошибка: не удалось создать сводную таблицу")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    skills_data.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('средние значения навыков по позициям', 
                 fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('позиции', fontsize=14, fontweight='bold')
    ax.set_ylabel('среднее значение навыка', fontsize=14, fontweight='bold')
    ax.legend(title='навыки', title_fontsize=12, fontsize=11, bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig('skills_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("диаграмма сохранена как 'skills_comparison.png'")

def print_summary_statistics(df):
    """вывод сводной статистики по датасету fifa 2021"""
    print("\n" + "="*60)
    print("сводная статистика fifa 2021")
    print("="*60)
    print(f"общее количество игроков: {len(df):,}")
    print(f"количество национальностей: {df['nationality'].nunique()}")
    print(f"количество позиций: {df['positions_formatted'].nunique()}")
    print(f"возрастной диапазон: {df['age'].min()} - {df['age'].max()} лет")
    print(f"рейтинговый диапазон: {df['ova'].min()} - {df['ova'].max()}")
    print(f"\nсредние значения:")
    print(f"возраст: {df['age'].mean():.1f} лет")
    print(f"рост: {df['height_cm'].mean():.1f} см")
    print(f"вес: {df['weight_kg'].mean():.1f} кг")
    print(f"общий рейтинг: {df['ova'].mean():.1f}")
    print(f"потенциал: {df['pot'].mean():.1f}")
    print(f"\nтоп-5 стран по количеству игроков:")
    top_countries = df['nationality'].value_counts().head(5)
    for country, count in top_countries.items():
        percentage = (count / len(df)) * 100
        print(f"{country}: {count:,} игроков ({percentage:.1f}%)")

def main():
    """основная функция"""
    print("графический анализ данных fifa 2021")
    print("="*50)
    df = load_data()
    if df is None:
        return
    print_summary_statistics(df)
    try:
        create_age_rating_distribution(df)
        create_correlation_heatmap(df)
        create_nationality_pie(df)
        create_rating_boxplot(df)
        create_skills_comparison(df)
        print("\n" + "="*50)
        print("все диаграммы успешно созданы!")
        print("="*50)
        print("созданные файлы:")
        print("- age_rating_distribution.png")
        print("- correlation_heatmap.png")
        print("- nationality_distribution.png")
        print("- rating_boxplot.png")
        print("- skills_comparison.png")
    except Exception as e:
        print(f"ошибка при создании диаграмм: {e}")

if __name__ == "__main__":
    main() 