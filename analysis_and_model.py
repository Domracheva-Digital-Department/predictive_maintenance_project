import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from ucimlrepo import fetch_ucirepo
import io

def data_processing_app():
    # Создаем разделы для различных этапов
    sections = st.tabs(["Загрузка данных", "Анализ", "Обучение модели", "Прогнозирование"])
    
    # Инициализация состояния сессии
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'test_features' not in st.session_state:
        st.session_state.test_features = None
    if 'test_target' not in st.session_state:
        st.session_state.test_target = None
    if 'normalizer' not in st.session_state:
        st.session_state.normalizer = None
    if 'encoder' not in st.session_state:
        st.session_state.encoder = None
    
    # Вкладка 1: данные
    with sections[0]:
        st.header("Работа с источниками данных")
        
        source_type = st.radio(
            "Источник данных:",
            ["Импорт из UCI", "Локальный CSV-файл"]
        )
        
        if source_type == "Импорт из UCI":
            if st.button("Загрузить из UCI"):
                with st.spinner("Идет загрузка..."):
                    # Получение данных из репозитория
                    repo_data = fetch_ucirepo(id=601)
                    features = repo_data.data.features
                    targets = repo_data.data.targets
                    combined_data = pd.concat([features, targets], axis=1)
                    st.session_state.dataset = combined_data
                    st.success("Данные успешно загружены!")
                    
                    # Отображение метаданных
                    st.subheader("Метаописание набора данных")
                    st.write(f"Объем данных: {combined_data.shape[0]} записей")
                    st.write(f"Количество параметров: {combined_data.shape[1]}")
                    st.write("Пример данных:")
                    st.dataframe(combined_data.sample(5))
        
        else:
            uploaded_data = st.file_uploader("Выберите файл для загрузки", type="csv")
            if uploaded_data is not None:
                combined_data = pd.read_csv(uploaded_data)
                st.session_state.dataset = combined_data
                st.success("Файл успешно загружен!")
                
                # Информация о структуре данных
                st.subheader("Структурная информация")
                st.write(f"Записей: {combined_data.shape[0]}")
                st.write(f"Параметров: {combined_data.shape[1]}")
                st.write("Случайная выборка:")
                st.dataframe(combined_data.sample(5))
        
        if st.session_state.dataset is not None:
            st.subheader("Проверка целостности данных")
            na_values = st.session_state.dataset.isna().sum()
            st.write(na_values)
            
            if na_values.sum() == 0:
                st.success("Пропуски отсутствуют!")
            else:
                st.warning(f"Обнаружено {na_values.sum()} пропущенных значений.")
    
    # Раздел 2: Анализ данных
    with sections[1]:
        st.header("Исследование данных")
        
        if st.session_state.dataset is not None:
            current_data = st.session_state.dataset
            
            # Обработка данных
            st.subheader("Подготовка данных")
            
            # Удаление некорректных столбцов
            cols_to_remove = st.multiselect(
                "Исключить параметры:",
                current_data.columns,
                default=["UDI", "Product ID"] if "UDI" in current_data.columns and "Product ID" in current_data.columns else []
            )
            
            if cols_to_remove:
                current_data = current_data.drop(columns=cols_to_remove)
                st.write("Обновленный набор:")
                st.dataframe(current_data.head())
            
            # Конвертация категориальных данных
            if 'Type' in current_data.columns:
                st.subheader("Кодирование категорий")
                st.write("Преобразование параметра 'Type'")
                
                encoder = LabelEncoder()
                current_data['Type'] = encoder.fit_transform(current_data['Type'])
                st.session_state.encoder = encoder
                
                st.write("Результат преобразования:")
                st.dataframe(current_data.head())
            
            # Статистика
            st.subheader("Числовые характеристики")
            st.write("Основные метрики:")
            st.dataframe(current_data.describe())
            
            # Анализ целевого параметра
            st.subheader("Анализ целевого признака")
            
            target_var = st.selectbox(
                "Целевой параметр:",
                ["Machine failure", "Outcome"] if "Machine failure" in current_data.columns else ["Outcome"]
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts = current_data[target_var].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            ax.set_title(f"Распределение {target_var}")
            
            # Добавление процентных соотношений
            total_samples = len(current_data)
            for index, value in enumerate(value_counts.values):
                percent = value / total_samples * 100
                ax.text(index, value + 30, f"{percent:.1f}%", ha='center')
            
            st.pyplot(fig)
            
            # Матрица корреляций
            st.subheader("Анализ взаимосвязей")
            
            numerical_cols = current_data.select_dtypes(include=['number']).columns
            correlation = current_data[numerical_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(correlation, annot=True, cmap='Spectral', fmt=".2f", ax=ax)
            st.pyplot(fig)
            
            # Сохранение изменений
            st.session_state.dataset = current_data
            
            # Визуализация распределений
            st.subheader("Распределение признаков")
            
            chosen_features = st.multiselect(
                "Выбор параметров для анализа:",
                numerical_cols,
                default=numerical_cols[:min(4, len(numerical_cols))]
            )
            
            if chosen_features:
                fig, axes = plt.subplots(len(chosen_features), 2, figsize=(15, 4*len(chosen_features)))
                
                for i, param in enumerate(chosen_features):
                    # Распределение
                    sns.histplot(data=current_data, x=param, hue=target_var, kde=True, ax=axes[i, 0])
                    axes[i, 0].set_title(f"Распределение {param}")
                    
                    # Боксплот
                    sns.boxplot(data=current_data, x=target_var, y=param, ax=axes[i, 1])
                    axes[i, 1].set_title(f"Распределение {param} по классам")
                
                plt.tight_layout()
                st.pyplot(fig)
        
        else:
            st.warning("Требуется загрузка данных в разделе 'Загрузка данных'")
    
    # Раздел 3: Построение модели
    with sections[2]:
        st.header("Настройка модели")
        
        if st.session_state.dataset is not None:
            current_data = st.session_state.dataset
            
            # Выбор целевого параметра
            target_var = st.selectbox(
                "Целевой параметр:",
                ["Machine failure", "Outcome"] if "Machine failure" in current_data.columns else ["Outcome"],
                key="target_var_select"
            )
            
            # Выбор предикторов
            predictors = [col for col in current_data.columns if col != target_var]
            selected_predictors = st.multiselect(
                "Предикторы для модели:",
                predictors,
                default=predictors
            )
            
            if selected_predictors:
                # Разделение на выборки
                X = current_data[selected_predictors]
                y = current_data[target_var]
                
                test_part = st.slider("Доля тестовой выборки (%)", 10, 40, 20) / 100
                seed = st.number_input("Seed для воспроизводимости", 0, 100, 42)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_part, random_state=seed
                )
                
                # Нормализация
                if st.checkbox("Применить нормализацию", value=True):
                    normalizer = StandardScaler()
                    X_train = normalizer.fit_transform(X_train)
                    X_test = normalizer.transform(X_test)
                    st.session_state.normalizer = normalizer
                
                # Сохранение тестовых данных
                st.session_state.test_features = X_test
                st.session_state.test_target = y_test
                
                # Выбор алгоритма
                algorithm = st.selectbox(
                    "Алгоритм:",
                    ["Логистическая регрессия", "Случайный лес", "Градиентный бустинг", "Метод опорных векторов"]
                )
                
                # Параметры модели
                if algorithm == "Логистическая регрессия":
                    reg_strength = st.slider("Сила регуляризации", 0.01, 10.0, 1.0)
                    iterations = st.slider("Лимит итераций", 100, 1000, 100)
                    
                    model = LogisticRegression(C=reg_strength, max_iter=iterations, random_state=seed)
                
                elif algorithm == "Случайный лес":
                    trees = st.slider("Число деревьев", 10, 200, 100)
                    depth = st.slider("Глубина деревьев", 2, 20, 10)
                    
                    model = RandomForestClassifier(
                        n_estimators=trees,
                        max_depth=depth,
                        random_state=seed
                    )
                
                elif algorithm == "Градиентный бустинг":
                    estimators = st.slider("Количество оценщиков", 10, 200, 100, key="xgb_est")
                    lr = st.slider("Темп обучения", 0.01, 0.3, 0.1)
                    tree_depth = st.slider("Глубина", 2, 10, 6, key="xgb_depth")
                    
                    model = XGBClassifier(
                        n_estimators=estimators,
                        learning_rate=lr,
                        max_depth=tree_depth,
                        random_state=seed
                    )
                
                elif algorithm == "Метод опорных векторов":
                    C_param = st.slider("Параметр C", 0.1, 10.0, 1.0, key="svm_c")
                    kernel_type = st.selectbox("Тип ядра", ["линейное", "радиальное", "полиномиальное"])
                    
                    model = SVC(C=C_param, kernel=kernel_type, probability=True, random_state=seed)
                
                # Обучение модели
               
                if st.button("🚀 Запустить обучение модели", type="primary", use_container_width=True):
                    with st.status("**Процесс обучения**", expanded=True) as status:
                        try:
                            # Блок обучения
                            st.write(f"🔧 Обучение {algorithm}...")
                            progress_bar = st.progress(0, text="Инициализация модели")
                            model.fit(X_train, y_train)
                            progress_bar.progress(30, text="Оптимизация параметров")
            
                            # Сохранение модели
                            st.session_state.trained_model = model
                            progress_bar.progress(70, text="Сохранение результатов")
            
                            # Вычисление метрик
                            predictions = model.predict(X_test)
                            proba = model.predict_proba(X_test)[:, 1]
                            acc = accuracy_score(y_test, predictions)
                            cm = confusion_matrix(y_test, predictions)
                            report = classification_report(y_test, predictions)
                            auc_score = roc_auc_score(y_test, proba)
            
                            status.update(label="Обучение завершено! ✅", state="complete")
                            progress_bar.progress(100)

                            # Визуализация результатов
                            st.balloons()
                            st.success(f"📊 Результаты: Точность - {acc:.2%} | AUC - {auc_score:.3f}")
            
                            # Основные метрики
                            with st.container(border=True):
                                st.subheader("📈 Ключевые показатели")
                                cols = st.columns(3)
                                with cols[0]:
                                    st.metric("Точность", f"{acc:.2%}", help="Доля верных предсказаний")
                                with cols[1]:
                                    st.metric("AUC-ROC", f"{auc_score:.3f}", help="Площадь под ROC-кривой")
                                with cols[2]:
                                    st.metric("Примеров обучено", X_train.shape[0], help="Размер обучающей выборки")

                            # Визуализация в табах
                            eval_tabs = st.tabs(["📊 Матрица ошибок", "📈 ROC-анализ", "📋 Отчет", "🎯 Важность признаков"])
            
                            with eval_tabs[0]:
                                # Матрица ошибок
                                fig, ax = plt.subplots(figsize=(8,6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn', 
                                          linewidths=.5, annot_kws={"size": 14}, 
                                          xticklabels=['0', '1'], 
                                          yticklabels=['0', '1'])
                                ax.set_xlabel('Прогноз', fontsize=12)
                                ax.set_ylabel('Факт', fontsize=12)
                                ax.set_title('Матрица неточностей', pad=20)
                                st.pyplot(fig)

                            with eval_tabs[1]:
                                # ROC-кривая
                                fig, ax = plt.subplots(figsize=(8,6))
                                fpr, tpr, _ = roc_curve(y_test, proba)
                                plt.plot(fpr, tpr, color='#2ecc71', lw=2, 
                                        label=f'{algorithm} (AUC = {auc_score:.3f})')
                                plt.plot([0, 1], [0, 1], color='#95a5a6', linestyle='--')
                                plt.fill_between(fpr, tpr, alpha=0.1, color='#2ecc71')
                                plt.xlabel('Ложноположительная частота', fontsize=12)
                                plt.ylabel('Истинноположительная частота', fontsize=12)
                                plt.legend(loc='lower right')
                                st.pyplot(fig)

                            with eval_tabs[2]:
                                # Отчет классификации
                                st.subheader("📑 Статистика классификации")
                                st.code(report, language='text')

                            with eval_tabs[3]:
                                # Важность признаков
                                if algorithm in ["Случайный лес", "Градиентный бустинг"]:
                                    importance = model.feature_importances_
                                    feat_importance = pd.DataFrame({
                                        'Признак': selected_predictors,
                                        'Вклад': importance
                                    }).sort_values('Вклад', ascending=False)

                                    fig, ax = plt.subplots(figsize=(10,6))
                                    sns.barplot(x='Вклад', y='Признак', data=feat_importance,
                                               palette='viridis', edgecolor='black')
                                    plt.title('Ранжирование признаков по важности', pad=15)
                                    plt.xlabel('Относительная важность', fontsize=12)
                                    plt.ylabel('')
                                    st.pyplot(fig)
                                else:
                                    st.info("ℹ️ Для выбранного алгоритма анализ важности признаков недоступен")

                        except Exception as e:
                            status.update(label="Ошибка обучения! ❌", state="error")
                            st.error(f"Произошла ошибка: {str(e)}")
                            st.exception(e)

                else:
                    st.warning("⚠️ Для начала обучения выберите хотя бы один предиктор")
    
        # Раздел 4: Прогнозирование
    with sections[3]:
        st.header("Прогнозная аналитика")
        
        if st.session_state.trained_model is not None and st.session_state.dataset is not None:
            current_data = st.session_state.dataset
            model = st.session_state.trained_model
            
            st.subheader("🔮 Параметры для прогнозирования")

            with st.form("prediction_form"):
                input_values = {}
    
                # Секция категориальных параметров
                if 'Type' in current_data.columns:
                    st.markdown("#### 🏷 Категориальные параметры")
                    type_col, _ = st.columns([3, 7])
                    with type_col:
                        selected_type = st.selectbox(
                            "Тип оборудования",
                            options=['L', 'M', 'H'],
                            format_func=lambda x: f"{x} (низкий)" if x == 'L' else 
                                                  f"{x} (средний)" if x == 'M' else 
                                                  f"{x} (высокий)",
                            help="Выберите категорию оборудования"
                        )
                        input_values['Type'] = {'L': 0, 'M': 1, 'H': 2}[selected_type]
    
                st.divider()
    
                # Секция числовых параметров
                st.markdown("#### 📊 Числовые параметры")
                numeric_params = [col for col in current_data.columns 
                                 if col not in ['Type', 'Machine failure', 'Outcome']]
    
                cols = st.columns(2)
                for i, param in enumerate(numeric_params):
                    with cols[i % 2]: 
                        min_val = float(current_data[param].min())
                        max_val = float(current_data[param].max())
                        avg_val = float(current_data[param].mean())
            
                        step = 0.01 if any(kw in param.lower() for kw in ['temp', 'torque']) else 1.0
            
                        input_values[param] = st.number_input(
                            label=f"**{param}**",
                            min_value=min_val,
                            max_value=max_val,
                            value=avg_val,
                            step=step,
                            format="%.2f" if step < 1.0 else None,
                            help=f"Допустимый диапазон: {min_val:.2f} - {max_val:.2f}",
                            key=f"input_{param}"
                        )
    
                st.divider()
    
                # Кнопка прогноза с иконкой
                submitted = st.form_submit_button(
                    "🚀 Рассчитать",
                    help="Нажмите для выполнения прогноза",
                    use_container_width=True
                )
            
            if submitted:
                input_df = pd.DataFrame([input_values])
                
                if st.session_state.normalizer is not None:
                    scaled_input = st.session_state.normalizer.transform(input_df)
                else:
                    scaled_input = input_df
                
                result = model.predict(scaled_input)
                probability = model.predict_proba(scaled_input)[:, 1]
                
                st.subheader("Результат прогноза")
                
                if result[0] == 1:
                    st.error(f"Прогноз: Высокий риск отказа (класс 1)")
                else:
                    st.success(f"Прогноз: Штатный режим (класс 0)")
                
                st.write(f"Вероятность аномалии: {probability[0]:.4f}")
                
                # Визуализация вероятности
                fig, ax = plt.subplots(figsize=(10, 2))
                try:
                    plt.style.use('seaborn-v0_8-whitegrid')  # Для matplotlib >= 3.6
                except:
                    try:
                        plt.style.use('seaborn-whitegrid')    # Для старых версий с установленным seaborn
                    except:
                        plt.style.use('ggplot')               # Резервный вариант

                # Основная визуализация
                bar = ax.barh(['RISK'], [probability[0]], 
                             color='#ff6b6b' if result[0] == 1 else '#1dd1a1',
                             height=0.6, 
                             edgecolor='black',
                             linewidth=0.8,
                             alpha=0.9)

                bar[0].set_capstyle('round')
                ax.axvline(x=0.5, color='#b2bec3', linestyle='--', linewidth=1.2, alpha=0.8)
                ax.text(probability[0] + 0.02, 0, f"{probability[0]:.4f}", va='center', ha='left', 
                        fontsize=12, fontweight='bold', color='#2d3436')
                ax.spines[['top', 'right', 'left']].set_visible(False)
                ax.tick_params(axis='y', which='both', left=False, labelsize=12)
                ax.set_xlim(0, 1)
                ax.set_xticks([])
                ax.set_facecolor('#f8f9fa')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Рекомендации
                st.subheader("Интерпретация")
                if result[0] == 1:
                    st.write("""
                    Обнаружен повышенный риск технического сбоя. Рекомендуется:
                    - Провести диагностику оборудования
                    - Проверить параметры технологического процесса
                    """)
                else:
                    st.write("""
                    Оборудование работает в штатном режиме. Рекомендуется:
                    - Продолжить мониторинг показателей
                    - Соблюдать график планового ТО
                    """)
        
        else:
            st.warning("Необходимо обучить модель в разделе 'Обучение модели'")

if __name__ == "__main__":
    data_processing_app()