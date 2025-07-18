import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from matplotlib import font_manager

# 模型加载
model = joblib.load('cat.pkl')

# 特征定义
feature_ranges = {
    'gender': {"type": "categorical", "options": [1, 2], "desc": "性别/Gender (1:男/Male, 2:女/Female)"},
    'srh': {"type": "categorical", "options": [1,2,3,4,5], "desc": "自评健康/Self-rated health (1-5: 很差/Very poor 到 很好/Very good)"},
    'adlab_c': {"type": "categorical", "options": [0,1,2,3,4,5,6], "desc": "日常活动能力/Activities of daily living (0-6: 无/None 到 完全依赖/Complete dependence)"},
    'arthre': {"type": "categorical", "options": [0, 1], "desc": "关节炎/Arthritis (0:无/No, 1:有/Yes)"},
    'digeste': {"type": "categorical", "options": [0, 1], "desc": "消化系统问题/Digestive issues (0:无/No, 1:有/Yes)"},
    'retire': {"type": "categorical", "options": [0, 1], "desc": "退休状态/Retirement status (0:未退休/Not retired, 1:已退休/Retired)"},
    'satlife': {"type": "categorical", "options": [1,2,3,4,5], "desc": "生活满意度/Life satisfaction (1-5: 非常不满意/Very dissatisfied 到 非常满意/Very satisfied)"},
    'sleep': {
        "type": "numerical", 
        "min": 0.0, 
        "max": 24.0, 
        "default": 8.0, 
        "desc": "睡眠时长/Sleep duration (小时/hours)",
        "step": 0.1,
        "format": "%.1f"
    },
    'disability': {"type": "categorical", "options": [0, 1], "desc": "残疾/Disability (0:无/No, 1:有/Yes)"},
    'internet': {"type": "categorical", "options": [0, 1], "desc": "互联网使用/Internet use (0:不使用/No, 1:使用/Yes)"},
    'hope': {"type": "categorical", "options": [1,2,3,4], "desc": "希望程度/Hope level (1-4: 很低/Very low 到 很高/Very high)"},
    'fall_down': {"type": "categorical", "options": [0, 1], "desc": "跌倒史/Fall history (0:无/No, 1:有/Yes)"},
    'eyesight_close': {"type": "categorical", "options": [1,2,3,4,5], "desc": "视力/Near vision (1-5: 很差/Very poor 到 很好/Very good)"},
    'hear': {"type": "categorical", "options": [1,2,3,4,5], "desc": "听力/Hearing (1-5: 很差/Very poor 到 很好/Very good)"},
    'edu': {"type": "categorical", "options": [1,2,3,4], "desc": "教育程度/Education level (1:小学以下/Below Primary, 2:小学/Primary, 3:中学/Secondary, 4:中学以上/Above Secondary)"},
    'pension': {"type": "categorical", "options": [0, 1], "desc": "养老保险/Pension (0:无/No, 1:有/Yes)"},
    'pain': {"type": "categorical", "options": [0, 1], "desc": "慢性疼痛/Chronic pain (0:无/No, 1:有/Yes)"}
}

# 创建独热编码器
categorical_features = [name for name, props in feature_ranges.items() if props["type"] == "categorical"]
numerical_features = [name for name, props in feature_ranges.items() if props["type"] == "numerical"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 准备训练数据格式（仅用于拟合编码器）
sample_data = {feature: [props["options"][0]] for feature, props in feature_ranges.items()}
sample_df = pd.DataFrame(sample_data)
preprocessor.fit(sample_df)  # 拟合编码器

# 界面布局
st.title("Depression Risk-Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

# 输入表单
feature_values = {}
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=properties["desc"],
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            step=properties.get("step", 1.0),
            format=properties.get("format", "%f"),
            key=f"num_{feature}"
        )
    else:
        value = st.selectbox(
            label=properties["desc"],
            options=properties["options"],
            key=f"cat_{feature}"
        )
    feature_values[feature] = value

# 预测与解释
if st.button("Predict"):
    try:
        # 创建DataFrame
        input_df = pd.DataFrame([feature_values])
        
        # 应用独热编码
        encoded_data = preprocessor.transform(input_df)
        
        # 获取特征名称
        feature_names = numerical_features.copy()
        for col in categorical_features:
            for opt in feature_ranges[col]["options"]:
                feature_names.append(f"{col}_{opt}")
        
        # 创建编码后的DataFrame
        encoded_df = pd.DataFrame(encoded_data.toarray(), columns=feature_names)
        
        # 确保列顺序与模型训练时一致
        if hasattr(model, 'feature_names_in_'):
            # 添加缺失的特征列（设为0）
            for col in model.feature_names_in_:
                if col not in encoded_df.columns:
                    encoded_df[col] = 0
            encoded_df = encoded_df[model.feature_names_in_]
        
        # 进行预测
        predicted_class = model.predict(encoded_df)[0]
        predicted_proba = model.predict_proba(encoded_df)[0]
        probability = predicted_proba[predicted_class] * 100

        # 结果显示
        text_en = f"Predicted probability: {probability:.2f}% ({'High risk' if predicted_class == 1 else 'Low risk'})"
        fig, ax = plt.subplots(figsize=(10,2))
        ax.text(0.5, 0.7, text_en, 
                fontsize=14, ha='center', va='center', fontname='Arial')
        ax.axis('off')
        st.pyplot(fig)

        # SHAP解释
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(encoded_df)
        
        if isinstance(shap_values, list):
            shap_values_class = shap_values[predicted_class][0]
            expected_value = explainer.expected_value[predicted_class]
        else:
            shap_values_class = shap_values[0]
            expected_value = explainer.expected_value

        plt.figure()
        shap_plot = shap.force_plot(
            expected_value,
            shap_values_class,
            encoded_df.iloc[0],
            matplotlib=True,
            show=False
        )
        st.pyplot(plt.gcf())
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
