import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np

# 1. 模型加载
@st.cache_resource
def load_model():
    try:
        model = joblib.load('cat.pkl')
        if not hasattr(model, 'predict'):
            st.error("Invalid model file!")
            st.stop()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model = load_model()

# 2. 特征定义（保持不变）
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

# 3. 创建预处理转换器
categorical_features = [name for name, props in feature_ranges.items() if props["type"] == "categorical"]
numerical_features = [name for name, props in feature_ranges.items() if props["type"] == "numerical"]

preprocessor = make_column_transformer(
    (OneHotEncoder(categories=[props["options"] for props in feature_ranges.values() if props["type"] == "categorical"]), 
     categorical_features),
    remainder='passthrough'
)

# 4. 拟合预处理器（使用所有可能的分类值）
sample_data = {feature: [props["options"][0]] for feature, props in feature_ranges.items()}
sample_df = pd.DataFrame(sample_data)
preprocessor.fit(sample_df)

# 5. 获取特征名称
def get_feature_names(column_transformer):
    feature_names = []
    for name, transformer, features in column_transformer.transformers_:
        if name == 'remainder':
            continue
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(features))
        else:
            feature_names.extend(features)
    return feature_names

feature_names = get_feature_names(preprocessor)

# 6. Streamlit界面
st.title("Depression Risk Prediction")
feature_values = {}

for feature, props in feature_ranges.items():
    if props["type"] == "numerical":
        feature_values[feature] = st.number_input(
            props["desc"],
            min_value=props["min"],
            max_value=props["max"],
            value=props["default"],
            step=props.get("step", 1.0),
            format=props.get("format", "%f")
        )
    else:
        feature_values[feature] = st.selectbox(
            props["desc"],
            options=props["options"]
        )

if st.button("Predict"):
    try:
        # 7. 准备输入数据
        input_df = pd.DataFrame([feature_values])
        
        # 8. 应用预处理
        encoded_data = preprocessor.transform(input_df)
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
        
        # 9. 确保特征匹配
        if hasattr(model, 'feature_names_in_'):
            missing = set(model.feature_names_in_) - set(encoded_df.columns)
            extra = set(encoded_df.columns) - set(model.feature_names_in_)
            
            if missing:
                st.error(f"Missing features: {missing}")
                st.stop()
            
            encoded_df = encoded_df[model.feature_names_in_]
        
        # 10. 进行预测
        prediction = model.predict(encoded_df)
        proba = model.predict_proba(encoded_df)
        
        # 11. 显示结果
        st.success(f"Prediction: {'High risk' if prediction[0] == 1 else 'Low risk'}")
        st.metric("Probability", f"{proba[0][prediction[0]]*100:.2f}%")
        
        # 12. SHAP解释
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(encoded_df)
        
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Model expects:", getattr(model, 'feature_names_in_', "Unknown"))
        st.write("Features provided:", encoded_df.columns.tolist() if 'encoded_df' in locals() else "Not generated")
