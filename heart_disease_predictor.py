import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from matplotlib import font_manager

# 设置中文字体（可选）
try:
    font_path = "SimHei.ttf"  # 替换为你的中文字体路径
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
except:
    st.warning("中文字体加载失败，将使用默认字体")

# 模型加载
@st.cache_resource
def load_model():
    try:
        model = joblib.load('cat.pkl')
        if not hasattr(model, 'predict'):
            st.error("加载的模型无效！请检查模型文件")
            st.stop()
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()

model = load_model()

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

# 界面布局
st.title("抑郁症风险预测模型 (Depression Risk-Prediction Model)")
st.header("请输入以下特征值:")

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
if st.button("预测/Predict"):
    try:
        # 创建DataFrame并确保列顺序
        features = pd.DataFrame([feature_values])
        
        # 类型转换
        for feature, props in feature_ranges.items():
            if props["type"] == "categorical":
                features[feature] = features[feature].astype(int)
        
        # 检查特征匹配
        if hasattr(model, 'feature_names_in_'):
            missing = set(model.feature_names_in_) - set(features.columns)
            if missing:
                st.error(f"缺少必要特征: {missing}")
                st.stop()
            
            # 重新排序特征
            features = features[model.feature_names_in_]
        
        # 进行预测
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]
        probability = predicted_proba[predicted_class] * 100

        # 结果显示
        risk_text = "高风险/High risk" if predicted_class == 1 else "低风险/Low risk"
        result_text = f"预测概率/Predicted probability: {probability:.2f}% ({risk_text})"
        
        fig, ax = plt.subplots(figsize=(10,2))
        ax.text(0.5, 0.7, result_text, 
                fontsize=14, ha='center', va='center')
        ax.axis('off')
        st.pyplot(fig)

        # SHAP解释
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            
            plt.figure()
            if isinstance(shap_values, list):
                # 多分类情况
                shap.force_plot(
                    explainer.expected_value[predicted_class],
                    shap_values[predicted_class][0],
                    features.iloc[0],
                    matplotlib=True,
                    show=False
                )
            else:
                # 二分类情况
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0],
                    features.iloc[0],
                    matplotlib=True,
                    show=False
                )
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.warning(f"SHAP解释生成失败: {str(e)}")

    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}")
        st.error("请检查输入数据是否正确")

# 添加模型信息展示
if st.checkbox("显示模型信息/Show model info"):
    if hasattr(model, 'feature_names_in_'):
        st.write("模型特征顺序/Model feature order:", model.feature_names_in_)
    st.write("模型参数/Model parameters:", model.get_params())
