import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 页面设置
st.set_page_config(
    page_title="🔥 木质素水热碳特性预测系统",
    page_icon="🔥",
    layout="wide"
)

# 模型路径
import os

MODEL_PATHS = {
    'C': os.path.join(os.path.dirname(__file__), 'C综合.pkl'),
    'H': os.path.join(os.path.dirname(__file__), 'H综合.pkl'),
    'O': os.path.join(os.path.dirname(__file__), 'O综合.pkl'),
    'N': os.path.join(os.path.dirname(__file__), 'N综合.pkl'),
    'FC': os.path.join(os.path.dirname(__file__), 'FC综合.pkl'),
    'VM': os.path.join(os.path.dirname(__file__), 'vm综合.pkl'),
    'ASH': os.path.join(os.path.dirname(__file__), 'ASH综合.pkl'),
    'HHV': os.path.join(os.path.dirname(__file__), 'HHV综合.pkl'),
    'EY': os.path.join(os.path.dirname(__file__), 'EY综合.pkl')
}


# 加载模型函数
@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        models[name] = joblib.load(path)
    return models

# 加载模型
try:
    models = load_models()
    st.sidebar.success("✅ 所有模型加载成功！")
except Exception as e:
    st.sidebar.error(f"❌ 模型加载失败：{e}")
    st.stop()

# 侧边栏输入参数
st.sidebar.header("⚙️ 输入参数设置")
with st.sidebar.expander("📌 基础参数", expanded=True):
    C = st.number_input("C (%)", 0.0, 100.0, 50.0)
    H = st.number_input("H (%)", 0.0, 100.0, 6.0)
    O = st.number_input("O (%)", 0.0, 100.0, 30.0)
    N = st.number_input("N (%)", 0.0, 100.0, 1.0)
    FC = st.number_input("FC (%)", 0.0, 100.0, 10.0)
    VM = st.number_input("VM (%)", 0.0, 100.0, 30.0)
    ASH = st.number_input("ASH (%)", 0.0, 100.0, 5.0)

with st.sidebar.expander("🔬 实验参数", expanded=True):
    HT = st.number_input("HT (°C)", 0, 2000, 800)
    Ht = st.number_input("Ht (s)", 0.0, 100.0, 10.0)

# 特征工程
@st.cache_data

def compute_features(input_dict):
    df = pd.DataFrame([input_dict])
    
    # 计算其他特征
    df['o_raw/c_raw'] = df['O'] / df['C'] * 12 / 16
    df['h_raw/c_raw'] = df['H'] / df['C'] * 12
    df['R'] = np.log(df['Ht'] * np.exp((df['HT'] - 100) / 14.75))
    df['HHV'] = 0.4059 * df['C']
    
    # 重命名原始特征列
    column_mapping = {
        'C': 'C-raw',
        'H': 'H-raw',
        'O': 'O-raw',
        'N': 'N-raw',
        'FC': 'FC-raw',
        'VM': 'VM-raw',
        'ASH': 'ASH-raw'
    }
    df = df.rename(columns=column_mapping)
    
    return df


# 主界面标题
st.title("🔥 木质素水热碳特性预测系统")

st.markdown("""
本页面基于集成机器学习模型，用于预测木质素水热碳关键参数。
""")
# 执行
if st.button("🚀 开始预测"):
    with st.spinner("🔄 正在计算，请稍候..."):
        # 输入数据
        input_data = {
            'C': C, 'H': H, 'O': O, 'N': N,
            'FC': FC, 'VM': VM, 'ASH': ASH,
            'HT': HT, 'Ht': Ht
        }

        features_df = compute_features(input_data)
        predictions = {}

        # 遍历模型预测
        for target, model in models.items():
            if target == 'O':
                X = features_df[['HT', 'Ht', 'C-raw', 'H-raw', 'O-raw', 'N-raw', 'FC-raw', 'VM-raw', 'ASH-raw']]
            else:
                X = features_df[['HT', 'Ht', 'C-raw', 'H-raw', 'O-raw', 'N-raw', 'FC-raw', 'VM-raw', 'ASH-raw', 'o_raw/c_raw', 'h_raw/c_raw', 'R', 'HHV']]

            predictions[target] = model.predict(X)[0]

        # 显示预测结果
        st.subheader("📊 预测结果展示")
        for key, value in predictions.items():
            st.metric(label=f"{key}预测值", value=f"{value:.2f}")



# 数据说明
with st.expander("📚 数据与公式说明", expanded=False):
    st.markdown(r"""
    **输入参数说明：**
    - **C/H/O/N (%)**: 燃料元素含量百分比
    - **FC/VM/ASH (%)**: 固定碳、挥发分、灰分含量
    - **HT (°C)**: 水热温度
    - **Ht (s)**: 水热时间

    **衍生特征计算公式：**
    - \( o_{raw}/c_{raw} = \frac{O}{C} \times \frac{12}{16} \)
    - \( h_{raw}/c_{raw} = \frac{H}{C} \times 12 \)
    - \( R = \ln\left(Ht \times e^{\frac{HT - 100}{14.75}}\right) \)
    - \( HHV = 0.4059 \times C \)
    """)

st.markdown("---")
st.caption("🧪 科研预测系统 | © 2025 内蒙古科技大学能源与环境学院")
