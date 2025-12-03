# 1. 安装依赖（首次运行时执行，之后可注释）
# !pip install streamlit pandas numpy scikit-learn --upgrade

# 2. 导入核心库
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 3. 页面配置（标题、图标）
st.set_page_config(
    page_title="🧒 儿童自闭症早期筛查工具",
    page_icon="🧒",
    layout="centered"  # 居中布局，适合移动端和电脑端
)

# 4. 加载并预处理数据（支持上传文件，增强容错）
@st.cache_resource  # 缓存数据，避免重复加载
def load_and_preprocess_data(uploaded_file):
    try:
        # 从上传文件读取数据（兼容中文编码）
         try:
        # 方案A：用户上传自定义数据集
        if uploaded_file:
            df = pd.read_csv(uploaded_file, encoding_errors='ignore')
        # 方案B：使用内置的公开自闭症数据集（部署后无需用户上传）
        else:
            # 公开数据集URL（GitHub上的标准自闭症筛查数据集，永久有效）
            public_dataset_url = "https://raw.githubusercontent.com/datasets-io/autism-screening/master/data/autism_screening.csv"
            df = pd.read_csv(public_dataset_url, encoding_errors='ignore')
        
        # 后续的数据处理逻辑不变（目标列匹配、特征编码等，保持原代码）
        # ...（此处省略原数据处理代码，直接复用之前的逻辑）...
        
        return (X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test, required_features, encoders, scaler)
    except Exception as e:
        st.error(f"数据预处理失败：{str(e)}")
        raise
        
        # 定义目标列（兼容多种列名写法）
        target_col = None
        target_candidates = ["class/asd", "asd", "class", "autism", "是否自闭症", "诊断结果"]
        for col in df.columns:
            if col.strip().lower() in target_candidates:
                target_col = col
                break
        if not target_col:
            raise ValueError("未找到目标列！需包含'Class/ASD'、'ASD'、'是否自闭症'等类似列名")
        
        # 定义必需特征列（核心特征，缺失则无法运行）
        required_features = ["age", "gender", "jundice", "austim"] + [f"A{i}_Score" for i in range(1, 11)]
        required_candidates = {
            "age": ["age", "年龄", "岁数"],
            "gender": ["gender", "性别", "sex"],
            "jundice": ["jundice", "黄疸", "出生黄疸"],
            "austim": ["austim", "自闭症亲属", "家族自闭症史"],
            "A1_Score": ["a1_score", "a1", "问题1"],
            "A2_Score": ["a2_score", "a2", "问题2"],
            "A3_Score": ["a3_score", "a3", "问题3"],
            "A4_Score": ["a4_score", "a4", "问题4"],
            "A5_Score": ["a5_score", "a5", "问题5"],
            "A6_Score": ["a6_score", "a6", "问题6"],
            "A7_Score": ["a7_score", "a7", "问题7"],
            "A8_Score": ["a8_score", "a8", "问题8"],
            "A9_Score": ["a9_score", "a9", "问题9"],
            "A10_Score": ["a10_score", "a10", "问题10"]
        }
        
        # 匹配实际列名（兼容大小写、空格、中文）
        feat_mapping = {}
        df_cols_lower = [col.strip().lower() for col in df.columns]
        missing_features = []
        for feat, candidates in required_candidates.items():
            matched = False
            for col_idx, col in enumerate(df.columns):
                if any(cand.lower() in df_cols_lower[col_idx] for cand in candidates):
                    feat_mapping[feat] = col
                    matched = True
                    break
            if not matched:
                missing_features.append(feat)
        if missing_features:
            raise ValueError(f"缺失必需特征列：{', '.join(missing_features)}！请确保CSV包含年龄、性别、A1-A10评分等列")
        
        # 提取特征和目标变量
        X = df[[feat_mapping[feat] for feat in required_features]]
        y = df[target_col]
        
        # 处理特征列缺失值和编码
        encoders = {}
        for col in X.columns:
            # 填充缺失值（分类特征用众数，数值特征用中位数）
            if X[col].dtype == "object":
                X[col] = X[col].fillna(X[col].mode()[0])
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders[col] = le
            else:
                X[col] = X[col].fillna(X[col].median())
        
        # 处理目标变量（统一编码为0/1）
        y = y.fillna(y.mode()[0])
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y)
        encoders["target"] = y_encoder
        
        # 数据拆分（分层抽样，保证类别分布一致）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 特征标准化（逻辑回归和SVM必需）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return (
            X_train, X_train_scaled, X_test, X_test_scaled,
            y_train, y_test, required_features, encoders, scaler
        )
    except Exception as e:
        st.error(f"数据预处理失败：{str(e)}")
        raise  # 抛出异常，终止后续流程

# 5. 训练3个预测模型（逻辑回归/随机森林/SVM）
@st.cache_resource  # 缓存模型，避免重复训练
def train_models(uploaded_file):
    data = load_and_preprocess_data(uploaded_file)
    X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test, feature_cols, encoders, scaler = data
    
    # 逻辑回归（增加正则化，避免过拟合）
    lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr.fit(X_train_scaled, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
    
    # 随机森林（调整参数，提升泛化能力）
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    
    # SVM（优化核函数参数）
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    
    return {
        "逻辑回归": (lr, lr_acc),
        "随机森林": (rf, rf_acc),
        "支持向量机": (svm, svm_acc)
    }, feature_cols, encoders, scaler

# 6. 定义通俗化问题（核心修复：给所有组件添加label参数）
def get_user_questions():
    # 所有组件均补充label参数（Streamlit必填），格式统一
    questions = {
        "age": (
            "1. 孩子的年龄是？",
            "slider",
            {
                "label": "年龄（岁）",  # 滑块必填label
                "min_value": 1, "max_value": 18, "value": 6, "step": 1,
                "help": "请选择孩子当前实际年龄，范围1-18岁"
            }
        ),
        "gender": (
            "2. 孩子的性别是？",
            "radio",
            {
                "label": "性别选择",  # Radio必填label
                "options": ["男", "女"], "index": 0,
                "help": "请选择孩子的生理性别"
            }
        ),
        "jundice": (
            "3. 孩子出生时是否有黄疸？",
            "radio",
            {
                "label": "出生黄疸",  # Radio必填label
                "options": ["是", "否"], "index": 1,
                "help": "黄疸指出生后皮肤、眼白发黄的情况，通常出生后2-3天出现"
            }
        ),
        "austim": (
            "4. 家族中是否有自闭症亲属？",
            "radio",
            {
                "label": "家族自闭症史",  # Radio必填label
                "options": ["是", "否"], "index": 1,
                "help": "包括父母、兄弟姐妹、祖父母等直系或旁系亲属"
            }
        ),
        "A1_Score": (
            "5. 孩子是否倾向于避免与他人进行眼神接触？",
            "radio",
            {
                "label": "避免眼神接触",  # Radio必填label
                "options": ["是", "否"], "index": 1,
                "help": "比如说话时不看对方眼睛，或刻意移开视线，持续时间较长"
            }
        ),
        "A2_Score": (
            "6. 孩子是否很少主动与他人发起对话？",
            "radio",
            {
                "label": "主动发起对话",  # Radio必填label
                "options": ["是", "否"], "index": 1,
                "help": "比如很少主动说“你好”“一起玩吗”，多是被动回应"
            }
        ),
        "A3_Score": (
            "7. 孩子是否难以理解他人的情绪（如生气、开心）？",
            "radio",
            {
                "label": "理解他人情绪",  # Radio必填label
                "options": ["是", "否"], "index": 1,
                "help": "比如别人难过时无反应，或无法分辨他人的开心/生气表情"
            }
        ),
        "A4_Score": (
            "8. 孩子是否喜欢重复做同一个动作（如拍手、转圈）？",
            "radio",
            {
                "label": "重复动作",  # Radio必填label
                "options": ["是", "否"], "index": 1,
                "help": "重复动作难以被打断，且不觉得无聊（如反复拍手、转圈、排列物品）"
            }
        ),
        "A5_Score": (
            "9. 孩子是否对特定物品有异常强烈的兴趣？",
            "radio",
            {
                "label": "异常强烈兴趣",  # Radio必填label
                "options": ["是", "否"], "index": 1,
                "help": "比如只关注玩具车/绘本，对其他玩具完全不感兴趣，持续数月以上"
            }
        ),
        "A6_Score": (
            "10. 孩子是否难以适应日常流程的变化？",
            "radio",
            {
                "label": "适应流程变化",  # Radio必填label
                "options": ["是", "否"], "index": 1,
                "help": "比如换衣服顺序、改吃饭时间时，会哭闹、抗拒或极度焦虑"
            }
        ),
        "A7_Score": (
            "11. 孩子是否很少用手势表达需求？",
            "radio",
            {
                "label": "用手势表达",  # Radio必填label
                "options": ["是", "否"], "index": 1,
                "help": "比如想要玩具时，不会用手指，只会拉大人的手或哭闹"
            }
        ),
        "A8_Score": (
            "12. 孩子是否对声音/光线/触觉特别敏感？",
            "radio",
            {
                "label": "感官敏感",  # Radio必填label
                "options": ["是", "否"], "index": 1,
                "help": "比如听到吸尘器声音捂耳朵，讨厌穿有标签的衣服，怕强光"
            }
        ),
        "A9_Score": (
            "13. 孩子是否很少参与假装游戏？",
            "radio",
            {
                "label": "参与假装游戏",  # Radio必填label
                "options": ["是", "否"], "index": 1,
                "help": "比如不会假装“给娃娃喂饭”“开玩具车”“假扮医生”等"
            }
        ),
        "A10_Score": (
            "14. 孩子是否难以记住日常小事？",
            "radio",
            {
                "label": "记住日常小事",  # Radio必填label
                "options": ["是", "否"], "index": 1,
                "help": "比如经常忘记自己的水杯放在哪里，或忘记每天要做的固定流程"
            }
        )
    }
    return questions

# 7. 网页交互逻辑（最终稳定版）
def main():
    # 页面标题和说明
    st.title("🧒 儿童自闭症早期筛查工具")
    st.markdown("""
    ### 工具说明
    该工具基于机器学习模型（逻辑回归+随机森林+支持向量机），通过14个通俗问题筛查儿童自闭症风险。
    - 适用人群：1-18岁儿童青少年
    - 结果性质：仅为初步参考，**不能替代专业医生诊断**
    - 建议：若结果提示“有倾向”或家长有疑虑，及时咨询儿科/发育行为科医生
    """)
    st.divider()
    
    # 第一步：上传数据文件
    st.subheader("📤 第一步：上传数据集（可选）")
    uploaded_file = st.file_uploader(
        "请上传自定义自闭症筛查数据集（CSV格式），不上传则使用默认公开数据集",
        type=["csv"],
        help="数据集需包含：1. 目标列（如'Class/ASD'）；2. 特征列（年龄、性别、A1-A10评分等）"
    )
    
    if not uploaded_file:
        st.info("请先上传CSV格式的数据集，工具将基于您的数据训练模型")
        st.stop()
    
    # 第二步：加载数据和训练模型（增强异常处理）
    models, feature_cols, encoders, scaler = None, None, None, None
    try:
        with st.spinner("📊 正在加载数据并训练模型..."):
            models, feature_cols, encoders, scaler = train_models(uploaded_file)
        st.success("✅ 模型训练成功！以下是模型测试集准确率：")
        # 展示模型准确率
        acc_df = pd.DataFrame({
            "模型名称": list(models.keys()),
            "测试集准确率": [f"{acc:.2%}" for _, acc in models.values()]
        })
        st.table(acc_df)
    except Exception as e:
        st.error(f"❌ 数据处理/模型训练失败：{str(e)}")
        st.info("💡 排查建议：\n1. 检查CSV文件列名是否包含目标列和必需特征列\n2. 确保文件编码为UTF-8（无乱码）\n3. 特征列无过多缺失值")
        st.stop()
    
    # 第三步：收集用户输入
    st.divider()
    st.subheader("📝 第二步：请回答以下14个问题")
    st.markdown("⚠️ 请根据孩子日常表现如实选择，结果将更准确")
    
    # 初始化Session State
    if "user_input" not in st.session_state:
        st.session_state.user_input = {}
    
    # 获取所有问题（确保每个组件都有label）
    questions = get_user_questions()
    
    # 逐个展示问题（分两列布局，更美观）
    cols = st.columns(2)
    question_list = list(questions.items())
    for idx, (feat_name, (q_text, comp_type, params)) in enumerate(question_list):
        with cols[idx % 2]:  # 交替放入左右两列
            st.markdown(f"**{q_text}**")
            if comp_type == "slider":
                val = st.slider(**params, key=f"slider_{feat_name}")
            elif comp_type == "radio":
                val = st.radio(**params, key=f"radio_{feat_name}")
            st.session_state.user_input[feat_name] = val
            st.write("")
    
    # 第四步：生成预测结果
    st.divider()
    if st.button("🔍 第三步：生成筛查结果", type="primary", use_container_width=True):
        # 检查输入完整性
        missing_inputs = [feat for feat in feature_cols if feat not in st.session_state.user_input]
        if missing_inputs:
            st.warning(f"⚠️ 缺少以下问题的输入：{', '.join(missing_inputs)}，请返回补充")
            st.stop()
        
        # 处理用户输入
        try:
            user_data = pd.DataFrame(index=[0], columns=feature_cols)
            for feat_name in feature_cols:
                val = st.session_state.user_input[feat_name]
                # 数值型特征
                if feat_name == "age":
                    user_data[feat_name] = val
                # 二分类特征编码（是→1，否→0；男→1，女→0）
                elif feat_name == "gender":
                    user_data[feat_name] = 1 if val == "男" else 0
                elif feat_name in ["jundice", "austim"]:
                    user_data[feat_name] = 1 if val == "是" else 0
                elif feat_name.startswith("A") and "_Score" in feat_name:
                    user_data[feat_name] = 1 if val == "是" else 0
            
            # 模型预测
            user_data_scaled = scaler.transform(user_data)
            predictions = {}
            for model_name, (model, acc) in models.items():
                if model_name in ["逻辑回归", "支持向量机"]:
                    pred = model.predict(user_data_scaled)[0]
                else:
                    pred = model.predict(user_data)[0]
                pred_label = "有自闭症倾向（需关注）" if pred == 1 else "无明显自闭症倾向"
                predictions[model_name] = (pred_label, acc)
            
            # 展示结果
            st.divider()
            st.subheader("📊 筛查结果汇总")
            result_df = pd.DataFrame({
                "预测模型": list(predictions.keys()),
                "模型准确率": [f"{acc:.2%}" for _, acc in predictions.values()],
                "筛查结果": [label for label, _ in predictions.values()]
            })
            st.table(result_df)
            
            # 综合结论
            positive_count = sum(1 for label, _ in predictions.values() if "有倾向" in label)
            st.divider()
            if positive_count >= 2:
                st.warning("""
                ⚠️ **综合结论：孩子存在自闭症倾向**
                建议：
                1. 尽快带孩子到正规医院的发育行为儿科、儿童精神科就诊
                2. 就诊时可提供该筛查结果作为参考
                3. 早期干预（3-6岁）对改善预后至关重要，不要拖延
                """)
            else:
                st.success("""
                ✅ **综合结论：孩子目前无明显自闭症倾向**
                建议：
                1. 继续观察孩子的社交、语言和行为发展
                2. 若后续出现异常表现（如语言倒退、社交回避），及时就医
                3. 保持良好的家庭互动，多引导孩子参与社交和游戏
                """)
        
        except Exception as e:
            st.error(f"❌ 预测过程出错：{str(e)}")
            st.info("请刷新页面重新尝试，或检查数据集格式是否正确")

# 8. 运行网页
if __name__ == "__main__":
    main()