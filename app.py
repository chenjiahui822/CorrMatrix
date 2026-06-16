"""
CorrMatrix Studio - 多角色统计分析平台（修复版）
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import warnings

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="CorrMatrix Studio - 多角色统计分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 角色数据库配置 ====================

INDUSTRIES = {
    "📚 教育科研": {
        "icon": "📚",
        "roles": {
            "👩‍🏫 中小学教师": {
                "description": "试题质量分析、成绩分析",
                "features": ["试题-总分相关", "难度区分度", "低质量题目标记"],
                "sample_data": "exam",
                "analysis_func": "teacher_analysis"
            },
            "🎓 大学教授/研究生": {
                "description": "论文数据分析、问卷信效度",
                "features": ["Cronbach's α", "KMO检验", "维度相关"],
                "sample_data": "questionnaire",
                "analysis_func": "academic_analysis"
            }
        }
    },
    "🏥 医疗健康": {
        "icon": "🏥",
        "roles": {
            "👨‍⚕️ 临床医生": {
                "description": "检验指标-疾病相关性",
                "features": ["指标-诊断相关", "风险因素排序"],
                "sample_data": "clinical",
                "analysis_func": "clinical_analysis"
            },
            "🔬 医学研究员": {
                "description": "临床试验数据分析",
                "features": ["分组对比", "指标相关性"],
                "sample_data": "clinical",
                "analysis_func": "clinical_analysis"
            }
        }
    },
    "📊 商业/市场/金融": {
        "icon": "📊",
        "roles": {
            "📈 市场研究员": {
                "description": "满意度分析",
                "features": ["满意度相关", "NPS分析"],
                "sample_data": "market",
                "analysis_func": "market_analysis"
            },
            "🛒 电商运营": {
                "description": "用户行为分析",
                "features": ["行为相关性", "转化分析"],
                "sample_data": "ecommerce",
                "analysis_func": "market_analysis"
            },
            "💰 金融分析师": {
                "description": "资产相关性分析",
                "features": ["资产相关矩阵", "风险分散"],
                "sample_data": "finance",
                "analysis_func": "finance_analysis"
            }
        }
    },
    "🏭 工业/质量/农业": {
        "icon": "🏭",
        "roles": {
            "🔧 质量工程师": {
                "description": "参数-质量相关",
                "features": ["关键指标识别", "过程能力"],
                "sample_data": "quality",
                "analysis_func": "quality_analysis"
            },
            "🔬 实验室技术员": {
                "description": "检测方法对比",
                "features": ["方法一致性", "重复性"],
                "sample_data": "quality",
                "analysis_func": "quality_analysis"
            },
            "🌱 农艺师": {
                "description": "环境-产量相关",
                "features": ["产量因素分析"],
                "sample_data": "agriculture",
                "analysis_func": "agriculture_analysis"
            }
        }
    },
    "💻 互联网/数据科学": {
        "icon": "💻",
        "roles": {
            "📉 数据分析师": {
                "description": "通用相关分析",
                "features": ["相关矩阵", "热力图"],
                "sample_data": "general",
                "analysis_func": "general_analysis"
            },
            "🤖 机器学习工程师": {
                "description": "特征-目标相关",
                "features": ["特征重要性", "特征筛选"],
                "sample_data": "ml",
                "analysis_func": "ml_analysis"
            }
        }
    },
    "🧠 心理/社会/人文": {
        "icon": "🧠",
        "roles": {
            "🧠 心理咨询师": {
                "description": "量表信效度",
                "features": ["Cronbach's α", "维度相关"],
                "sample_data": "psychology",
                "analysis_func": "psychology_analysis"
            },
            "👥 人力资源HR": {
                "description": "测评-绩效相关",
                "features": ["预测效度", "高潜识别"],
                "sample_data": "hr",
                "analysis_func": "hr_analysis"
            }
        }
    }
}


# ==================== 示例数据生成 ====================

def generate_sample_data(data_type):
    """生成示例数据"""
    np.random.seed(42)
    n = 100

    if data_type == "exam":
        return pd.DataFrame({
            '学生ID': [f'S{i:03d}' for i in range(1, n + 1)],
            '语文': np.random.normal(75, 12, n),
            '数学': np.random.normal(72, 15, n),
            '英语': np.random.normal(78, 11, n),
            '科学': np.random.normal(70, 13, n),
            '总分': np.random.normal(295, 40, n)
        })

    elif data_type == "clinical":
        return pd.DataFrame({
            '患者ID': [f'P{i:03d}' for i in range(1, n + 1)],
            '年龄': np.random.normal(55, 12, n),
            '血压': np.random.normal(130, 15, n),
            '血糖': np.random.normal(5.6, 1.2, n),
            '胆固醇': np.random.normal(5.2, 1.0, n),
            '诊断结果': np.random.choice([0, 1], n, p=[0.7, 0.3])
        })

    elif data_type == "market":
        return pd.DataFrame({
            '受访者ID': [f'R{i:03d}' for i in range(1, n + 1)],
            '产品满意度': np.random.normal(3.5, 1.0, n),
            '服务满意度': np.random.normal(3.8, 0.9, n),
            '价格满意度': np.random.normal(3.2, 1.1, n),
            '品牌忠诚度': np.random.normal(3.6, 1.0, n),
            '推荐意愿': np.random.normal(3.7, 1.0, n)
        })

    elif data_type == "finance":
        returns = pd.DataFrame({
            '日期': pd.date_range('2023-01-01', periods=n, freq='D'),
            '股票A': np.random.normal(0.001, 0.02, n),
            '股票B': np.random.normal(0.0005, 0.018, n),
            '股票C': np.random.normal(0.0008, 0.022, n),
            '债券基金': np.random.normal(0.0003, 0.005, n)
        })
        return returns

    elif data_type == "quality":
        return pd.DataFrame({
            '批次': [f'B{i:03d}' for i in range(1, n + 1)],
            '温度': np.random.normal(150, 10, n),
            '压力': np.random.normal(5, 0.5, n),
            '速度': np.random.normal(100, 8, n),
            '良品率': np.random.normal(0.97, 0.02, n),
            '缺陷数': np.random.poisson(5, n)
        })

    elif data_type == "psychology":
        return pd.DataFrame({
            '被试ID': [f'P{i:03d}' for i in range(1, n + 1)],
            '焦虑_1': np.random.randint(1, 5, n),
            '焦虑_2': np.random.randint(1, 5, n),
            '焦虑_3': np.random.randint(1, 5, n),
            '抑郁_1': np.random.randint(1, 5, n),
            '抑郁_2': np.random.randint(1, 5, n),
            '抑郁_3': np.random.randint(1, 5, n),
            '生活质量_1': np.random.randint(1, 5, n),
            '生活质量_2': np.random.randint(1, 5, n),
            '生活质量_3': np.random.randint(1, 5, n)
        })

    elif data_type == "hr":
        return pd.DataFrame({
            '员工ID': [f'E{i:03d}' for i in range(1, n + 1)],
            '年龄': np.random.normal(32, 8, n),
            '工龄': np.random.normal(5, 4, n),
            '能力测评': np.random.normal(75, 10, n),
            '潜力测评': np.random.normal(70, 12, n),
            '绩效评分': np.random.normal(80, 10, n)
        })

    elif data_type == "agriculture":
        return pd.DataFrame({
            '地块ID': [f'F{i:03d}' for i in range(1, n + 1)],
            '施肥量': np.random.normal(50, 15, n),
            '降雨量': np.random.normal(120, 30, n),
            '温度': np.random.normal(25, 5, n),
            '产量': np.random.normal(500, 80, n)
        })

    else:
        return pd.DataFrame({
            '变量A': np.random.normal(50, 15, n),
            '变量B': np.random.normal(60, 12, n),
            '变量C': np.random.normal(45, 10, n),
            '变量D': np.random.normal(55, 14, n),
            '变量E': np.random.normal(48, 11, n)
        })


# ==================== 通用分析函数 ====================

def get_numeric_df(df):
    """获取数值型列"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) == 0:
        st.error("没有找到数值型列，请确保数据包含数值")
        return None
    return numeric_df


def calculate_correlation(df):
    """计算相关系数矩阵和p值"""
    numeric_df = get_numeric_df(df)
    if numeric_df is None:
        return None, None

    if len(numeric_df.columns) < 2:
        st.warning("需要至少2个数值变量")
        return None, None

    corr_matrix = numeric_df.corr(method='pearson')

    # 计算p值
    p_matrix = pd.DataFrame(np.ones_like(corr_matrix),
                            index=corr_matrix.index,
                            columns=corr_matrix.columns)
    for i in range(len(numeric_df.columns)):
        for j in range(len(numeric_df.columns)):
            if i != j:
                _, p = stats.pearsonr(numeric_df.iloc[:, i], numeric_df.iloc[:, j])
                p_matrix.iloc[i, j] = p

    return corr_matrix, p_matrix


def create_heatmap(corr_matrix, title="相关系数热力图"):
    """创建热力图"""
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect='auto',
        zmin=-1, zmax=1,
        title=title
    )
    fig.update_layout(height=500, width=700)
    return fig


def get_sig_stars(p):
    """获取显著性标记"""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


# ==================== 角色分析函数 ====================

def teacher_analysis(df):
    """中小学教师分析"""
    st.subheader("📝 试题质量分析报告")

    numeric_df = get_numeric_df(df)
    if numeric_df is None:
        return

    # 识别总分列
    total_cols = [col for col in numeric_df.columns if '总分' in col or '总' in col]
    subject_cols = [col for col in numeric_df.columns if col not in total_cols]

    if total_cols and len(subject_cols) > 0:
        total_col = total_cols[0]
        st.write(f"### 📌 各科目与{total_col}的相关性")

        results = []
        for col in subject_cols:
            corr, p = stats.pearsonr(numeric_df[col].dropna(), numeric_df[total_col].dropna())
            results.append({
                '科目': col,
                '相关系数': corr,
                '显著性': get_sig_stars(p),
                '评价': '优秀' if abs(corr) > 0.6 else '良好' if abs(corr) > 0.4 else '一般'
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

    # 描述统计
    st.write("### 📊 成绩描述统计")
    st.dataframe(numeric_df.describe())

    # 相关性热力图
    corr_matrix, _ = calculate_correlation(df)
    if corr_matrix is not None:
        st.write("### 🔥 成绩相关性热力图")
        fig = create_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)


def academic_analysis(df):
    """学术研究分析"""
    st.subheader("📊 问卷信效度分析")

    numeric_df = get_numeric_df(df)
    if numeric_df is None:
        return

    # Cronbach's α
    def cronbach_alpha(data):
        k = len(data.columns)
        if k < 2:
            return np.nan
        item_variances = data.var(axis=0, ddof=1)
        total_var = data.sum(axis=1).var(ddof=1)
        return (k / (k - 1)) * (1 - item_variances.sum() / total_var)

    alpha = cronbach_alpha(numeric_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cronbach's α", f"{alpha:.3f}")
    with col2:
        quality = "优秀" if alpha > 0.9 else "良好" if alpha > 0.8 else "可接受" if alpha > 0.7 else "需改进"
        st.metric("信度评价", quality)
    with col3:
        st.metric("分析项数", len(numeric_df.columns))

    # 相关性热力图
    corr_matrix, p_matrix = calculate_correlation(df)
    if corr_matrix is not None:
        st.write("### 🔥 变量相关性热力图")
        fig = create_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)

        # 显示相关系数矩阵
        with st.expander("📋 详细相关系数矩阵"):
            corr_display = corr_matrix.round(3).astype(str)
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    if i != j and p_matrix is not None:
                        corr_display.iloc[i, j] += get_sig_stars(p_matrix.iloc[i, j])
            st.dataframe(corr_display)


def clinical_analysis(df):
    """临床分析"""
    st.subheader("🏥 临床指标分析报告")

    numeric_df = get_numeric_df(df)
    if numeric_df is None:
        return

    # 识别诊断列
    diag_cols = [col for col in numeric_df.columns if '诊断' in col or '结果' in col]

    if diag_cols:
        diag_col = diag_cols[0]
        st.write(f"### 🩺 指标与{diag_col}的相关性")

        results = []
        for col in numeric_df.columns:
            if col != diag_col:
                corr, p = stats.pointbiserialr(numeric_df[diag_col], numeric_df[col])
                results.append({
                    '指标': col,
                    '相关系数': corr,
                    '显著性': get_sig_stars(p),
                    '风险等级': '高风险' if abs(corr) > 0.3 else '中风险' if abs(corr) > 0.2 else '低风险'
                })

        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

    # 相关性热力图
    corr_matrix, _ = calculate_correlation(df)
    if corr_matrix is not None:
        st.write("### 🔥 指标相关性热力图")
        fig = create_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)

    # 描述统计
    st.write("### 📊 指标描述统计")
    st.dataframe(numeric_df.describe())


def market_analysis(df):
    """市场分析"""
    st.subheader("📈 市场调研分析报告")

    numeric_df = get_numeric_df(df)
    if numeric_df is None:
        return

    # 相关性热力图
    corr_matrix, _ = calculate_correlation(df)
    if corr_matrix is not None:
        st.write("### 🔥 变量相关性热力图")
        fig = create_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)

        # 强相关性发现
        st.write("### 📌 强相关性发现")
        strong_corr = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    strong_corr.append({
                        '变量1': corr_matrix.index[i],
                        '变量2': corr_matrix.columns[j],
                        '相关系数': corr_matrix.iloc[i, j]
                    })

        if strong_corr:
            st.dataframe(pd.DataFrame(strong_corr))
        else:
            st.info("未发现强相关性 (>0.5)")

    # 描述统计
    st.write("### 📊 描述统计")
    st.dataframe(numeric_df.describe())


def finance_analysis(df):
    """金融分析"""
    st.subheader("💰 金融资产相关性分析")

    numeric_df = get_numeric_df(df)
    if numeric_df is None:
        return

    # 相关性热力图
    corr_matrix, _ = calculate_correlation(df)
    if corr_matrix is not None:
        st.write("### 📊 资产收益率相关性矩阵")
        fig = create_heatmap(corr_matrix, "资产相关性热力图")
        st.plotly_chart(fig, use_container_width=True)

        # 低相关组合（分散风险）
        st.write("### 💡 风险分散建议")
        low_corr = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i, j]) < 0.3:
                    low_corr.append({
                        '资产1': corr_matrix.index[i],
                        '资产2': corr_matrix.columns[j],
                        '相关系数': corr_matrix.iloc[i, j],
                        '分散效果': '优秀' if abs(corr_matrix.iloc[i, j]) < 0.2 else '良好'
                    })

        if low_corr:
            st.success(f"发现 {len(low_corr)} 对低相关性资产组合")
            st.dataframe(pd.DataFrame(low_corr))
        else:
            st.info("所有资产对相关性均大于0.3，分散效果有限")


def quality_analysis(df):
    """质量分析"""
    st.subheader("🔧 生产过程质量分析")

    numeric_df = get_numeric_df(df)
    if numeric_df is None:
        return

    # 识别质量指标
    quality_cols = [col for col in numeric_df.columns if '良品' in col or '缺陷' in col or '率' in col]
    param_cols = [col for col in numeric_df.columns if col not in quality_cols]

    if quality_cols and param_cols:
        st.write("### 📊 工艺参数与质量指标的相关性")

        results = []
        for quality in quality_cols:
            for param in param_cols:
                corr, p = stats.pearsonr(numeric_df[quality].dropna(), numeric_df[param].dropna())
                results.append({
                    '质量指标': quality,
                    '工艺参数': param,
                    '相关系数': corr,
                    '显著性': get_sig_stars(p)
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

    # 过程能力分析
    st.write("### 📈 过程能力分析")
    process_stats = pd.DataFrame({
        '指标': numeric_df.columns,
        '均值': numeric_df.mean().round(2),
        '标准差': numeric_df.std().round(2),
        '最小值': numeric_df.min(),
        '最大值': numeric_df.max()
    })
    st.dataframe(process_stats)

    # 相关性热力图
    corr_matrix, _ = calculate_correlation(df)
    if corr_matrix is not None:
        st.write("### 🔥 参数相关性热力图")
        fig = create_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)


def agriculture_analysis(df):
    """农业分析"""
    st.subheader("🌱 农业产量分析")

    numeric_df = get_numeric_df(df)
    if numeric_df is None:
        return

    # 识别产量列
    yield_cols = [col for col in numeric_df.columns if '产量' in col]

    if yield_cols:
        yield_col = yield_cols[0]
        st.write(f"### 📈 影响{yield_col}的因素分析")

        results = []
        for col in numeric_df.columns:
            if col != yield_col:
                corr, p = stats.pearsonr(numeric_df[yield_col].dropna(), numeric_df[col].dropna())
                results.append({
                    '影响因素': col,
                    '相关系数': corr,
                    '显著性': get_sig_stars(p),
                    '重要性': '高' if abs(corr) > 0.4 else '中' if abs(corr) > 0.25 else '低'
                })

        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 关键因素
        key_factors = results_df[results_df['重要性'] == '高']
        if len(key_factors) > 0:
            st.success(f"🎯 发现 {len(key_factors)} 个关键影响因素")

    # 相关性热力图
    corr_matrix, _ = calculate_correlation(df)
    if corr_matrix is not None:
        st.write("### 🔥 变量相关性热力图")
        fig = create_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)


def general_analysis(df):
    """通用分析"""
    st.subheader("📊 通用相关性分析")

    numeric_df = get_numeric_df(df)
    if numeric_df is None:
        return

    # 相关性热力图
    corr_matrix, p_matrix = calculate_correlation(df)
    if corr_matrix is not None:
        st.write("### 🔥 相关系数热力图")
        fig = create_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)

        # 相关系数矩阵
        with st.expander("📋 详细相关系数矩阵"):
            corr_display = corr_matrix.round(3).astype(str)
            if p_matrix is not None:
                for i in range(len(corr_matrix)):
                    for j in range(len(corr_matrix)):
                        if i != j:
                            corr_display.iloc[i, j] += get_sig_stars(p_matrix.iloc[i, j])
            st.dataframe(corr_display)

        # 最强相关性
        st.write("### 🎯 最强相关性对")
        corr_flat = corr_matrix.unstack()
        corr_flat = corr_flat[corr_flat.index.get_level_values(0) != corr_flat.index.get_level_values(1)]
        top_pairs = corr_flat.abs().nlargest(5)

        for (var1, var2), corr_val in top_pairs.items():
            p_val = p_matrix.loc[var1, var2] if p_matrix is not None else None
            sig = get_sig_stars(p_val) if p_val is not None else ''
            st.write(f"- **{var1} ↔ {var2}**: {corr_val:.3f} {sig}")

    # 描述统计
    with st.expander("📊 描述统计"):
        st.dataframe(numeric_df.describe())


def ml_analysis(df):
    """机器学习特征分析"""
    st.subheader("🤖 特征工程分析")

    numeric_df = get_numeric_df(df)
    if numeric_df is None:
        return

    # 尝试识别目标列
    target_cols = [col for col in numeric_df.columns if '目标' in col or 'target' in col.lower()]

    if target_cols:
        target = target_cols[0]
        st.write(f"### 🎯 特征与目标变量'{target}'的相关性")

        features = [col for col in numeric_df.columns if col != target]

        results = []
        for feat in features:
            corr, p = stats.pearsonr(numeric_df[target].dropna(), numeric_df[feat].dropna())
            results.append({
                '特征': feat,
                '相关系数': corr,
                'p值': p,
                '显著性': get_sig_stars(p),
                '重要性': '高' if abs(corr) > 0.4 else '中' if abs(corr) > 0.25 else '低'
            })

        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 重要特征
        important = results_df[results_df['重要性'] == '高']
        if len(important) > 0:
            st.success(f"✅ 发现 {len(important)} 个重要特征")
    else:
        # 无目标列，显示特征间相关性
        general_analysis(df)


def psychology_analysis(df):
    """心理量表分析"""
    st.subheader("🧠 心理量表信效度分析")

    numeric_df = get_numeric_df(df)
    if numeric_df is None:
        return

    # Cronbach's α
    def cronbach_alpha(data):
        k = len(data.columns)
        if k < 2:
            return np.nan
        item_variances = data.var(axis=0, ddof=1)
        total_var = data.sum(axis=1).var(ddof=1)
        return (k / (k - 1)) * (1 - item_variances.sum() / total_var)

    alpha = cronbach_alpha(numeric_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cronbach's α", f"{alpha:.3f}")
    with col2:
        quality = "优秀" if alpha > 0.9 else "良好" if alpha > 0.8 else "可接受" if alpha > 0.7 else "需改进"
        st.metric("信度评价", quality)
    with col3:
        st.metric("分析项数", len(numeric_df.columns))

    # 相关性热力图
    corr_matrix, _ = calculate_correlation(df)
    if corr_matrix is not None:
        st.write("### 🔥 量表条目相关性热力图")
        fig = create_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)

    # 维度分组（按前缀）
    st.write("### 📊 描述统计")
    st.dataframe(numeric_df.describe())


def hr_analysis(df):
    """人力资源分析"""
    st.subheader("👥 人才测评分析")

    numeric_df = get_numeric_df(df)
    if numeric_df is None:
        return

    # 识别绩效列
    perf_cols = [col for col in numeric_df.columns if '绩效' in col or '业绩' in col]
    assess_cols = [col for col in numeric_df.columns if '测评' in col or '能力' in col or '潜力' in col]

    if perf_cols and assess_cols:
        st.write("### 🎯 测评维度与绩效的相关性")

        results = []
        for perf in perf_cols:
            for assess in assess_cols:
                corr, p = stats.pearsonr(numeric_df[perf].dropna(), numeric_df[assess].dropna())
                results.append({
                    '绩效指标': perf,
                    '测评维度': assess,
                    '相关系数': corr,
                    '显著性': get_sig_stars(p)
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 最佳预测指标
        best = results_df.nlargest(5, '相关系数')
        st.success("🏆 最佳预测指标")
        st.dataframe(best)

    # 相关性热力图
    corr_matrix, _ = calculate_correlation(df)
    if corr_matrix is not None:
        st.write("### 🔥 测评指标相关性")
        fig = create_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)


# ==================== 主程序 ====================

def main():
    st.title("🎯 CorrMatrix Studio")
    st.markdown("*多角色统计分析平台 - 选择您的职业，获得专属分析*")

    # 初始化session state
    if 'selected_industry' not in st.session_state:
        st.session_state.selected_industry = list(INDUSTRIES.keys())[0]
    if 'selected_role' not in st.session_state:
        st.session_state.selected_role = list(INDUSTRIES[st.session_state.selected_industry]["roles"].keys())[0]
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # 侧边栏
    with st.sidebar:
        st.markdown("---")

        # 行业选择
        st.subheader("🏢 选择行业领域")
        industries = list(INDUSTRIES.keys())

        for industry in industries:
            if st.button(f"{INDUSTRIES[industry]['icon']} {industry}", key=f"btn_{industry}", use_container_width=True):
                st.session_state.selected_industry = industry
                st.session_state.selected_role = list(INDUSTRIES[industry]["roles"].keys())[0]
                st.session_state.data_loaded = False
                st.rerun()

        st.markdown("---")

        # 角色选择
        st.subheader("👤 选择职业角色")
        roles = INDUSTRIES[st.session_state.selected_industry]["roles"]
        role_options = list(roles.keys())

        selected_role = st.selectbox(
            "职业角色",
            role_options,
            index=role_options.index(
                st.session_state.selected_role) if st.session_state.selected_role in role_options else 0
        )

        if selected_role != st.session_state.selected_role:
            st.session_state.selected_role = selected_role
            st.session_state.data_loaded = False
            st.rerun()

        st.markdown("---")

        # 显示角色描述
        if st.session_state.selected_role:
            role_info = roles[st.session_state.selected_role]
            st.info(f"**{st.session_state.selected_role}**\n\n{role_info['description']}")

            st.markdown("**核心功能：**")
            for feat in role_info['features']:
                st.write(f"• {feat}")

        st.markdown("---")

        # 数据上传
        st.subheader("📁 数据上传")
        uploaded_file = st.file_uploader(
            "上传CSV或Excel文件",
            type=['csv', 'xlsx', 'xls']
        )

        # 示例数据
        use_example = st.checkbox("使用示例数据", value=not uploaded_file)

        if use_example and st.session_state.selected_role:
            if st.button("📊 生成示例数据", use_container_width=True):
                sample_type = roles[st.session_state.selected_role]["sample_data"]
                st.session_state.df = generate_sample_data(sample_type)
                st.session_state.data_loaded = True
                st.success(f"✅ 已生成{st.session_state.selected_role}示例数据")
                st.rerun()

        elif uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.session_state.data_loaded = True
                st.success("✅ 数据上传成功")
                st.rerun()
            except Exception as e:
                st.error(f"数据加载失败: {e}")

    # 主内容区域
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df

        # 数据概览
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总行数", len(df))
        with col2:
            st.metric("总列数", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("数值列数", len(numeric_cols))
        with col4:
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            st.metric("缺失率", f"{missing_pct:.1f}%")

        # 数据预览
        with st.expander("📋 数据预览"):
            st.dataframe(df.head(10))

        st.markdown("---")

        # 调用角色专属分析
        role_key = st.session_state.selected_role
        if role_key:
            role_info = INDUSTRIES[st.session_state.selected_industry]["roles"][role_key]
            analysis_func_name = role_info["analysis_func"]

            # 动态调用分析函数
            analysis_func = globals().get(analysis_func_name)
            if analysis_func:
                analysis_func(df)
            else:
                st.warning(f"分析功能开发中")
                general_analysis(df)

    else:
        # 欢迎界面
        st.info("👈 请从左侧选择行业和职业角色，然后上传数据或生成示例数据")

        st.markdown("""
        ### 🎯 快速开始

        1. **选择行业领域**（教育/医疗/商业/工业/互联网/心理）
        2. **选择职业角色**（16个角色可选）
        3. **生成示例数据** 或 **上传自己的数据**
        4. **查看专属分析报告**

        ### 📊 支持的角色类型

        | 行业 | 角色 |
        |------|------|
        | 📚 教育科研 | 中小学教师、大学教授/研究生 |
        | 🏥 医疗健康 | 临床医生、医学研究员 |
        | 📊 商业/市场/金融 | 市场研究员、电商运营、金融分析师 |
        | 🏭 工业/质量/农业 | 质量工程师、实验室技术员、农艺师 |
        | 💻 互联网/数据科学 | 数据分析师、机器学习工程师 |
        | 🧠 心理/社会/人文 | 心理咨询师、人力资源HR |
        """)

    # 页脚
    st.markdown("---")
    st.markdown(
        "<center>CorrMatrix Studio - 为每个职业定制的统计分析工具</center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()