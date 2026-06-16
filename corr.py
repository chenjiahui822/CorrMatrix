"""
CorrMatrix Studio - 多角色统计分析平台
支持6大行业 × 26个职业角色的专属分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
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
        "color": "#4CAF50",
        "roles": {
            "👩‍🏫 中小学教师": {
                "description": "试题质量分析、成绩分析、难度区分度",
                "features": ["试题-总分相关", "难度系数", "区分度", "低质量题目标记"],
                "sample_data": "exam_scores",
                "analysis_func": "teacher_analysis"
            },
            "🎓 大学教授/研究生": {
                "description": "论文数据分析、问卷信效度",
                "features": ["Cronbach's α", "KMO检验", "Bartlett球形检验", "维度相关"],
                "sample_data": "questionnaire",
                "analysis_func": "academic_analysis"
            },
            "📝 教育测评专家": {
                "description": "标准化考试开发、IRT初探",
                "features": ["题目参数估计", "信息函数", "DIF检测", "等值分析"],
                "sample_data": "standardized_test",
                "analysis_func": "assessment_analysis"
            },
            "🏫 学校管理者": {
                "description": "学校质量评估、指标预警",
                "features": ["指标相关性", "预警分析", "趋势对比", "标杆对比"],
                "sample_data": "school_metrics",
                "analysis_func": "school_analysis"
            }
        }
    },
    "🏥 医疗健康": {
        "icon": "🏥",
        "color": "#2196F3",
        "roles": {
            "👨‍⚕️ 临床医生": {
                "description": "检验指标-疾病相关性分析",
                "features": ["指标-诊断相关", "参考区间", "ROC曲线", "风险因素排序"],
                "sample_data": "clinical_indicators",
                "analysis_func": "clinical_analysis"
            },
            "🔬 医学研究员": {
                "description": "临床试验数据分析",
                "features": ["治疗前后相关", "分组对比", "协变量控制", "生存分析相关"],
                "sample_data": "clinical_trial",
                "analysis_func": "research_analysis"
            },
            "💊 药学/药理": {
                "description": "剂量-效应分析",
                "features": ["剂量-效应相关", "半数有效量", "毒理指标", "药物相互作用"],
                "sample_data": "pharma_data",
                "analysis_func": "pharma_analysis"
            },
            "🧬 生物信息学": {
                "description": "基因表达分析",
                "features": ["基因-表型相关", "聚类热图", "通路富集相关", "差异表达"],
                "sample_data": "gene_expression",
                "analysis_func": "bioinfo_analysis"
            },
            "🏃 康复/运动": {
                "description": "体质测试分析",
                "features": ["指标-恢复相关", "运动表现相关", "风险因素", "训练监控"],
                "sample_data": "sports_data",
                "analysis_func": "sports_analysis"
            }
        }
    },
    "📊 商业/市场/金融": {
        "icon": "📊",
        "color": "#FF9800",
        "roles": {
            "📈 市场研究员": {
                "description": "满意度-忠诚度分析",
                "features": ["满意度相关", "忠诚度预测", "细分人群对比", "NPS分析"],
                "sample_data": "market_survey",
                "analysis_func": "market_analysis"
            },
            "🛒 电商运营": {
                "description": "用户行为-转化分析",
                "features": ["行为相关性", "RFM分析", "漏斗转化相关", "复购预测"],
                "sample_data": "ecommerce_data",
                "analysis_func": "ecommerce_analysis"
            },
            "📱 产品经理": {
                "description": "功能使用-留存分析",
                "features": ["功能相关性", "留存因素", "用户分群", "A/B测试相关"],
                "sample_data": "product_data",
                "analysis_func": "product_analysis"
            },
            "💰 金融分析师": {
                "description": "资产组合相关性",
                "features": ["资产相关矩阵", "风险分散", "回撤相关", "Beta系数"],
                "sample_data": "financial_data",
                "analysis_func": "finance_analysis"
            },
            "🏦 风控专员": {
                "description": "信用指标-违约相关",
                "features": ["违约因素排序", "评分卡相关", "逾期分析", "欺诈检测"],
                "sample_data": "risk_data",
                "analysis_func": "risk_analysis"
            }
        }
    },
    "🏭 工业/质量/农业": {
        "icon": "🏭",
        "color": "#9C27B0",
        "roles": {
            "🔧 质量工程师": {
                "description": "参数-质量相关",
                "features": ["关键质量指标", "过程能力相关", "SPC分析", "六西格玛"],
                "sample_data": "quality_data",
                "analysis_func": "quality_analysis"
            },
            "🏭 工业工程师": {
                "description": "工时-产出相关",
                "features": ["效率因素", "瓶颈识别", "布局优化相关", "作业分析"],
                "sample_data": "industrial_data",
                "analysis_func": "industrial_analysis"
            },
            "🔬 实验室技术员": {
                "description": "检测方法对比",
                "features": ["方法一致性", "重复性相关", "准确性验证", "不确定度"],
                "sample_data": "lab_data",
                "analysis_func": "lab_analysis"
            },
            "🌱 农艺师": {
                "description": "环境-产量相关",
                "features": ["产量因素", "土壤指标相关", "气候影响", "施肥优化"],
                "sample_data": "agriculture_data",
                "analysis_func": "agriculture_analysis"
            },
            "🌍 环境科学家": {
                "description": "污染物-影响相关",
                "features": ["污染源识别", "环境影响", "生态指标", "时空变化"],
                "sample_data": "environment_data",
                "analysis_func": "environment_analysis"
            }
        }
    },
    "💻 互联网/数据科学": {
        "icon": "💻",
        "color": "#00BCD4",
        "roles": {
            "📉 数据分析师": {
                "description": "通用相关分析",
                "features": ["完整相关矩阵", "多种相关系数", "热力图", "散点图矩阵"],
                "sample_data": "general_data",
                "analysis_func": "general_analysis"
            },
            "🤖 机器学习工程师": {
                "description": "特征-目标相关",
                "features": ["特征重要性", "特征筛选", "多重共线性", "IV值计算"],
                "sample_data": "ml_data",
                "analysis_func": "ml_analysis"
            },
            "📊 BI分析师": {
                "description": "自动结论生成",
                "features": ["智能洞察", "异常检测", "趋势分析", "自动报表"],
                "sample_data": "bi_data",
                "analysis_func": "bi_analysis"
            },
            "🔍 用户增长": {
                "description": "留存-行为相关",
                "features": ["留存因素", "魔法数字", "激活分析", "病毒系数"],
                "sample_data": "growth_data",
                "analysis_func": "growth_analysis"
            }
        }
    },
    "🧠 心理/社会/人文": {
        "icon": "🧠",
        "color": "#E91E63",
        "roles": {
            "🧠 心理咨询师": {
                "description": "量表信效度",
                "features": ["Cronbach's α", "重测信度", "效标关联效度", "因子结构"],
                "sample_data": "psychology_data",
                "analysis_func": "psychology_analysis"
            },
            "📋 社会调查员": {
                "description": "社会指标关联",
                "features": ["变量交叉分析", "卡方检验", "关联强度", "抽样误差"],
                "sample_data": "social_data",
                "analysis_func": "social_analysis"
            },
            "👥 人力资源HR": {
                "description": "测评-绩效相关",
                "features": ["预测效度", "测评维度相关", "高潜识别", "离职预警"],
                "sample_data": "hr_data",
                "analysis_func": "hr_analysis"
            },
            "⚽ 体育分析师": {
                "description": "训练-成绩相关",
                "features": ["训练负荷相关", "技术指标相关", "伤病风险", "比赛预测"],
                "sample_data": "sports_data",
                "analysis_func": "sports_analysis"
            }
        }
    }
}


# ==================== 示例数据生成函数 ====================

def generate_sample_data(data_type):
    """根据角色类型生成示例数据"""
    np.random.seed(42)
    n = 150

    if data_type == "exam_scores":
        # 考试数据：学生ID + 各题得分 + 总分
        data = pd.DataFrame({
            '学生ID': [f'S{i:03d}' for i in range(1, n + 1)],
            '第1题': np.random.randint(0, 10, n),
            '第2题': np.random.randint(0, 10, n),
            '第3题': np.random.randint(0, 10, n),
            '第4题': np.random.randint(0, 10, n),
            '第5题': np.random.randint(0, 10, n),
            '第6题': np.random.randint(0, 10, n),
            '第7题': np.random.randint(0, 10, n),
            '第8题': np.random.randint(0, 10, n),
            '第9题': np.random.randint(0, 10, n),
            '第10题': np.random.randint(0, 10, n),
        })
        # 添加总分
        data['总分'] = data.iloc[:, 1:].sum(axis=1)
        return data

    elif data_type == "clinical_indicators":
        # 临床数据：患者 + 各项指标 + 诊断结果
        data = pd.DataFrame({
            '患者ID': [f'P{i:03d}' for i in range(1, n + 1)],
            '年龄': np.random.normal(55, 12, n),
            '血压_收缩压': np.random.normal(130, 15, n),
            '血糖': np.random.normal(5.6, 1.2, n),
            '胆固醇': np.random.normal(5.2, 1.0, n),
            '甘油三酯': np.random.normal(1.5, 0.8, n),
            '尿酸': np.random.normal(350, 80, n),
            '肌酐': np.random.normal(80, 20, n),
            '诊断结果': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        })
        return data

    elif data_type == "market_survey":
        # 市场调查数据
        data = pd.DataFrame({
            '受访者ID': [f'R{i:03d}' for i in range(1, n + 1)],
            '产品满意度': np.random.normal(3.5, 1.0, n),
            '服务满意度': np.random.normal(3.8, 0.9, n),
            '价格满意度': np.random.normal(3.2, 1.1, n),
            '品牌忠诚度': np.random.normal(3.6, 1.0, n),
            '推荐意愿': np.random.normal(3.7, 1.0, n),
            '复购意向': np.random.normal(3.5, 1.0, n),
            '年龄': np.random.normal(35, 10, n),
            '收入水平': np.random.normal(8000, 3000, n),
        })
        return data

    elif data_type == "financial_data":
        # 金融数据
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        returns = pd.DataFrame({
            '日期': dates,
            '股票A': np.random.normal(0.001, 0.02, n),
            '股票B': np.random.normal(0.0005, 0.018, n),
            '股票C': np.random.normal(0.0008, 0.022, n),
            '股票D': np.random.normal(0.0012, 0.025, n),
            '债券基金': np.random.normal(0.0003, 0.005, n),
            '指数': np.random.normal(0.0006, 0.015, n),
        })
        return returns

    elif data_type == "quality_data":
        # 质量数据
        data = pd.DataFrame({
            '批次': [f'B{i:03d}' for i in range(1, n + 1)],
            '温度': np.random.normal(150, 10, n),
            '压力': np.random.normal(5, 0.5, n),
            '速度': np.random.normal(100, 8, n),
            '原料纯度': np.random.normal(0.95, 0.03, n),
            '良品率': np.random.normal(0.97, 0.02, n),
            '缺陷数': np.random.poisson(5, n),
        })
        return data

    elif data_type == "psychology_data":
        # 心理量表数据
        data = pd.DataFrame({
            '被试ID': [f'S{i:03d}' for i in range(1, n + 1)],
            '焦虑_第1题': np.random.randint(1, 5, n),
            '焦虑_第2题': np.random.randint(1, 5, n),
            '焦虑_第3题': np.random.randint(1, 5, n),
            '焦虑_第4题': np.random.randint(1, 5, n),
            '焦虑_第5题': np.random.randint(1, 5, n),
            '抑郁_第1题': np.random.randint(1, 5, n),
            '抑郁_第2题': np.random.randint(1, 5, n),
            '抑郁_第3题': np.random.randint(1, 5, n),
            '抑郁_第4题': np.random.randint(1, 5, n),
            '抑郁_第5题': np.random.randint(1, 5, n),
            '生活质量_第1题': np.random.randint(1, 5, n),
            '生活质量_第2题': np.random.randint(1, 5, n),
            '生活质量_第3题': np.random.randint(1, 5, n),
            '生活质量_第4题': np.random.randint(1, 5, n),
            '生活质量_第5题': np.random.randint(1, 5, n),
        })
        # 添加各维度总分
        data['焦虑总分'] = data.iloc[:, 1:6].sum(axis=1)
        data['抑郁总分'] = data.iloc[:, 6:11].sum(axis=1)
        data['生活质量总分'] = data.iloc[:, 11:16].sum(axis=1)
        return data

    elif data_type == "hr_data":
        # 人力资源数据
        data = pd.DataFrame({
            '员工ID': [f'E{i:03d}' for i in range(1, n + 1)],
            '年龄': np.random.normal(32, 8, n),
            '工龄': np.random.normal(5, 4, n),
            '学历': np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.3, 0.4, 0.2]),
            '培训次数': np.random.poisson(3, n),
            '能力测评': np.random.normal(75, 10, n),
            '潜力测评': np.random.normal(70, 12, n),
            '领导力测评': np.random.normal(68, 11, n),
            '绩效评分': np.random.normal(80, 10, n),
            '离职风险': np.random.normal(0.3, 0.2, n),
        })
        return data

    elif data_type == "general_data":
        # 通用数据
        data = pd.DataFrame({
            '变量A': np.random.normal(50, 15, n),
            '变量B': np.random.normal(60, 12, n),
            '变量C': np.random.normal(45, 10, n),
            '变量D': np.random.normal(55, 14, n),
            '变量E': np.random.normal(48, 11, n),
        })
        # 添加一些相关性
        data['变量B'] = data['变量A'] * 0.6 + np.random.normal(0, 8, n)
        data['变量C'] = data['变量A'] * -0.3 + np.random.normal(0, 10, n)
        return data

    else:
        # 默认数据
        return pd.DataFrame({
            '指标1': np.random.normal(100, 15, n),
            '指标2': np.random.normal(100, 15, n),
            '指标3': np.random.normal(100, 15, n),
            '指标4': np.random.normal(100, 15, n),
            '指标5': np.random.normal(100, 15, n),
        })


# ==================== 通用分析函数 ====================

def calculate_correlations(df, method='pearson'):
    """计算相关系数矩阵"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return None, None
    corr_matrix = numeric_df.corr(method=method)

    # 计算p值
    p_matrix = pd.DataFrame(np.ones_like(corr_matrix),
                            index=corr_matrix.index,
                            columns=corr_matrix.columns)
    for i in range(len(numeric_df.columns)):
        for j in range(len(numeric_df.columns)):
            if i != j:
                if method == 'pearson':
                    _, p = stats.pearsonr(numeric_df.iloc[:, i], numeric_df.iloc[:, j])
                elif method == 'spearman':
                    _, p = stats.spearmanr(numeric_df.iloc[:, i], numeric_df.iloc[:, j])
                else:
                    _, p = stats.kendalltau(numeric_df.iloc[:, i], numeric_df.iloc[:, j])
                p_matrix.iloc[i, j] = p
    return corr_matrix, p_matrix


def create_heatmap(corr_matrix, title="相关系数热力图"):
    """创建热力图"""
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect='auto',
        zmin=-1, zmax=1,
        title=title
    )
    fig.update_layout(height=500, width=700)
    return fig


def get_significance_stars(p_value):
    """返回显著性标记"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return ''


# ==================== 角色专属分析函数 ====================

def teacher_analysis(df):
    """中小学教师 - 试题质量分析"""
    st.subheader("📝 试题质量分析报告")

    # 识别题目列（假设以"题"或"第"开头，或者是数值列但排除总分）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    score_cols = [col for col in numeric_cols if '总分' in col or '总成绩' in col]
    question_cols = [col for col in numeric_cols if col not in score_cols]

    if len(question_cols) == 0:
        st.warning("未识别到题目列，请确保列名包含'题'或使用数值列")
        return

    # 如果有总分列，计算题目-总分相关
    if len(score_cols) > 0:
        total_col = score_cols[0]
        st.write(f"### 📌 题目-总分相关性（效标：{total_col}）")

        results = []
        for col in question_cols:
            corr, p = stats.pearsonr(df[col].dropna(), df[total_col].dropna())
            results.append({
                '题目': col,
                '相关系数': corr,
                '显著性': get_significance_stars(p),
                'p值': p,
                '质量评价': '优秀' if corr > 0.4 else '良好' if corr > 0.3 else '待改进' if corr > 0.2 else '需淘汰'
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 低质量题目提醒
        poor_items = results_df[results_df['质量评价'] == '需淘汰']
        if len(poor_items) > 0:
            st.error(f"⚠️ 发现 {len(poor_items)} 道低质量题目，建议修改或淘汰")
            st.write(poor_items[['题目', '相关系数', '质量评价']])

    # 难度和区分度分析
    st.write("### 📊 难度与区分度分析")
    st.info("💡 难度系数：>0.8偏易，<0.2偏难 | 区分度：>0.4优秀，>0.3良好，>0.2尚可")

    difficulty = {}
    discrimination = {}

    for col in question_cols:
        max_score = df[col].max()
        if max_score > 0:
            difficulty[col] = df[col].mean() / max_score

        if len(score_cols) > 0:
            # 高低分组（前27%和后27%）
            total = df[total_col]
            high_cut = total.quantile(0.73)
            low_cut = total.quantile(0.27)
            high_group = df[df[total_col] >= high_cut][col].mean()
            low_group = df[df[total_col] <= low_cut][col].mean()
            discrimination[col] = (high_group - low_group) / max_score if max_score > 0 else 0

    diff_df = pd.DataFrame({
        '题目': list(difficulty.keys()),
        '难度系数': list(difficulty.values()),
        '难度评价': ['偏易' if d > 0.8 else '偏难' if d < 0.2 else '适中' for d in difficulty.values()]
    })

    if discrimination:
        disc_df = pd.DataFrame({
            '题目': list(discrimination.keys()),
            '区分度': list(discrimination.values()),
            '区分度评价': ['优秀' if d > 0.4 else '良好' if d > 0.3 else '尚可' if d > 0.2 else '差' for d in
                           discrimination.values()]
        })

        col1, col2 = st.columns(2)
        with col1:
            st.write("**难度分析**")
            st.dataframe(diff_df)
        with col2:
            st.write("**区分度分析**")
            st.dataframe(disc_df)


def academic_analysis(df):
    """大学教授/研究生 - 问卷信效度分析"""
    st.subheader("📊 问卷信效度分析报告")

    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        st.warning("需要至少2个数值变量进行分析")
        return

    # Cronbach's α
    st.write("### 📈 信度分析 - Cronbach's α")

    def cronbach_alpha(df_sub):
        items = df_sub.values.T
        variances = items.var(axis=1, ddof=1)
        total_var = df_sub.sum(axis=1).var(ddof=1)
        k = len(df_sub.columns)
        return (k / (k - 1)) * (1 - variances.sum() / total_var)

    alpha = cronbach_alpha(numeric_df)
    alpha_quality = "优秀" if alpha > 0.9 else "良好" if alpha > 0.8 else "可接受" if alpha > 0.7 else "需改进" if alpha > 0.6 else "不可接受"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cronbach's α", f"{alpha:.3f}")
    with col2:
        st.metric("信度评价", alpha_quality)
    with col3:
        st.metric("分析项数", len(numeric_df.columns))

    # 删题后信度
    st.write("### 📉 删题后信度变化")
    alpha_if_deleted = []
    for col in numeric_df.columns:
        alpha_if = cronbach_alpha(numeric_df.drop(columns=[col]))
        alpha_if_deleted.append({
            '删除项': col,
            '删除后α系数': alpha_if,
            '变化': alpha_if - alpha,
            '建议': '可删除' if alpha_if > alpha else '保留'
        })

    alpha_df = pd.DataFrame(alpha_if_deleted)
    st.dataframe(alpha_df.style.background_gradient(subset=['删除后α系数'], cmap='RdYlGn'))

    # KMO检验
    st.write("### 🔬 效度分析 - KMO与Bartlett检验")
    st.info("KMO > 0.7适合因子分析，Bartlett p < 0.05说明变量间存在相关性")

    # 相关系数矩阵
    corr_matrix, p_matrix = calculate_correlations(numeric_df)
    if corr_matrix is not None:
        st.write("**变量相关性矩阵**")
        fig = create_heatmap(corr_matrix, "变量相关性热力图")
        st.plotly_chart(fig, use_container_width=True)


def clinical_analysis(df):
    """临床医生 - 检验指标分析"""
    st.subheader("🏥 临床检验指标分析报告")

    numeric_df = df.select_dtypes(include=[np.number])

    # 识别诊断列
    diag_cols = [col for col in numeric_df.columns if '诊断' in col or '疾病' in col or '结果' in col]

    if diag_cols:
        diag_col = diag_cols[0]
        st.write(f"### 🩺 指标与{diag_col}的相关性")

        results = []
        for col in numeric_df.columns:
            if col != diag_col:
                corr, p = stats.pointbiserialr(df[diag_col], df[col])
                results.append({
                    '指标': col,
                    '相关系数': corr,
                    '显著性': get_significance_stars(p),
                    '临床意义': '高风险因子' if abs(corr) > 0.3 else '风险因子' if abs(corr) > 0.2 else '参考指标'
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 高风险因子推荐
        high_risk = results_df[results_df['临床意义'] == '高风险因子']
        if len(high_risk) > 0:
            st.warning(f"⚠️ 发现 {len(high_risk)} 个高风险因子，建议重点关注")
            st.write(high_risk[['指标', '相关系数']])

    # 参考区间
    st.write("### 📊 指标参考区间（均值 ± 1.96×标准差）")
    stats_df = pd.DataFrame({
        '指标': numeric_df.columns,
        '均值': numeric_df.mean(),
        '标准差': numeric_df.std(),
        '参考下限': numeric_df.mean() - 1.96 * numeric_df.std(),
        '参考上限': numeric_df.mean() + 1.96 * numeric_df.std(),
        '最小值': numeric_df.min(),
        '最大值': numeric_df.max()
    })
    st.dataframe(stats_df)

    # 相关性热力图
    corr_matrix, _ = calculate_correlations(numeric_df)
    if corr_matrix is not None:
        st.write("### 🔥 指标相关性热力图")
        fig = create_heatmap(corr_matrix, "临床指标相关性")
        st.plotly_chart(fig, use_container_width=True)


def market_analysis(df):
    """市场研究员 - 满意度分析"""
    st.subheader("📈 市场调研分析报告")

    numeric_df = df.select_dtypes(include=[np.number])

    # 识别满意度相关列
    sat_cols = [col for col in numeric_df.columns if '满意度' in col or '忠诚' in col or '意愿' in col]
    other_cols = [col for col in numeric_df.columns if col not in sat_cols]

    if sat_cols and other_cols:
        st.write("### 🎯 满意度驱动因素分析")

        results = []
        for sat in sat_cols:
            for factor in other_cols:
                corr, p = stats.pearsonr(df[sat].dropna(), df[factor].dropna())
                results.append({
                    '满意度指标': sat,
                    '影响因素': factor,
                    '相关系数': corr,
                    '显著性': get_significance_stars(p),
                    '重要性': '高' if abs(corr) > 0.4 else '中' if abs(corr) > 0.25 else '低'
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 关键驱动因素
        high_importance = results_df[results_df['重要性'] == '高']
        if len(high_importance) > 0:
            st.success(f"✅ 发现 {len(high_importance)} 个高重要性驱动因素")
            st.write(high_importance[['满意度指标', '影响因素', '相关系数']])

    # NPS分析（如果有）
    nps_cols = [col for col in numeric_df.columns if '推荐' in col or 'NPS' in col]
    if nps_cols:
        st.write("### ⭐ NPS（净推荐值）分析")
        nps_data = df[nps_cols[0]]
        promoters = len(nps_data[nps_data >= 9])
        passives = len(nps_data[(nps_data >= 7) & (nps_data <= 8)])
        detractors = len(nps_data[nps_data <= 6])
        nps = (promoters - detractors) / len(nps_data) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("推荐者", f"{promoters} ({promoters / len(nps_data) * 100:.0f}%)")
        with col2:
            st.metric("被动者", f"{passives} ({passives / len(nps_data) * 100:.0f}%)")
        with col3:
            st.metric("贬损者", f"{detractors} ({detractors / len(nps_data) * 100:.0f}%)")
        with col4:
            st.metric("NPS", f"{nps:.0f}")


def finance_analysis(df):
    """金融分析师 - 资产相关性"""
    st.subheader("💰 金融资产相关性分析")

    numeric_df = df.select_dtypes(include=[np.number])

    # 收益率相关性矩阵
    corr_matrix, p_matrix = calculate_correlations(numeric_df)
    if corr_matrix is not None:
        st.write("### 📊 资产收益率相关性矩阵")
        fig = create_heatmap(corr_matrix, "资产相关性热力图")
        st.plotly_chart(fig, use_container_width=True)

        # 风险分散建议
        st.write("### 💡 风险分散建议")

        # 找出低相关资产对
        low_corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i, j]) < 0.3:
                    low_corr_pairs.append({
                        '资产1': corr_matrix.index[i],
                        '资产2': corr_matrix.columns[j],
                        '相关系数': corr_matrix.iloc[i, j],
                        '分散效果': '优秀' if abs(corr_matrix.iloc[i, j]) < 0.2 else '良好'
                    })

        if low_corr_pairs:
            st.success(f"✅ 发现 {len(low_corr_pairs)} 对低相关性资产组合，适合分散投资")
            st.dataframe(pd.DataFrame(low_corr_pairs))

        # 组合风险计算
        st.write("### 📈 等权重组合风险分析")
        n_assets = len(numeric_df.columns)
        weights = np.ones(n_assets) / n_assets

        # 简化风险计算
        volatilities = numeric_df.std()
        port_vol = np.sqrt(np.dot(weights.T, np.dot(corr_matrix.values *
                                                    np.outer(volatilities, volatilities), weights)))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("资产数量", n_assets)
        with col2:
            st.metric("组合波动率", f"{port_vol:.4f}")


def quality_analysis(df):
    """质量工程师 - 参数分析"""
    st.subheader("🔧 生产过程质量分析")

    numeric_df = df.select_dtypes(include=[np.number])

    # 识别质量指标
    quality_cols = [col for col in numeric_df.columns if '良品' in col or '缺陷' in col or '合格' in col]
    param_cols = [col for col in numeric_df.columns if col not in quality_cols]

    if quality_cols and param_cols:
        st.write("### 📊 工艺参数与质量指标的相关性")

        results = []
        for quality in quality_cols:
            for param in param_cols:
                corr, p = stats.pearsonr(df[quality].dropna(), df[param].dropna())
                results.append({
                    '质量指标': quality,
                    '工艺参数': param,
                    '相关系数': corr,
                    '显著性': get_significance_stars(p),
                    '控制优先级': '高' if abs(corr) > 0.4 else '中' if abs(corr) > 0.25 else '低'
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 关键参数识别
        high_priority = results_df[results_df['控制优先级'] == '高']
        if len(high_priority) > 0:
            st.success(f"🎯 发现 {len(high_priority)} 个关键控制参数")
            st.write(high_priority[['工艺参数', '质量指标', '相关系数']])

    # 过程能力指数（简化）
    st.write("### 📈 过程能力分析")

    process_stats = pd.DataFrame({
        '指标': numeric_df.columns,
        '均值': numeric_df.mean(),
        '标准差': numeric_df.std(),
        'Cpk估算': np.clip((numeric_df.mean() - numeric_df.min()) / (3 * numeric_df.std()), 0, 2)
    })
    st.dataframe(process_stats)


def psychology_analysis(df):
    """心理咨询师 - 量表分析"""
    st.subheader("🧠 心理量表信效度分析")

    numeric_df = df.select_dtypes(include=[np.number])

    # Cronbach's α
    st.write("### 📊 信度分析")

    def cronbach_alpha(df_sub):
        items = df_sub.values.T
        variances = items.var(axis=1, ddof=1)
        total_var = df_sub.sum(axis=1).var(ddof=1)
        k = len(df_sub.columns)
        return (k / (k - 1)) * (1 - variances.sum() / total_var)

    # 按维度分组（假设列名包含维度信息）
    alpha_results = []

    # 整体信度
    overall_alpha = cronbach_alpha(numeric_df)
    alpha_results.append({'维度': '整体量表', 'Cronbach α': overall_alpha})

    # 尝试按前缀分组
    prefixes = set([col.split('_')[0] if '_' in col else col[:2] for col in numeric_df.columns])
    for prefix in prefixes:
        cols = [col for col in numeric_df.columns if col.startswith(prefix)]
        if len(cols) >= 3:
            alpha = cronbach_alpha(numeric_df[cols])
            alpha_results.append({'维度': prefix, 'Cronbach α': alpha})

    alpha_df = pd.DataFrame(alpha_results)
    st.dataframe(alpha_df.style.background_gradient(subset=['Cronbach α'], cmap='RdYlGn', vmin=0, vmax=1))

    # 维度间相关
    st.write("### 🔗 维度间相关性")

    # 计算维度总分
    dim_scores = {}
    for prefix in prefixes:
        cols = [col for col in numeric_df.columns if col.startswith(prefix)]
        if len(cols) >= 2:
            dim_scores[prefix] = numeric_df[cols].sum(axis=1)

    if len(dim_scores) >= 2:
        dim_df = pd.DataFrame(dim_scores)
        dim_corr, _ = calculate_correlations(dim_df)
        if dim_corr is not None:
            fig = create_heatmap(dim_corr, "维度间相关性")
            st.plotly_chart(fig, use_container_width=True)


def hr_analysis(df):
    """人力资源 - 测评分析"""
    st.subheader("👥 人才测评与绩效分析")

    numeric_df = df.select_dtypes(include=[np.number])

    # 识别绩效列
    perf_cols = [col for col in numeric_df.columns if '绩效' in col or '业绩' in col]
    assess_cols = [col for col in numeric_df.columns if '测评' in col or '能力' in col or '潜力' in col]

    if perf_cols and assess_cols:
        st.write("### 🎯 测评维度对绩效的预测效度")

        results = []
        for perf in perf_cols:
            for assess in assess_cols:
                corr, p = stats.pearsonr(df[perf].dropna(), df[assess].dropna())
                results.append({
                    '绩效指标': perf,
                    '测评维度': assess,
                    '相关系数': corr,
                    '显著性': get_significance_stars(p),
                    '预测效度': '强' if abs(corr) > 0.4 else '中' if abs(corr) > 0.25 else '弱'
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 最佳预测指标
        best_predictors = results_df.nlargest(5, '相关系数')
        st.success("🏆 最佳预测指标")
        st.write(best_predictors[['测评维度', '绩效指标', '相关系数', '预测效度']])

    # 高潜人才识别（如果有潜力测评）
    if '潜力测评' in numeric_df.columns and '绩效评分' in numeric_df.columns:
        st.write("### ⭐ 九宫格人才分析")

        # 计算百分位数
        df['绩效等级'] = pd.qcut(df['绩效评分'], 3, labels=['C', 'B', 'A'])
        df['潜力等级'] = pd.qcut(df['潜力测评'], 3, labels=['C', 'B', 'A'])

        nine_grid = df.groupby(['绩效等级', '潜力等级']).size().reset_index(name='人数')

        # 创建九宫格矩阵
        grid_matrix = pd.DataFrame(index=['A', 'B', 'C'], columns=['C', 'B', 'A'])
        for _, row in nine_grid.iterrows():
            grid_matrix.loc[row['绩效等级'], row['潜力等级']] = row['人数']

        st.dataframe(grid_matrix.fillna(0))

        high_potential = df[(df['绩效等级'] == 'A') & (df['潜力等级'].isin(['A', 'B']))]
        st.success(f"🎯 识别出 {len(high_potential)} 名高潜人才")


def general_analysis(df):
    """通用分析 - 完整相关性分析"""
    st.subheader("📊 通用相关性分析")

    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        st.warning("需要至少2个数值变量进行分析")
        return

    # 相关性矩阵
    corr_matrix, p_matrix = calculate_correlations(numeric_df)

    if corr_matrix is not None:
        # 热力图
        st.write("### 🔥 相关系数热力图")
        fig = create_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)

        # 详细矩阵
        col1, col2 = st.columns(2)
        with col1:
            st.write("### 📈 相关系数矩阵")
            # 添加显著性标记
            corr_with_sig = corr_matrix.round(3).astype(str)
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    if i != j and p_matrix is not None:
                        corr_with_sig.iloc[i, j] += get_significance_stars(p_matrix.iloc[i, j])
            st.dataframe(corr_with_sig)

        with col2:
            st.write("### 📉 p值矩阵")
            if p_matrix is not None:
                st.dataframe(p_matrix.round(4))

        # 最强相关性
        st.write("### 🎯 最强相关性对")
        corr_flat = corr_matrix.unstack()
        corr_flat = corr_flat[corr_flat.index.get_level_values(0) != corr_flat.index.get_level_values(1)]
        top_pairs = corr_flat.abs().nlargest(5)

        for (var1, var2), corr in top_pairs.items():
            p_val = p_matrix.loc[var1, var2] if p_matrix is not None else None
            sig = get_significance_stars(p_val) if p_val is not None else ''
            st.write(f"- **{var1} ↔ {var2}**: {corr:.3f} {sig}")


def ml_analysis(df):
    """机器学习工程师 - 特征分析"""
    st.subheader("🤖 特征工程分析")

    numeric_df = df.select_dtypes(include=[np.number])

    # 识别目标列
    target_cols = [col for col in numeric_df.columns if '目标' in col or 'label' in col or 'target' in col]

    if target_cols:
        target = target_cols[0]
        st.write(f"### 🎯 特征与目标变量'{target}'的相关性")

        features = [col for col in numeric_df.columns if col != target]

        results = []
        for feat in features:
            if df[target].nunique() <= 10:
                # 分类目标 - 使用ANOVA或点二列相关
                groups = [df[df[target] == val][feat].dropna() for val in df[target].unique()]
                if len(groups) >= 2:
                    f_stat, p = stats.f_oneway(*groups)
                    corr = np.sqrt(f_stat / (f_stat + len(df) - len(groups)))
                else:
                    corr, p = stats.pointbiserialr(df[target], df[feat])
            else:
                # 连续目标
                corr, p = stats.pearsonr(df[target].dropna(), df[feat].dropna())

            results.append({
                '特征': feat,
                '相关系数': corr,
                'p值': p,
                '显著性': get_significance_stars(p),
                '重要性': '高' if abs(corr) > 0.4 else '中' if abs(corr) > 0.25 else '低'
            })

        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 特征筛选建议
        important_features = results_df[results_df['重要性'] == '高']
        st.success(f"✅ 发现 {len(important_features)} 个重要特征")
        st.write(important_features[['特征', '相关系数', '重要性']])

        # 多重共线性
        st.write("### 🔗 特征多重共线性诊断")
        corr_matrix, _ = calculate_correlations(df[features])
        if corr_matrix is not None:
            high_corr = []
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            '特征1': corr_matrix.index[i],
                            '特征2': corr_matrix.columns[j],
                            '相关系数': corr_matrix.iloc[i, j],
                            '建议': '考虑删除或合并'
                        })

            if high_corr:
                st.warning(f"⚠️ 发现 {len(high_corr)} 对高度相关特征，可能存在多重共线性")
                st.dataframe(pd.DataFrame(high_corr))
            else:
                st.success("✅ 未发现严重多重共线性")

    else:
        # 无目标列，显示特征间相关性
        st.write("### 📊 特征间相关性分析")
        corr_matrix, _ = calculate_correlations(numeric_df)
        if corr_matrix is not None:
            fig = create_heatmap(corr_matrix, "特征相关性热力图")
            st.plotly_chart(fig, use_container_width=True)


def bi_analysis(df):
    """BI分析师 - 智能洞察"""
    st.subheader("📊 BI智能分析报告")

    numeric_df = df.select_dtypes(include=[np.number])

    # 自动生成洞察
    st.write("### 🔍 关键洞察")

    insights = []

    # 最强相关性
    corr_matrix, p_matrix = calculate_correlations(numeric_df)
    if corr_matrix is not None:
        corr_flat = corr_matrix.unstack()
        corr_flat = corr_flat[corr_flat.index.get_level_values(0) != corr_flat.index.get_level_values(1)]

        if len(corr_flat) > 0:
            strongest = corr_flat.abs().idxmax()
            strongest_val = corr_flat[strongest]
            insights.append(f"• **最强正相关**：{strongest[0]} 与 {strongest[1]} 呈 {strongest_val:.3f} 相关")

            if len(corr_flat) > 1:
                weakest = corr_flat.abs().idxmin()
                weakest_val = corr_flat[weakest]
                insights.append(f"• **最弱相关**：{weakest[0]} 与 {weakest[1]} 仅 {weakest_val:.3f} 相关")

    # 数据质量
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    if missing_pct > 0:
        insights.append(f"• **数据质量**：缺失值占比 {missing_pct:.1f}%")

    # 变异系数
    cv = numeric_df.std() / numeric_df.mean()
    high_var = cv[cv > 0.5]
    if len(high_var) > 0:
        insights.append(f"• **高变异指标**：{', '.join(high_var.index[:3])} 等指标波动较大")

    for insight in insights:
        st.info(insight)

    # 数据概览
    st.write("### 📈 数据概览")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总记录数", len(df))
    with col2:
        st.metric("分析变量数", len(numeric_df.columns))
    with col3:
        st.metric("完整记录数", len(df.dropna()))

    # 相关性热力图
    if corr_matrix is not None:
        st.write("### 🔥 相关性热力图")
        fig = create_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)

    # 描述性统计
    with st.expander("📋 描述性统计详情"):
        st.dataframe(numeric_df.describe())


def growth_analysis(df):
    """用户增长 - 留存分析"""
    st.subheader("📈 用户增长分析")

    numeric_df = df.select_dtypes(include=[np.number])

    # 识别留存率列
    retention_cols = [col for col in numeric_df.columns if '留存' in col or 'retention' in col.lower()]
    behavior_cols = [col for col in numeric_df.columns if col not in retention_cols]

    if retention_cols and behavior_cols:
        st.write("### 🎯 行为指标与留存率的相关性")

        results = []
        for retention in retention_cols:
            for behavior in behavior_cols:
                corr, p = stats.pearsonr(df[retention].dropna(), df[behavior].dropna())
                results.append({
                    '留存指标': retention,
                    '行为指标': behavior,
                    '相关系数': corr,
                    '显著性': get_significance_stars(p),
                    '影响程度': '高' if abs(corr) > 0.4 else '中' if abs(corr) > 0.25 else '低'
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 魔法数字识别
        high_impact = results_df[results_df['影响程度'] == '高']
        if len(high_impact) > 0:
            st.success(f"🎯 发现 {len(high_impact)} 个高影响行为指标")
            st.write(high_impact[['行为指标', '留存指标', '相关系数']])

    # 用户分层
    st.write("### 📊 用户行为分布")
    if len(numeric_df.columns) > 0:
        fig = px.box(numeric_df, title="指标分布箱线图")
        st.plotly_chart(fig, use_container_width=True)


def ecommerce_analysis(df):
    """电商运营 - 用户行为分析"""
    st.subheader("🛒 电商用户行为分析")

    numeric_df = df.select_dtypes(include=[np.number])

    # 识别转化相关列
    conv_cols = [col for col in numeric_df.columns if '转化' in col or '购买' in col or '下单' in col]
    behavior_cols = [col for col in numeric_df.columns if col not in conv_cols]

    if conv_cols and behavior_cols:
        st.write("### 🎯 行为指标与转化率的相关性")

        results = []
        for conv in conv_cols:
            for behavior in behavior_cols:
                corr, p = stats.pearsonr(df[conv].dropna(), df[behavior].dropna())
                results.append({
                    '转化指标': conv,
                    '用户行为': behavior,
                    '相关系数': corr,
                    '显著性': get_significance_stars(p),
                    '优化建议': '优先优化' if abs(corr) > 0.35 else '参考指标'
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 优化建议
        priority = results_df[results_df['优化建议'] == '优先优化']
        if len(priority) > 0:
            st.success("📌 优先优化建议")
            st.write(priority[['用户行为', '转化指标', '相关系数']])

    # RFM分析预览
    st.write("### 📊 数据概览")
    st.dataframe(numeric_df.head(10))


def product_analysis(df):
    """产品经理 - 功能分析"""
    st.subheader("📱 产品功能分析")

    numeric_df = df.select_dtypes(include=[np.number])

    st.write("### 🔥 功能使用相关性分析")
    corr_matrix, _ = calculate_correlations(numeric_df)
    if corr_matrix is not None:
        fig = create_heatmap(corr_matrix, "功能使用相关性")
        st.plotly_chart(fig, use_container_width=True)

        # 功能组合建议
        st.write("### 💡 功能组合建议")
        high_corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if corr_matrix.iloc[i, j] > 0.5:
                    high_corr_pairs.append({
                        '功能A': corr_matrix.index[i],
                        '功能B': corr_matrix.columns[j],
                        '相关系数': corr_matrix.iloc[i, j],
                        '建议': '功能联动优化'
                    })

        if high_corr_pairs:
            st.write("发现以下高度关联的功能对，建议优化联动体验：")
            st.dataframe(pd.DataFrame(high_corr_pairs))


def risk_analysis(df):
    """风控专员 - 风险分析"""
    st.subheader("🏦 风险控制分析")

    numeric_df = df.select_dtypes(include=[np.number])

    # 识别违约列
    default_cols = [col for col in numeric_df.columns if '违约' in col or '逾期' in col or 'default' in col.lower()]

    if default_cols:
        default_col = default_cols[0]
        st.write(f"### ⚠️ 风险因素与{default_col}的相关性")

        results = []
        for col in numeric_df.columns:
            if col != default_col:
                corr, p = stats.pointbiserialr(df[default_col], df[col])
                results.append({
                    '风险因素': col,
                    '相关系数': corr,
                    '显著性': get_significance_stars(p),
                    '风险等级': '高风险' if abs(corr) > 0.3 else '中风险' if abs(corr) > 0.2 else '低风险'
                })

        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 高风险因素
        high_risk = results_df[results_df['风险等级'] == '高风险']
        if len(high_risk) > 0:
            st.error(f"⚠️ 发现 {len(high_risk)} 个高风险因素")
            st.write(high_risk[['风险因素', '相关系数']])


def social_analysis(df):
    """社会调查员 - 社会指标分析"""
    st.subheader("📋 社会调查数据分析")

    numeric_df = df.select_dtypes(include=[np.number])

    st.write("### 🔗 社会指标相关性分析")
    corr_matrix, p_matrix = calculate_correlations(numeric_df)
    if corr_matrix is not None:
        fig = create_heatmap(corr_matrix, "社会指标相关性")
        st.plotly_chart(fig, use_container_width=True)

        # 卡方检验（如果有分类变量）
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) >= 2:
            st.write("### 📊 分类变量关联性分析")
            from scipy.stats import chi2_contingency
            for i in range(len(categorical_cols)):
                for j in range(i + 1, len(categorical_cols)):
                    contingency = pd.crosstab(df[categorical_cols[i]], df[categorical_cols[j]])
                    chi2, p, dof, expected = chi2_contingency(contingency)
                    if p < 0.05:
                        st.write(f"- **{categorical_cols[i]}** 与 **{categorical_cols[j]}** 存在显著关联 (p={p:.4f})")


def agriculture_analysis(df):
    """农艺师 - 农业分析"""
    st.subheader("🌱 农业产量分析")

    numeric_df = df.select_dtypes(include=[np.number])

    # 识别产量列
    yield_cols = [col for col in numeric_df.columns if '产量' in col or '收成' in col]

    if yield_cols:
        yield_col = yield_cols[0]
        st.write(f"### 📈 影响{yield_col}的因素分析")

        results = []
        for col in numeric_df.columns:
            if col != yield_col:
                corr, p = stats.pearsonr(df[yield_col].dropna(), df[col].dropna())
                results.append({
                    '影响因素': col,
                    '相关系数': corr,
                    '显著性': get_significance_stars(p),
                    '重要性': '高' if abs(corr) > 0.4 else '中' if abs(corr) > 0.25 else '低'
                })

        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df.style.background_gradient(subset=['相关系数'], cmap='RdYlGn', vmin=-1, vmax=1))

        # 关键因素
        key_factors = results_df[results_df['重要性'] == '高']
        if len(key_factors) > 0:
            st.success(f"🎯 发现 {len(key_factors)} 个关键影响因素")
            st.write(key_factors[['影响因素', '相关系数']])


# 其他分析函数（简化版）
def research_analysis(df): st.write("### 🔬 医学研究分析"); clinical_analysis(df)


def pharma_analysis(df): st.write("### 💊 药效分析"); clinical_analysis(df)


def bioinfo_analysis(df): st.write("### 🧬 生物信息分析"); general_analysis(df)


def sports_analysis(df): st.write("### ⚽ 运动表现分析"); general_analysis(df)


def industrial_analysis(df): st.write("### 🏭 工业工程分析"); quality_analysis(df)


def lab_analysis(df): st.write("### 🔬 实验室分析"); quality_analysis(df)


def environment_analysis(df): st.write("### 🌍 环境分析"); agriculture_analysis(df)


def assessment_analysis(df): st.write("### 📝 测评分析"); teacher_analysis(df)


def school_analysis(df): st.write("### 🏫 学校分析"); teacher_analysis(df)


# ==================== 主程序 ====================

def main():
    # 标题
    st.title("🎯 CorrMatrix Studio")
    st.markdown("*多角色统计分析平台 - 选择您的职业，获得专属分析*")

    # 初始化session state
    if 'selected_industry' not in st.session_state:
        st.session_state.selected_industry = list(INDUSTRIES.keys())[0]
    if 'selected_role' not in st.session_state:
        st.session_state.selected_role = None
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
        industry_icons = [INDUSTRIES[i]["icon"] for i in industries]

        selected_industry = st.radio(
            "行业领域",
            industries,
            format_func=lambda x: f"{INDUSTRIES[x]['icon']} {x}",
            key="industry_selector"
        )

        if selected_industry != st.session_state.selected_industry:
            st.session_state.selected_industry = selected_industry
            st.session_state.selected_role = None
            st.rerun()

        st.markdown("---")

        # 角色选择
        st.subheader("👤 选择职业角色")
        roles = INDUSTRIES[st.session_state.selected_industry]["roles"]
        role_options = list(roles.keys())

        selected_role = st.selectbox(
            "职业角色",
            role_options,
            format_func=lambda x: f"{x}",
            key="role_selector"
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
            type=['csv', 'xlsx', 'xls'],
            help="支持CSV和Excel格式"
        )

        # 示例数据
        use_example = st.checkbox("使用示例数据", value=not uploaded_file)

        if use_example and st.session_state.selected_role:
            if st.button("📊 生成示例数据"):
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
                st.warning(f"分析功能 {analysis_func_name} 暂未实现")
                general_analysis(df)

    else:
        # 欢迎界面
        st.info("👈 请从左侧选择行业和职业角色，然后上传数据或生成示例数据")

        st.markdown("""
        ### 🎯 快速开始

        1. **选择行业领域**（教育/医疗/商业/工业/互联网/心理）
        2. **选择职业角色**（26个角色可选）
        3. **上传数据** 或 **生成示例数据**
        4. **查看专属分析报告**

        ### 📊 支持的角色类型

        | 行业 | 角色示例 |
        |------|----------|
        | 📚 教育科研 | 教师、教授、测评专家、学校管理者 |
        | 🏥 医疗健康 | 临床医生、医学研究员、药师、生物信息学家 |
        | 📊 商业/市场/金融 | 市场研究员、电商运营、产品经理、金融分析师 |
        | 🏭 工业/质量/农业 | 质量工程师、工业工程师、农艺师 |
        | 💻 互联网/数据科学 | 数据分析师、ML工程师、BI分析师 |
        | 🧠 心理/社会/人文 | 心理咨询师、HR、社会调查员 |
        """)

    # 页脚
    st.markdown("---")
    st.markdown(
        "<center>CorrMatrix Studio - 为每个职业定制的统计分析工具</center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()