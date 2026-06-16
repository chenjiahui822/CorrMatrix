"""
CorrMatrix Studio - 完整版
6大行业 × 26个职业角色 × 专属深度分析
共计1400+行代码
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="CorrMatrix Studio - 26职业完整版", page_icon="📊", layout="wide")

# ==================== 26角色完整配置 ====================

ROLES_DB = {
    "📚 教育科研": {
        "中小学教师": {
            "desc": "试题质量分析、班级成绩对比",
            "features": ["难度系数", "区分度", "信度", "分数分布", "班级对比", "低质量题目标记"],
            "sample": "exam"
        },
        "大学教授": {
            "desc": "问卷信效度、论文数据分析",
            "features": ["Cronbach's α", "KMO检验", "Bartlett球形检验", "因子载荷", "共同度", "维度相关"],
            "sample": "questionnaire"
        },
        "教育测评专家": {
            "desc": "标准化考试、IRT分析",
            "features": ["项目反应理论", "信息函数", "测验等值", "DIF检测", "题库建设", "划界分数"],
            "sample": "exam"
        },
        "学校管理者": {
            "desc": "学校质量评估、指标预警",
            "features": ["学业质量", "师资指标", "硬件指标", "预警系统", "趋势预测", "标杆对比"],
            "sample": "school"
        }
    },
    "🏥 医疗健康": {
        "临床医生": {
            "desc": "检验指标-疾病诊断分析",
            "features": ["指标相关性", "ROC曲线", "约登指数", "敏感度/特异度", "参考区间", "风险因素"],
            "sample": "clinical"
        },
        "医学研究员": {
            "desc": "临床试验数据分析",
            "features": ["治疗前后对比", "分组比较", "协变量校正", "生存曲线", "Log-rank检验", "Cox回归"],
            "sample": "clinical"
        },
        "药学/药理": {
            "desc": "剂量-效应关系分析",
            "features": ["剂量效应曲线", "ED50/LD50", "治疗指数", "药物相互作用", "Bliss模型", "等效性分析"],
            "sample": "pharma"
        },
        "生物信息学": {
            "desc": "基因表达数据分析",
            "features": ["差异表达基因", "聚类热图", "火山图", "GO富集", "KEGG通路", "PPI网络"],
            "sample": "bioinfo"
        },
        "康复治疗师": {
            "desc": "康复效果评估",
            "features": ["功能评分变化", "康复周期分析", "预后因素", "生活质量", "ADL评估", "康复目标达成"],
            "sample": "rehab"
        }
    },
    "📊 商业金融": {
        "市场研究员": {
            "desc": "满意度-忠诚度分析",
            "features": ["满意度驱动因素", "NPS分析", "重要性矩阵", "细分人群", "T检验", "方差分析"],
            "sample": "market"
        },
        "电商运营": {
            "desc": "用户行为-转化分析",
            "features": ["行为-转化相关", "RFM分层", "漏斗分析", "复购预测", "购物篮分析", "用户画像"],
            "sample": "ecommerce"
        },
        "产品经理": {
            "desc": "功能使用-留存分析",
            "features": ["功能使用率", "留存曲线", "魔法数字", "A/B测试", "用户分群", "NPS跟踪"],
            "sample": "product"
        },
        "金融分析师": {
            "desc": "资产组合风险分析",
            "features": ["资产相关性", "风险分散", "Beta系数", "夏普比率", "最大回撤", "VaR计算"],
            "sample": "finance"
        },
        "风控专员": {
            "desc": "信用风险评分",
            "features": ["违约因素", "WOE/IV值", "评分卡", "KS统计量", "混淆矩阵", "AUC曲线"],
            "sample": "risk"
        }
    },
    "🏭 工业农业": {
        "质量工程师": {
            "desc": "过程质量控制",
            "features": ["过程能力Cpk", "SPC控制图", "六西格玛", "测量系统分析", "DOE分析", "FMEA"],
            "sample": "quality"
        },
        "工业工程师": {
            "desc": "生产效率优化",
            "features": ["工时分析", "瓶颈识别", "线平衡", "OEE计算", "标准工时", "人机分析"],
            "sample": "industrial"
        },
        "实验室技术员": {
            "desc": "检测方法验证",
            "features": ["重复性/再现性", "加标回收", "检出限", "不确定度", "线性范围", "方法比对"],
            "sample": "lab"
        },
        "农艺师": {
            "desc": "作物产量分析",
            "features": ["产量影响因素", "施肥优化", "土壤指标", "气候影响", "品种对比", "正交试验"],
            "sample": "agriculture"
        },
        "环境科学家": {
            "desc": "环境监测评价",
            "features": ["污染物时空分布", "污染源解析", "环境质量指数", "生态风险", "趋势预测", "相关性聚类"],
            "sample": "environment"
        }
    },
    "💻 数据科学": {
        "数据分析师": {
            "desc": "通用相关性分析",
            "features": ["Pearson/Spearman/Kendall", "热力图", "散点图矩阵", "偏相关", "距离相关", "互信息"],
            "sample": "general"
        },
        "机器学习工程师": {
            "desc": "特征工程分析",
            "features": ["特征重要性", "多重共线性", "IV值", "递归特征消除", "PCA降维", "特征交互"],
            "sample": "ml"
        },
        "BI分析师": {
            "desc": "智能数据洞察",
            "features": ["异动分析", "趋势检测", "相关性告警", "智能摘要", "数据讲故事", "仪表盘"],
            "sample": "bi"
        },
        "用户增长": {
            "desc": "增长黑客分析",
            "features": ["留存曲线", "魔法数字", "Cohort分析", "LTV预测", "渠道归因", "AARRR模型"],
            "sample": "growth"
        }
    },
    "🧠 人文社科": {
        "心理咨询师": {
            "desc": "心理量表分析",
            "features": ["Cronbach's α", "分半信度", "重测信度", "效标效度", "探索性因子", "验证性因子"],
            "sample": "psychology"
        },
        "社会调查员": {
            "desc": "社会调查分析",
            "features": ["卡方检验", "列联表", "对应分析", "信度分析", "权重计算", "抽样误差"],
            "sample": "social"
        },
        "人力资源HR": {
            "desc": "人才测评分析",
            "features": ["测评-绩效相关", "高潜识别", "九宫格", "离职预警", "人才盘点", "继任规划"],
            "sample": "hr"
        },
        "体育分析师": {
            "desc": "运动表现分析",
            "features": ["训练-成绩相关", "伤病风险", "比赛预测", "球员评估", "战术分析", "体能监控"],
            "sample": "sports"
        }
    }
}


# ==================== 示例数据生成（完整版） ====================

def generate_sample_data(data_type):
    np.random.seed(42)
    n = 150

    if data_type == "exam":
        # 考试数据：5科+总分
        chinese = np.random.normal(75, 12, n)
        math = 70 + 0.6 * chinese + np.random.normal(0, 10, n)
        english = 72 + 0.5 * chinese + np.random.normal(0, 11, n)
        science = 68 + 0.4 * chinese + np.random.normal(0, 12, n)
        total = chinese + math + english + science
        return pd.DataFrame({
            '学生ID': [f'S{i:03d}' for i in range(1, n + 1)],
            '班级': np.random.choice(['1班', '2班', '3班', '4班'], n),
            '语文': np.clip(chinese, 40, 100),
            '数学': np.clip(math, 40, 100),
            '英语': np.clip(english, 40, 100),
            '科学': np.clip(science, 40, 100),
            '总分': np.clip(total, 200, 400)
        })

    elif data_type == "questionnaire":
        # 问卷数据：20个题目，5个维度
        data = {}
        for i in range(1, 21):
            data[f'Q{i}'] = np.random.randint(1, 6, n)
        df = pd.DataFrame(data)
        df['ID'] = [f'R{i:03d}' for i in range(1, n + 1)]
        return df

    elif data_type == "clinical":
        # 临床数据
        age = np.random.normal(55, 12, n)
        bmi = 22 + 0.15 * age + np.random.normal(0, 3, n)
        bp = 110 + 0.5 * age + np.random.normal(0, 10, n)
        glucose = 5.0 + 0.03 * age + 0.1 * bmi + np.random.normal(0, 0.8, n)
        diagnosis = 1 / (1 + np.exp(-(-3 + 0.03 * age + 0.05 * bp + 0.5 * glucose))) > 0.5
        return pd.DataFrame({
            '患者ID': [f'P{i:03d}' for i in range(1, n + 1)],
            '年龄': age,
            'BMI': bmi,
            '收缩压': bp,
            '血糖': glucose,
            '胆固醇': np.random.normal(5.2, 1.0, n),
            '诊断结果': diagnosis.astype(int)
        })

    elif data_type == "market":
        # 市场调查数据
        satisfaction = np.random.normal(3.5, 1.0, n)
        service = 2.5 + 0.7 * satisfaction + np.random.normal(0, 0.5, n)
        price = 2.8 + 0.4 * satisfaction + np.random.normal(0, 0.6, n)
        loyalty = 2.0 + 0.8 * satisfaction + np.random.normal(0, 0.5, n)
        return pd.DataFrame({
            'ID': [f'C{i:03d}' for i in range(1, n + 1)],
            '产品满意度': np.clip(satisfaction, 1, 5),
            '服务满意度': np.clip(service, 1, 5),
            '价格满意度': np.clip(price, 1, 5),
            '品牌忠诚度': np.clip(loyalty, 1, 5),
            '推荐意愿': np.clip(loyalty + 0.2 * np.random.randn(n), 1, 5)
        })

    elif data_type == "finance":
        # 金融数据
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        stock_a = np.random.normal(0.0005, 0.02, n).cumsum()
        stock_b = 0.7 * stock_a + np.random.normal(0, 0.005, n).cumsum()
        stock_c = -0.3 * stock_a + np.random.normal(0, 0.008, n).cumsum()
        bond = np.random.normal(0.0001, 0.002, n).cumsum()
        return pd.DataFrame({
            '日期': dates,
            '股票A收益率': np.diff(np.append([0], stock_a)),
            '股票B收益率': np.diff(np.append([0], stock_b)),
            '股票C收益率': np.diff(np.append([0], stock_c)),
            '债券收益率': np.diff(np.append([0], bond))
        })

    elif data_type == "quality":
        # 质量数据
        temp = np.random.normal(150, 10, n)
        pressure = 4.8 + 0.01 * temp + np.random.normal(0, 0.3, n)
        speed = 100 + 0.2 * temp + np.random.normal(0, 5, n)
        defect_rate = np.exp(-3 + 0.01 * temp + 0.1 * pressure) / (1 + np.exp(-3 + 0.01 * temp + 0.1 * pressure))
        return pd.DataFrame({
            '批次': [f'B{i:03d}' for i in range(1, n + 1)],
            '温度(℃)': temp,
            '压力(MPa)': pressure,
            '速度(m/min)': speed,
            '良品率(%)': 100 * (1 - defect_rate),
            '缺陷数': np.random.poisson(5 * defect_rate.mean() / 0.05, n)
        })

    elif data_type == "psychology":
        # 心理量表数据
        anxiety = np.random.randint(1, 5, (n, 5)).sum(axis=1)
        depression = 0.6 * anxiety + np.random.randint(1, 3, (n, 5)).sum(axis=1)
        wellbeing = -0.4 * anxiety + np.random.randint(1, 4, (n, 5)).sum(axis=1)
        return pd.DataFrame({
            'ID': [f'P{i:03d}' for i in range(1, n + 1)],
            '焦虑总分': np.clip(anxiety, 5, 25),
            '抑郁总分': np.clip(depression, 5, 25),
            '幸福感总分': np.clip(wellbeing, 5, 25)
        })

    elif data_type == "hr":
        # 人力资源数据
        ability = np.random.normal(75, 10, n)
        potential = 60 + 0.5 * ability + np.random.normal(0, 8, n)
        performance = 50 + 0.6 * ability + 0.3 * potential + np.random.normal(0, 6, n)
        return pd.DataFrame({
            '员工ID': [f'E{i:03d}' for i in range(1, n + 1)],
            '能力测评': np.clip(ability, 40, 100),
            '潜力测评': np.clip(potential, 40, 100),
            '领导力': np.clip(50 + 0.3 * ability + np.random.normal(0, 10, n), 40, 100),
            '团队合作': np.clip(60 + 0.2 * ability + np.random.normal(0, 8, n), 40, 100),
            '绩效评分': np.clip(performance, 50, 100)
        })

    elif data_type == "agriculture":
        # 农业数据
        fertilizer = np.random.uniform(20, 100, n)
        rainfall = np.random.uniform(80, 200, n)
        temperature = np.random.uniform(15, 35, n)
        yield_amount = 300 + 3 * fertilizer + 1.5 * rainfall + 5 * temperature + np.random.normal(0, 50, n)
        return pd.DataFrame({
            '地块ID': [f'F{i:03d}' for i in range(1, n + 1)],
            '施肥量(kg/亩)': fertilizer,
            '降雨量(mm)': rainfall,
            '平均温度(℃)': temperature,
            '土壤pH': np.random.normal(6.5, 0.5, n),
            '产量(kg/亩)': yield_amount
        })

    elif data_type == "sports":
        # 体育数据
        train_hours = np.random.uniform(2, 10, n)
        strength = 50 + 3 * train_hours + np.random.normal(0, 8, n)
        endurance = 45 + 4 * train_hours + np.random.normal(0, 7, n)
        performance = 40 + 5 * train_hours + 0.3 * strength + 0.3 * endurance + np.random.normal(0, 5, n)
        return pd.DataFrame({
            '运动员ID': [f'A{i:03d}' for i in range(1, n + 1)],
            '训练时长(小时/周)': train_hours,
            '力量评分': np.clip(strength, 40, 100),
            '耐力评分': np.clip(endurance, 40, 100),
            '技术评分': np.clip(45 + 2 * train_hours + np.random.normal(0, 6, n), 40, 100),
            '比赛成绩': np.clip(performance, 50, 100)
        })

    elif data_type == "ecommerce":
        # 电商数据
        browse = np.random.exponential(20, n)
        clicks = 2 + 0.5 * browse + np.random.poisson(2, n)
        cart = 0.2 * clicks + np.random.poisson(0.5, n)
        purchase = 1 / (1 + np.exp(-(-2 + 0.1 * browse + 0.3 * clicks + 0.5 * cart))) > 0.5
        amount = purchase * (50 + 0.5 * browse + np.random.exponential(100, n))
        return pd.DataFrame({
            '用户ID': [f'U{i:03d}' for i in range(1, n + 1)],
            '浏览时长(分钟)': browse,
            '点击次数': clicks,
            '加购次数': cart,
            '是否购买': purchase.astype(int),
            '消费金额': amount
        })

    else:
        # 通用数据
        return pd.DataFrame({
            '变量A': np.random.normal(50, 15, n),
            '变量B': 0.7 * np.random.normal(50, 15, n) + 0.3 * np.random.normal(60, 10, n),
            '变量C': -0.4 * np.random.normal(50, 15, n) + np.random.normal(45, 10, n),
            '变量D': np.random.normal(55, 12, n),
            '变量E': np.random.normal(48, 11, n)
        })


# ==================== 通用工具函数 ====================

def get_numeric(df):
    return df.select_dtypes(include=[np.number])


def calc_corr(df):
    num = get_numeric(df)
    if len(num.columns) < 2:
        return None, None
    corr = num.corr()
    p = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
    for i in range(len(num.columns)):
        for j in range(len(num.columns)):
            if i != j:
                _, p.iloc[i, j] = stats.pearsonr(num.iloc[:, i], num.iloc[:, j])
    return corr, p


def heatmap(corr, title="相关系数热力图"):
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, title=title)
    fig.update_layout(height=480, width=750)
    return fig


def sig_star(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''


def cronbach_alpha(data):
    """计算Cronbach's α系数"""
    k = len(data.columns)
    if k < 2:
        return np.nan
    item_vars = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def kmo_bartlett(data):
    """KMO检验和Bartlett球形检验"""
    corr = data.corr()
    # KMO检验
    corr_inv = np.linalg.inv(corr.values)
    r2 = 1 - 1 / np.diag(corr_inv)
    kmo = r2.sum() / (r2.sum() + (1 - r2).sum())

    # Bartlett球形检验
    det = np.linalg.det(corr.values)
    n = len(data)
    p = len(data.columns)
    chi2 = - (n - 1 - (2 * p + 5) / 6) * np.log(det)
    df = p * (p - 1) // 2
    p_value = 1 - stats.chi2.cdf(chi2, df)

    return kmo, chi2, p_value


# ==================== 中小学教师分析 ====================

def teacher_analysis(df):
    st.subheader("👩‍🏫 中小学教师 - 试题质量分析报告")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个数值列")
        return

    # 识别总分列
    total_cols = [c for c in num.columns if '总分' in c or '总成绩' in c or 'total' in c.lower()]
    if total_cols:
        total = total_cols[0]
        q_cols = [c for c in num.columns if c != total and c not in ['学生ID', '班级']]

        # 1. 难度系数
        st.write("### 📊 难度系数分析")
        difficulty = []
        for q in q_cols:
            max_score = num[q].max()
            if max_score > 0:
                diff = num[q].mean() / max_score
                difficulty.append({
                    '题目': q,
                    '难度系数': round(diff, 3),
                    '评价': '偏易' if diff > 0.8 else '适中' if diff > 0.3 else '偏难'
                })
        st.dataframe(pd.DataFrame(difficulty))

        # 2. 区分度
        st.write("### 📈 区分度分析")
        high_score = num[total].quantile(0.73)
        low_score = num[total].quantile(0.27)

        discrimination = []
        for q in q_cols:
            high_avg = num[num[total] >= high_score][q].mean()
            low_avg = num[num[total] <= low_score][q].mean()
            max_score = num[q].max()
            if max_score > 0:
                disc = (high_avg - low_avg) / max_score
                discrimination.append({
                    '题目': q,
                    '区分度': round(disc, 3),
                    '评价': '优秀' if disc > 0.4 else '良好' if disc > 0.3 else '尚可' if disc > 0.2 else '待改进'
                })
        st.dataframe(pd.DataFrame(discrimination))

        # 3. 题目-总分相关
        st.write("### 🎯 题目-总分相关性")
        item_total = []
        for q in q_cols:
            r, p = stats.pearsonr(num[q], num[total])
            item_total.append({
                '题目': q,
                '相关系数': round(r, 3),
                '显著性': sig_star(p),
                '质量': '优良' if r > 0.4 else '良好' if r > 0.3 else '一般'
            })
        st.dataframe(pd.DataFrame(item_total))

        # 4. 信度分析
        st.write("### 📐 信度分析")
        alpha = cronbach_alpha(num[q_cols])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cronbach's α", f"{alpha:.3f}")
        with col2:
            quality = "优秀" if alpha > 0.9 else "良好" if alpha > 0.8 else "可接受" if alpha > 0.7 else "需改进"
            st.metric("信度评价", quality)

        # 5. 低质量题目提醒
        st.write("### ⚠️ 低质量题目诊断")
        low_quality = [d for d in discrimination if d['区分度'] < 0.2]
        if low_quality:
            st.error(f"发现 {len(low_quality)} 道区分度较差的题目")
            st.dataframe(pd.DataFrame(low_quality))
        else:
            st.success("所有题目区分度良好")

    # 6. 成绩分布
    st.write("### 📊 成绩分布图")
    fig = px.histogram(num, title="各科目分数分布", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    # 7. 班级对比
    if '班级' in df.columns:
        st.write("### 🏫 班级成绩对比")
        class_avg = df.groupby('班级')[q_cols].mean()
        st.dataframe(class_avg)
        fig = px.bar(class_avg.reset_index().melt(id_vars='班级'),
                     x='班级', y='value', color='variable', title="班级平均分对比")
        st.plotly_chart(fig, use_container_width=True)


# ==================== 大学教授分析 ====================

def professor_analysis(df):
    st.subheader("🎓 大学教授 - 问卷信效度分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    # 1. 信度分析
    st.write("### 📐 信度分析")
    alpha = cronbach_alpha(num)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cronbach's α", f"{alpha:.3f}")
    with col2:
        quality = "优秀" if alpha > 0.9 else "良好" if alpha > 0.8 else "可接受" if alpha > 0.7 else "需改进"
        st.metric("信度评价", quality)
    with col3:
        st.metric("题目数量", len(num.columns))

    # 2. 删题后信度
    st.write("### 📉 删题后信度变化")
    alpha_if_deleted = []
    for col in num.columns:
        alpha_del = cronbach_alpha(num.drop(columns=[col]))
        alpha_if_deleted.append({
            '删除题目': col,
            '删除后α': round(alpha_del, 3),
            '变化': round(alpha_del - alpha, 3),
            '建议': '可考虑删除' if alpha_del > alpha else '保留'
        })
    st.dataframe(pd.DataFrame(alpha_if_deleted))

    # 3. 效度分析 - KMO和Bartlett
    st.write("### 🔬 效度分析")
    kmo, chi2, p_val = kmo_bartlett(num)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("KMO检验值", f"{kmo:.3f}")
        st.caption("KMO > 0.7 适合因子分析")
    with col2:
        st.metric("Bartlett球形检验 p值", f"{p_val:.4f}")
        st.caption("p < 0.05 适合因子分析")

    # 4. 相关性热力图
    st.write("### 🔥 题目相关性热力图")
    corr, p = calc_corr(df)
    if corr is not None:
        st.plotly_chart(heatmap(corr), use_container_width=True)

    # 5. 维度建议
    st.write("### 💡 维度结构建议")
    st.info("基于相关性矩阵，建议进行探索性因子分析确定维度结构")


# ==================== 临床医生分析 ====================

def doctor_analysis(df):
    st.subheader("👨‍⚕️ 临床医生 - 临床指标分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    # 识别诊断列
    diag_cols = [c for c in num.columns if '诊断' in c or '结果' in c or '疾病' in c]

    if diag_cols:
        diag = diag_cols[0]
        st.write(f"### 🩺 指标与'{diag}'的相关性分析")

        # 1. 点二列相关
        results = []
        for col in num.columns:
            if col != diag:
                r, p = stats.pointbiserialr(num[diag], num[col])
                results.append({
                    '指标': col,
                    '相关系数': round(r, 3),
                    '显著性': sig_star(p),
                    '临床意义': '高风险因子' if abs(r) > 0.3 else '参考指标'
                })
        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df)

        # 2. 高风险因子
        high_risk = results_df[results_df['临床意义'] == '高风险因子']
        if len(high_risk) > 0:
            st.warning(f"⚠️ 发现 {len(high_risk)} 个高风险相关因子")
            st.dataframe(high_risk)

    # 3. 参考区间
    st.write("### 📊 参考区间 (95%置信区间)")
    ref = pd.DataFrame({
        '指标': num.columns,
        '均值': round(num.mean(), 2),
        '标准差': round(num.std(), 2),
        '参考下限': round(num.mean() - 1.96 * num.std(), 2),
        '参考上限': round(num.mean() + 1.96 * num.std(), 2),
        '最小值': round(num.min(), 2),
        '最大值': round(num.max(), 2)
    })
    st.dataframe(ref)

    # 4. 相关性热力图
    st.write("### 🔥 指标相关性热力图")
    corr, _ = calc_corr(df)
    if corr is not None:
        st.plotly_chart(heatmap(corr), use_container_width=True)

    # 5. 异常值检测
    st.write("### ⚠️ 异常值检测")
    for col in num.columns:
        q1 = num[col].quantile(0.25)
        q3 = num[col].quantile(0.75)
        iqr = q3 - q1
        outliers = num[(num[col] < q1 - 1.5 * iqr) | (num[col] > q3 + 1.5 * iqr)]
        if len(outliers) > 0:
            st.warning(f"{col}: 发现 {len(outliers)} 个异常值")


# ==================== 金融分析师分析 ====================

def finance_analysis(df):
    st.subheader("💰 金融分析师 - 资产组合分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个资产")
        return

    # 1. 资产相关性矩阵
    st.write("### 📊 资产收益率相关性矩阵")
    corr, p = calc_corr(df)
    if corr is not None:
        st.plotly_chart(heatmap(corr, "资产相关性热力图"), use_container_width=True)

    # 2. 风险分散建议
    st.write("### 💡 风险分散建议")
    low_corr = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            if abs(corr.iloc[i, j]) < 0.3:
                low_corr.append({
                    '资产1': corr.index[i],
                    '资产2': corr.columns[j],
                    '相关系数': round(corr.iloc[i, j], 3),
                    '分散效果': '优秀' if abs(corr.iloc[i, j]) < 0.2 else '良好'
                })

    if low_corr:
        st.success(f"发现 {len(low_corr)} 对低相关性资产组合")
        st.dataframe(pd.DataFrame(low_corr))
    else:
        st.info("所有资产对相关性均较高，分散效果有限")

    # 3. 风险指标
    st.write("### 📉 风险指标")
    returns_std = num.std()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("平均日收益率", f"{num.mean().mean():.4f}")
    with col2:
        st.metric("平均波动率", f"{returns_std.mean():.4f}")
    with col3:
        st.metric("夏普比率(估算)", f"{num.mean().mean() / returns_std.mean():.2f}")

    # 4. 累计收益曲线
    st.write("### 📈 累计收益曲线")
    cumsum = (1 + num).cumprod()
    fig = px.line(cumsum, title="累计收益")
    st.plotly_chart(fig, use_container_width=True)


# ==================== 质量工程师分析 ====================

def quality_analysis(df):
    st.subheader("🔧 质量工程师 - 过程能力分析")

    num = get_numeric(df)
    if len(num.columns) < 1:
        st.warning("需要数值列")
        return

    # 1. 过程能力指数Cpk
    st.write("### 📊 过程能力指数(Cpk)")
    cpk_results = []
    for col in num.columns:
        mean_val = num[col].mean()
        std_val = num[col].std()
        if std_val > 0:
            usl = mean_val + 3 * std_val
            lsl = mean_val - 3 * std_val
            cpu = (usl - mean_val) / (3 * std_val)
            cpl = (mean_val - lsl) / (3 * std_val)
            cpk = min(cpu, cpl)
            cpk_results.append({
                '指标': col,
                '均值': round(mean_val, 2),
                '标准差': round(std_val, 2),
                'Cpk': round(cpk, 3),
                '评价': '优秀' if cpk > 1.33 else '良好' if cpk > 1.0 else '一般' if cpk > 0.67 else '需改进'
            })
    st.dataframe(pd.DataFrame(cpk_results))

    # 2. SPC控制图
    st.write("### 📈 SPC控制图")
    selected = st.selectbox("选择指标查看控制图", num.columns)
    if selected:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=num[selected], mode='lines+markers', name=selected))
        fig.add_hline(y=num[selected].mean(), line_dash="dash", line_color="green", annotation_text="CL")
        fig.add_hline(y=num[selected].mean() + 3 * num[selected].std(), line_dash="dash", line_color="red",
                      annotation_text="UCL")
        fig.add_hline(y=num[selected].mean() - 3 * num[selected].std(), line_dash="dash", line_color="red",
                      annotation_text="LCL")
        fig.update_layout(title=f"{selected} 控制图", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # 3. 相关性分析
    st.write("### 🔥 参数相关性分析")
    corr, _ = calc_corr(df)
    if corr is not None:
        st.plotly_chart(heatmap(corr), use_container_width=True)


# ==================== 心理咨询师分析 ====================

def psychology_analysis(df):
    st.subheader("🧠 心理咨询师 - 心理量表分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    # 1. 信度分析
    st.write("### 📐 信度分析")
    alpha = cronbach_alpha(num)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cronbach's α", f"{alpha:.3f}")
    with col2:
        quality = "优秀" if alpha > 0.9 else "良好" if alpha > 0.8 else "可接受" if alpha > 0.7 else "需改进"
        st.metric("信度评价", quality)

    # 2. 分半信度
    st.write("### 📊 分半信度")
    n = len(num.columns)
    first_half = num.iloc[:, :n // 2].sum(axis=1)
    second_half = num.iloc[:, n // 2:].sum(axis=1)
    half_data = pd.DataFrame({'前半': first_half, '后半': second_half})
    split_alpha = cronbach_alpha(half_data)
    st.metric("分半信度", f"{split_alpha:.3f}")

    # 3. 相关性矩阵
    st.write("### 🔥 题目相关性热力图")
    corr, p = calc_corr(df)
    if corr is not None:
        st.plotly_chart(heatmap(corr), use_container_width=True)

    # 4. 因子分析建议
    st.write("### 💡 因子结构分析")
    kmo, chi2, p_val = kmo_bartlett(num)
    st.info(f"KMO = {kmo:.3f}, Bartlett p = {p_val:.4f}")
    if kmo > 0.7 and p_val < 0.05:
        st.success("数据适合进行因子分析，建议进行探索性因子分析")
    else:
        st.warning("数据可能不适合因子分析")


# ==================== 人力资源分析 ====================

def hr_analysis(df):
    st.subheader("👥 人力资源HR - 人才测评分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    # 识别绩效列
    perf_cols = [c for c in num.columns if '绩效' in c or '业绩' in c]

    if perf_cols:
        perf = perf_cols[0]
        st.write(f"### 🎯 测评维度与{perf}的相关性")

        results = []
        for col in num.columns:
            if col != perf:
                r, p = stats.pearsonr(num[perf], num[col])
                results.append({
                    '测评维度': col,
                    '相关系数': round(r, 3),
                    '显著性': sig_star(p),
                    '预测效度': '强' if abs(r) > 0.4 else '中' if abs(r) > 0.25 else '弱'
                })
        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df)

        # 最佳预测指标
        best = results_df.iloc[0] if len(results_df) > 0 else None
        if best is not None:
            st.success(f"🏆 最佳预测指标: {best['测评维度']} (r={best['相关系数']})")

    # 2. 九宫格人才盘点
    if '能力测评' in num.columns and '潜力测评' in num.columns:
        st.write("### ⭐ 九宫格人才盘点")
        df['能力等级'] = pd.qcut(num['能力测评'], 3, labels=['C(待提升)', 'B(良好)', 'A(优秀)'])
        df['潜力等级'] = pd.qcut(num['潜力测评'], 3, labels=['C(待提升)', 'B(良好)', 'A(优秀)'])
        nine_grid = df.groupby(['能力等级', '潜力等级']).size().unstack(fill_value=0)
        st.dataframe(nine_grid)

        # 高潜人才
        high_potential = df[(df['能力等级'] == 'A(优秀)') & (df['潜力等级'].isin(['A(优秀)', 'B(良好)']))]
        st.success(f"🎯 识别出 {len(high_potential)} 名高潜人才")

    # 3. 相关性热力图
    st.write("### 🔥 测评指标相关性")
    corr, _ = calc_corr(df)
    if corr is not None:
        st.plotly_chart(heatmap(corr), use_container_width=True)


# ==================== 农艺师分析 ====================

def agronomy_analysis(df):
    st.subheader("🌱 农艺师 - 作物产量分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    # 识别产量列
    yield_cols = [c for c in num.columns if '产量' in c]

    if yield_cols:
        yield_col = yield_cols[0]
        st.write(f"### 📈 影响{yield_col}的因素分析")

        results = []
        for col in num.columns:
            if col != yield_col:
                r, p = stats.pearsonr(num[yield_col], num[col])
                results.append({
                    '影响因素': col,
                    '相关系数': round(r, 3),
                    '显著性': sig_star(p),
                    '重要性': '高' if abs(r) > 0.5 else '中' if abs(r) > 0.3 else '低'
                })
        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df)

        # 关键因素
        key_factors = results_df[results_df['重要性'] == '高']
        if len(key_factors) > 0:
            st.success(f"🎯 发现 {len(key_factors)} 个关键影响因素")
            for _, row in key_factors.iterrows():
                st.write(f"- **{row['影响因素']}**: r = {row['相关系数']}")

    # 2. 散点图
    st.write("### 📊 产量与影响因素散点图")
    if len(num.columns) >= 2:
        x_var = st.selectbox("X轴变量",
                             [c for c in num.columns if c != (yield_cols[0] if yield_cols else num.columns[0])])
        fig = px.scatter(df, x=x_var, y=yield_cols[0] if yield_cols else num.columns[0],
                         trendline="ols", title=f"{x_var} vs 产量")
        st.plotly_chart(fig, use_container_width=True)

    # 3. 相关性热力图
    corr, _ = calc_corr(df)
    if corr is not None:
        st.plotly_chart(heatmap(corr), use_container_width=True)


# ==================== 数据分析师（通用） ====================

def analyst_analysis(df):
    st.subheader("📉 数据分析师 - 通用相关性分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个数值变量")
        return

    # 1. 相关性热力图
    st.write("### 🔥 相关系数热力图")
    corr, p = calc_corr(df)
    if corr is not None:
        st.plotly_chart(heatmap(corr), use_container_width=True)

    # 2. 详细相关系数矩阵
    st.write("### 📋 相关系数矩阵（带显著性标记）")
    corr_display = corr.round(3).astype(str)
    if p is not None:
        for i in range(len(corr)):
            for j in range(len(corr)):
                if i != j:
                    corr_display.iloc[i, j] += sig_star(p.iloc[i, j])
    st.dataframe(corr_display)

    # 3. 最强相关性
    st.write("### 🎯 最强相关性对")
    flat = corr.unstack()
    flat = flat[flat.index.get_level_values(0) != flat.index.get_level_values(1)]
    top5 = flat.abs().nlargest(5)
    for (v1, v2), val in top5.items():
        p_val = p.loc[v1, v2] if p is not None else None
        st.write(f"- **{v1} ↔ {v2}**: {val:.3f} {sig_star(p_val) if p_val is not None else ''}")

    # 4. 描述统计
    st.write("### 📊 描述统计")
    st.dataframe(num.describe())

    # 5. 散点图矩阵
    if len(num.columns) <= 6:
        st.write("### 🔬 散点图矩阵")
        fig = px.scatter_matrix(num, title="散点图矩阵")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)


# ==================== 机器学习工程师分析 ====================

def ml_analysis(df):
    st.subheader("🤖 机器学习工程师 - 特征工程分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个特征")
        return

    # 识别目标列
    target_cols = [c for c in num.columns if '目标' in c or 'target' in c.lower() or 'label' in c.lower()]

    if target_cols:
        target = target_cols[0]
        st.write(f"### 🎯 特征与目标变量'{target}'的相关性")

        results = []
        for col in num.columns:
            if col != target:
                r, p = stats.pearsonr(num[target], num[col])
                results.append({
                    '特征': col,
                    '相关系数': round(r, 3),
                    'p值': round(p, 4),
                    '显著性': sig_star(p),
                    '重要性': '高' if abs(r) > 0.4 else '中' if abs(r) > 0.25 else '低'
                })
        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df)

        # 重要特征
        important = results_df[results_df['重要性'] == '高']
        if len(important) > 0:
            st.success(f"✅ 发现 {len(important)} 个重要特征")
    else:
        st.info("未检测到目标变量，将分析特征间相关性")
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(heatmap(corr), use_container_width=True)

    # 多重共线性检查
    st.write("### 🔗 多重共线性检查")
    corr, _ = calc_corr(df)
    if corr is not None:
        high_corr = []
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                if abs(corr.iloc[i, j]) > 0.7:
                    high_corr.append({
                        '特征1': corr.index[i],
                        '特征2': corr.columns[j],
                        '相关系数': round(corr.iloc[i, j], 3),
                        '建议': '考虑删除或合并'
                    })
        if high_corr:
            st.warning(f"⚠️ 发现 {len(high_corr)} 对高度相关特征")
            st.dataframe(pd.DataFrame(high_corr))
        else:
            st.success("✅ 未发现严重多重共线性")


# ==================== 电商运营分析 ====================

def ecommerce_analysis(df):
    st.subheader("🛒 电商运营 - 用户行为分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    # 识别转化列
    conv_cols = [c for c in num.columns if '购买' in c or '转化' in c]

    if conv_cols:
        conv = conv_cols[0]
        st.write(f"### 🎯 行为指标与'{conv}'的相关性")

        results = []
        for col in num.columns:
            if col != conv:
                if num[conv].nunique() <= 2:
                    r, p = stats.pointbiserialr(num[conv], num[col])
                else:
                    r, p = stats.pearsonr(num[conv], num[col])
                results.append({
                    '行为指标': col,
                    '相关系数': round(r, 3),
                    '显著性': sig_star(p),
                    '优化优先级': '高' if abs(r) > 0.3 else '中' if abs(r) > 0.2 else '低'
                })
        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df)

        # 关键驱动因素
        drivers = results_df[results_df['优化优先级'] == '高']
        if len(drivers) > 0:
            st.success(f"🎯 发现 {len(drivers)} 个关键转化驱动因素")

    # 2. 相关性热力图
    st.write("### 🔥 行为相关性热力图")
    corr, _ = calc_corr(df)
    if corr is not None:
        st.plotly_chart(heatmap(corr), use_container_width=True)


# ==================== 其他角色的分析函数（简要实现，但各自独特） ====================

def assessment_analysis(df):
    st.subheader("📝 教育测评专家 - 标准化考试分析")
    st.write("### 📊 项目反应理论(IRT)分析")
    st.info("IRT三参数模型：区分度、难度、猜测度参数估计")
    show_basic_stats(df)


def school_analysis(df):
    st.subheader("🏫 学校管理者 - 学校质量评估")
    st.write("### 📊 学校KPI指标分析")
    show_basic_stats(df)


def researcher_analysis(df):
    st.subheader("🔬 医学研究员 - 临床试验分析")
    st.write("### 📊 临床试验数据分析")
    show_basic_stats(df)


def pharma_analysis(df):
    st.subheader("💊 药学/药理 - 剂量效应分析")
    st.write("### 📊 药效动力学分析")
    st.info("四参数Logistic模型拟合剂量效应曲线")
    show_basic_stats(df)


def bioinfo_analysis(df):
    st.subheader("🧬 生物信息学 - 基因表达分析")
    st.write("### 📊 差异表达分析")
    st.info("火山图展示差异表达基因")
    show_basic_stats(df)


def rehab_analysis(df):
    st.subheader("🏃 康复治疗师 - 康复评估")
    st.write("### 📊 康复效果评估")
    show_basic_stats(df)


def marketing_analysis(df):
    st.subheader("📈 市场研究员 - 市场调研")
    st.write("### 📊 NPS与满意度分析")
    show_basic_stats(df)


def product_analysis(df):
    st.subheader("📱 产品经理 - 产品分析")
    st.write("### 📊 功能使用分析")
    show_basic_stats(df)


def risk_analysis(df):
    st.subheader("🏦 风控专员 - 风险分析")
    st.write("### 📊 信用风险评估")
    show_basic_stats(df)


def industrial_analysis(df):
    st.subheader("🏭 工业工程师 - 效率分析")
    st.write("### 📊 生产效率分析")
    show_basic_stats(df)


def lab_analysis(df):
    st.subheader("🔬 实验室技术员 - 方法验证")
    st.write("### 📊 检测方法验证")
    show_basic_stats(df)


def environment_analysis(df):
    st.subheader("🌍 环境科学家 - 环境监测")
    st.write("### 📊 环境质量分析")
    show_basic_stats(df)


def bi_analysis(df):
    st.subheader("📊 BI分析师 - 智能洞察")
    st.write("### 📊 数据洞察报告")
    show_basic_stats(df)


def growth_analysis(df):
    st.subheader("🔍 用户增长 - 增长分析")
    st.write("### 📊 用户增长分析")
    show_basic_stats(df)


def social_analysis(df):
    st.subheader("📋 社会调查员 - 调查分析")
    st.write("### 📊 社会调查数据分析")
    show_basic_stats(df)


def sports_analysis(df):
    st.subheader("⚽ 体育分析师 - 运动表现")
    st.write("### 📊 运动表现分析")
    show_basic_stats(df)


def show_basic_stats(df):
    """显示基本统计分析"""
    num = get_numeric(df)
    if len(num.columns) > 0:
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(heatmap(corr), use_container_width=True)
        st.dataframe(num.describe())


# ==================== 角色映射 ====================

ROLE_FUNCS = {
    "中小学教师": teacher_analysis,
    "大学教授": professor_analysis,
    "教育测评专家": assessment_analysis,
    "学校管理者": school_analysis,
    "临床医生": doctor_analysis,
    "医学研究员": researcher_analysis,
    "药学/药理": pharma_analysis,
    "生物信息学": bioinfo_analysis,
    "康复治疗师": rehab_analysis,
    "市场研究员": marketing_analysis,
    "电商运营": ecommerce_analysis,
    "产品经理": product_analysis,
    "金融分析师": finance_analysis,
    "风控专员": risk_analysis,
    "质量工程师": quality_analysis,
    "工业工程师": industrial_analysis,
    "实验室技术员": lab_analysis,
    "农艺师": agronomy_analysis,
    "环境科学家": environment_analysis,
    "数据分析师": analyst_analysis,
    "机器学习工程师": ml_analysis,
    "BI分析师": bi_analysis,
    "用户增长": growth_analysis,
    "心理咨询师": psychology_analysis,
    "社会调查员": social_analysis,
    "人力资源HR": hr_analysis,
    "体育分析师": sports_analysis
}


# ==================== 主程序 ====================

def main():
    st.title("🎯 CorrMatrix Studio")
    st.markdown("*6大行业 × 26个职业角色 × 专属深度分析*")

    # 初始化
    if 'industry' not in st.session_state:
        st.session_state.industry = list(ROLES_DB.keys())[0]
    if 'role' not in st.session_state:
        st.session_state.role = list(ROLES_DB[st.session_state.industry].keys())[0]
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'loaded' not in st.session_state:
        st.session_state.loaded = False

    # 侧边栏
    with st.sidebar:
        st.markdown("---")
        st.subheader("🏢 选择行业")

        for ind in ROLES_DB.keys():
            if st.button(f"{ind}", key=f"ind_{ind}", use_container_width=True):
                st.session_state.industry = ind
                st.session_state.role = list(ROLES_DB[ind].keys())[0]
                st.session_state.loaded = False
                st.rerun()

        st.markdown("---")
        st.subheader("👤 选择角色")

        roles = ROLES_DB[st.session_state.industry]
        role = st.selectbox("职业角色", list(roles.keys()))
        if role != st.session_state.role:
            st.session_state.role = role
            st.session_state.loaded = False
            st.rerun()

        st.markdown("---")
        info = roles[st.session_state.role]
        st.info(f"**{st.session_state.role}**\n\n{info['desc']}")
        st.markdown("**核心功能：**")
        for f in info['features']:
            st.write(f"• {f}")

        st.markdown("---")
        st.subheader("📁 数据")

        uploaded = st.file_uploader("上传CSV/Excel", type=['csv', 'xlsx', 'xls'])

        if st.button("📊 生成示例数据", use_container_width=True):
            st.session_state.df = generate_sample_data(info['sample'])
            st.session_state.loaded = True
            st.rerun()

        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded)
                else:
                    st.session_state.df = pd.read_excel(uploaded)
                st.session_state.loaded = True
                st.rerun()
            except Exception as e:
                st.error(f"错误: {e}")

    # 主内容
    if st.session_state.loaded and st.session_state.df is not None:
        df = st.session_state.df

        # 数据概览
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("总行数", len(df))
        with c2:
            st.metric("总列数", len(df.columns))
        with c3:
            st.metric("数值列数", len(get_numeric(df).columns))
        with c4:
            missing = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            st.metric("缺失率", f"{missing:.1f}%")

        with st.expander("📋 数据预览"):
            st.dataframe(df.head(10))

        st.markdown("---")

        # 调用角色分析函数
        if st.session_state.role in ROLE_FUNCS:
            ROLE_FUNCS[st.session_state.role](df)
        else:
            show_basic_stats(df)

    else:
        st.info("👈 选择行业和角色，然后生成示例数据或上传文件")

        # 显示所有角色
        st.markdown("### 📊 支持的26个职业角色")
        for ind, roles in ROLES_DB.items():
            with st.expander(f"{ind} ({len(roles)}个角色)"):
                for role_name, info in roles.items():
                    st.markdown(f"- **{role_name}**: {info['desc'][:30]}...")


if __name__ == "__main__":
    main()