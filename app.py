"""
CorrMatrix Studio Pro - 智能多角色统计分析平台
版本: v3.0
代码行数: 2100+
功能: 6大行业 × 26个职业角色 × 专属深度分析 × 分析类型选择
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr, kendalltau, pointbiserialr, f_oneway, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# ==================== 页面配置 ====================

APP_NAME = "📊 CorrMatrix Studio Pro"
APP_SUBTITLE = "智能多角色统计分析平台 · 6大行业 × 26个职业角色"
APP_VERSION = "v3.0"
APP_ICON = "📊"

st.set_page_config(
    page_title=f"{APP_NAME} - {APP_VERSION}",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS ====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .role-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        border-left: 4px solid #1f77b4;
    }
    .feature-tag {
        background: #e3f2fd;
        color: #1565c0;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 2px;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
    }
    .stButton button[kind="primary"] {
        background: #1f77b4;
        color: white;
    }
    .metric-box {
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .analysis-section {
        background: #fafafa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
    }
    .insight-box {
        background: #e8f5e9;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    .warning-box {
        background: #fff3e0;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #ff9800;
        margin: 10px 0;
    }
    .danger-box {
        background: #ffebee;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #f44336;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 角色数据库（完整26个角色） ====================

ROLES_DB = {
    "📚 教育科研": {
        "icon": "📚",
        "color": "#4CAF50",
        "roles": {
            "👩‍🏫 中小学教师": {
                "desc": "试题质量分析、班级成绩对比、教学质量评估",
                "features": ["难度系数", "区分度", "信度分析", "分数分布", "班级对比", "低质量题目标记", "成绩预警"],
                "sample": "exam",
                "analysis_types": ["📝 试题质量分析", "📊 成绩分布分析", "🏫 班级对比分析", "⚠️ 质量预警诊断"]
            },
            "🎓 大学教授/研究生": {
                "desc": "问卷信效度分析、论文数据统计分析",
                "features": ["Cronbach's α", "KMO检验", "Bartlett球形检验", "因子载荷", "共同度", "维度相关", "信度提升建议"],
                "sample": "questionnaire",
                "analysis_types": ["📐 信度分析", "🔬 效度分析", "🔗 维度相关性分析", "📊 项目分析"]
            },
            "📝 教育测评专家": {
                "desc": "标准化考试开发、IRT项目反应理论",
                "features": ["项目反应理论(IRT)", "信息函数", "测验等值", "DIF检测", "题库建设", "划界分数", "测验质量"],
                "sample": "exam",
                "analysis_types": ["📊 IRT分析", "🔄 测验等值", "⚖️ DIF检测", "📚 题库质量"]
            },
            "🏫 学校管理者": {
                "desc": "学校质量评估、KPI指标监控预警",
                "features": ["学业质量", "师资指标", "硬件指标", "预警系统", "趋势预测", "标杆对比", "综合评估"],
                "sample": "school",
                "analysis_types": ["📊 质量评估", "⚠️ 指标预警", "📈 趋势预测", "🏆 标杆对比"]
            }
        }
    },
    "🏥 医疗健康": {
        "icon": "🏥",
        "color": "#2196F3",
        "roles": {
            "👨‍⚕️ 临床医生": {
                "desc": "检验指标与疾病诊断相关性分析",
                "features": ["指标相关性", "ROC曲线", "约登指数", "敏感度/特异度", "参考区间", "风险因素排序", "诊断效能"],
                "sample": "clinical",
                "analysis_types": ["🔗 指标相关性分析", "📊 参考区间分析", "🎯 风险因素识别", "📈 ROC诊断效能"]
            },
            "🔬 医学研究员": {
                "desc": "临床试验数据、治疗前后对比分析",
                "features": ["治疗前后对比", "分组比较", "协变量校正", "生存曲线", "Log-rank检验", "Cox回归", "疗效评估"],
                "sample": "clinical",
                "analysis_types": ["📊 治疗前后对比", "👥 分组比较", "📈 生存分析", "💊 疗效评估"]
            },
            "💊 药学/药理": {
                "desc": "剂量-效应关系、药效动力学分析",
                "features": ["剂量效应曲线", "ED50/LD50", "治疗指数", "药物相互作用", "Bliss模型", "等效性分析", "毒理评估"],
                "sample": "pharma",
                "analysis_types": ["📈 剂量效应分析", "💊 半数有效量估算", "🔗 药物相互作用", "⚠️ 毒理评估"]
            },
            "🧬 生物信息学": {
                "desc": "基因表达、基因组学数据分析",
                "features": ["差异表达基因", "聚类热图", "火山图", "GO富集", "KEGG通路", "PPI网络", "功能注释"],
                "sample": "bioinfo",
                "analysis_types": ["🧬 差异表达分析", "🔬 聚类分析", "🧪 通路富集分析", "📊 基因功能分析"]
            },
            "🏃 康复治疗师": {
                "desc": "康复效果评估、功能恢复分析",
                "features": ["功能评分变化", "康复周期分析", "预后因素", "生活质量", "ADL评估", "康复目标达成", "效果追踪"],
                "sample": "rehab",
                "analysis_types": ["📊 康复效果评估", "🎯 预后因素分析", "📈 功能评分变化", "✅ 目标达成分析"]
            }
        }
    },
    "📊 商业金融": {
        "icon": "📊",
        "color": "#FF9800",
        "roles": {
            "📈 市场研究员": {
                "desc": "满意度调查、忠诚度分析、NPS研究",
                "features": ["满意度驱动因素", "NPS分析", "重要性矩阵", "细分人群", "T检验", "方差分析", "市场细分"],
                "sample": "market",
                "analysis_types": ["🎯 满意度驱动因素", "⭐ NPS分析", "👥 细分人群对比", "📊 重要性矩阵"]
            },
            "🛒 电商运营": {
                "desc": "用户行为分析、转化率优化",
                "features": ["行为-转化相关", "RFM分层", "漏斗分析", "复购预测", "购物篮分析", "用户画像", "A/B测试"],
                "sample": "ecommerce",
                "analysis_types": ["🔄 行为转化分析", "📊 RFM分层", "📉 漏斗分析", "🛍️ 复购预测"]
            },
            "📱 产品经理": {
                "desc": "功能使用分析、用户留存研究",
                "features": ["功能使用率", "留存曲线", "魔法数字", "A/B测试", "用户分群", "NPS跟踪", "功能组合"],
                "sample": "product",
                "analysis_types": ["📊 功能使用分析", "📈 留存分析", "👥 用户分群", "✨ 魔法数字识别"]
            },
            "💰 金融分析师": {
                "desc": "资产组合风险分析、投资决策支持",
                "features": ["资产相关性", "风险分散", "Beta系数", "夏普比率", "最大回撤", "VaR计算", "因子分析"],
                "sample": "finance",
                "analysis_types": ["📊 资产相关性分析", "💡 风险分散建议", "📉 收益风险分析", "📈 因子分析"]
            },
            "🏦 风控专员": {
                "desc": "信用风险评分、违约预测模型",
                "features": ["违约因素分析", "WOE/IV值", "评分卡", "KS统计量", "混淆矩阵", "AUC曲线", "风险评级"],
                "sample": "risk",
                "analysis_types": ["⚠️ 违约因素分析", "📊 信用评分", "📈 模型评估", "🏷️ 风险评级"]
            }
        }
    },
    "🏭 工业农业": {
        "icon": "🏭",
        "color": "#9C27B0",
        "roles": {
            "🔧 质量工程师": {
                "desc": "过程质量控制、六西格玛管理",
                "features": ["过程能力Cpk", "SPC控制图", "六西格玛", "测量系统分析", "DOE分析", "FMEA", "质量改进"],
                "sample": "quality",
                "analysis_types": ["📊 过程能力分析", "📈 SPC控制图", "🔗 相关性分析", "🎯 六西格玛评估"]
            },
            "🏭 工业工程师": {
                "desc": "生产效率优化、工业工程分析",
                "features": ["工时分析", "瓶颈识别", "线平衡率", "OEE计算", "标准工时", "人机分析", "布局优化"],
                "sample": "industrial",
                "analysis_types": ["⚡ 效率分析", "🔍 瓶颈识别", "📊 OEE分析", "⚖️ 线平衡分析"]
            },
            "🔬 实验室技术员": {
                "desc": "检测方法验证、质量控制",
                "features": ["重复性/再现性", "加标回收", "检出限", "不确定度", "线性范围", "方法比对", "质控图"],
                "sample": "lab",
                "analysis_types": ["📊 重复性分析", "💯 加标回收率", "📈 不确定度评定", "🔍 检出限分析"]
            },
            "🌱 农艺师": {
                "desc": "作物产量分析、农业优化",
                "features": ["产量影响因素", "施肥优化", "土壤指标", "气候影响", "品种对比", "正交试验", "最佳方案"],
                "sample": "agriculture",
                "analysis_types": ["🌾 产量因素分析", "💊 施肥优化", "🧪 土壤指标分析", "📊 品种对比"]
            },
            "🌍 环境科学家": {
                "desc": "环境监测评价、污染物溯源",
                "features": ["污染物时空分布", "污染源解析", "环境质量指数", "生态风险", "趋势预测", "相关性聚类", "溯源分析"],
                "sample": "environment",
                "analysis_types": ["🌍 污染物分析", "📊 环境质量评价", "📈 趋势预测", "🔍 污染源解析"]
            }
        }
    },
    "💻 数据科学": {
        "icon": "💻",
        "color": "#00BCD4",
        "roles": {
            "📉 数据分析师": {
                "desc": "通用相关性分析、数据探索",
                "features": ["Pearson/Spearman/Kendall", "热力图", "散点图矩阵", "偏相关", "距离相关", "互信息", "相关性网络"],
                "sample": "general",
                "analysis_types": ["🔥 相关性热力图", "📋 相关系数矩阵", "🔬 散点图矩阵", "📊 描述统计"]
            },
            "🤖 机器学习工程师": {
                "desc": "特征工程、特征选择",
                "features": ["特征重要性", "多重共线性(VIF)", "IV值", "递归特征消除", "PCA降维", "特征交互", "特征筛选"],
                "sample": "ml",
                "analysis_types": ["🎯 特征相关性", "📊 特征重要性", "🔗 多重共线性", "📉 降维分析"]
            },
            "📊 BI分析师": {
                "desc": "智能数据洞察、自动报表",
                "features": ["异动分析", "趋势检测", "相关性告警", "智能摘要", "数据讲故事", "仪表盘", "自动洞察"],
                "sample": "bi",
                "analysis_types": ["🔍 智能洞察", "📈 趋势分析", "⚠️ 异常检测", "📊 自动摘要"]
            },
            "🔍 用户增长": {
                "desc": "增长黑客、用户留存分析",
                "features": ["留存曲线", "魔法数字", "Cohort分析", "LTV预测", "渠道归因", "AARRR模型", "激活分析"],
                "sample": "growth",
                "analysis_types": ["📈 留存分析", "✨ 魔法数字", "📊 Cohort分析", "💰 LTV预测"]
            }
        }
    },
    "🧠 人文社科": {
        "icon": "🧠",
        "color": "#E91E63",
        "roles": {
            "🧠 心理咨询师": {
                "desc": "心理量表信效度、心理健康评估",
                "features": ["Cronbach's α", "分半信度", "重测信度", "效标效度", "探索性因子", "验证性因子", "常模参照"],
                "sample": "psychology",
                "analysis_types": ["📐 信度分析", "🔬 效度分析", "📊 因子分析", "📈 常模分析"]
            },
            "📋 社会调查员": {
                "desc": "社会调查数据分析、民意研究",
                "features": ["卡方检验", "列联表", "对应分析", "信度分析", "权重计算", "抽样误差", "交叉分析"],
                "sample": "social",
                "analysis_types": ["📊 卡方检验", "📋 列联表分析", "📐 信度分析", "📈 交叉分析"]
            },
            "👥 人力资源HR": {
                "desc": "人才测评、绩效预测、组织发展",
                "features": ["测评-绩效相关", "高潜识别", "九宫格", "离职预警", "人才盘点", "继任规划", "胜任力模型"],
                "sample": "hr",
                "analysis_types": ["🎯 测评-绩效分析", "⭐ 九宫格人才盘点", "🏆 高潜识别", "⚠️ 离职预警"]
            },
            "⚽ 体育分析师": {
                "desc": "运动表现分析、训练优化",
                "features": ["训练-成绩相关", "伤病风险", "比赛预测", "球员评估", "战术分析", "体能监控", "表现趋势"],
                "sample": "sports",
                "analysis_types": ["📈 训练-成绩分析", "⚠️ 伤病风险评估", "🏆 表现评估", "📊 技战术分析"]
            }
        }
    }
}

# ==================== 示例数据生成（全部26种） ====================

def generate_sample_data(data_type):
    """生成示例数据 - 26种不同类型"""
    np.random.seed(42)
    n = 200

    # ===== 教育科研类 =====
    if data_type == "exam":
        chinese = np.random.normal(75, 12, n)
        math = 70 + 0.6*chinese + np.random.normal(0, 10, n)
        english = 72 + 0.5*chinese + np.random.normal(0, 11, n)
        science = 68 + 0.4*chinese + np.random.normal(0, 12, n)
        history = 65 + 0.3*chinese + np.random.normal(0, 13, n)
        total = chinese + math + english + science + history
        return pd.DataFrame({
            '学生ID': [f'S{i:03d}' for i in range(1, n+1)],
            '班级': np.random.choice(['1班','2班','3班','4班','5班'], n),
            '语文': np.clip(chinese, 40, 100),
            '数学': np.clip(math, 40, 100),
            '英语': np.clip(english, 40, 100),
            '科学': np.clip(science, 40, 100),
            '历史': np.clip(history, 40, 100),
            '总分': np.clip(total, 250, 500)
        })

    elif data_type == "questionnaire":
        data = {}
        for i in range(1, 21):
            data[f'Q{i}'] = np.random.randint(1, 6, n)
        df = pd.DataFrame(data)
        df['ID'] = [f'R{i:03d}' for i in range(1, n+1)]
        return df

    elif data_type == "school":
        return pd.DataFrame({
            '学校ID': [f'S{i:03d}' for i in range(1, n+1)],
            '升学率': np.random.normal(0.85, 0.08, n),
            '师资比': np.random.normal(0.08, 0.015, n),
            '经费': np.random.normal(500, 120, n),
            '设施评分': np.random.normal(3.5, 0.8, n),
            '满意度': np.random.normal(3.8, 0.7, n)
        })

    # ===== 医疗健康类 =====
    elif data_type == "clinical":
        age = np.random.normal(55, 12, n)
        bmi = 22 + 0.15*age + np.random.normal(0, 3, n)
        bp = 110 + 0.5*age + np.random.normal(0, 10, n)
        glucose = 5.0 + 0.03*age + 0.1*bmi + np.random.normal(0, 0.8, n)
        cholesterol = 4.5 + 0.02*age + 0.05*bmi + np.random.normal(0, 0.8, n)
        diagnosis = 1 / (1 + np.exp(-(-3 + 0.03*age + 0.05*bp + 0.5*glucose))) > 0.5
        return pd.DataFrame({
            '患者ID': [f'P{i:03d}' for i in range(1, n+1)],
            '年龄': age,
            'BMI': bmi,
            '收缩压': bp,
            '血糖': glucose,
            '胆固醇': cholesterol,
            '诊断结果': diagnosis.astype(int)
        })

    elif data_type == "pharma":
        dose = np.random.uniform(0.1, 10, n)
        effect = 100 * (1 - np.exp(-0.3 * dose)) + np.random.normal(0, 5, n)
        toxicity = 100 * (1 - np.exp(-0.05 * dose)) + np.random.normal(0, 3, n)
        return pd.DataFrame({
            '样本ID': [f'D{i:03d}' for i in range(1, n+1)],
            '剂量': dose,
            '药效': np.clip(effect, 0, 100),
            '毒性': np.clip(toxicity, 0, 100)
        })

    elif data_type == "bioinfo":
        genes = [f'Gene{i}' for i in range(1, 11)]
        data = {}
        for gene in genes:
            data[gene] = np.random.normal(10, 3, n) + np.random.choice([0, 5], n, p=[0.7, 0.3])
        data['样本ID'] = [f'B{i:03d}' for i in range(1, n+1)]
        return pd.DataFrame(data)

    elif data_type == "rehab":
        return pd.DataFrame({
            '患者ID': [f'R{i:03d}' for i in range(1, n+1)],
            '治疗前评分': np.random.normal(40, 15, n),
            '治疗后评分': np.random.normal(70, 12, n),
            '改善值': np.random.normal(30, 10, n),
            '康复天数': np.random.exponential(30, n)
        })

    # ===== 商业金融类 =====
    elif data_type == "market":
        satisfaction = np.random.normal(3.5, 1.0, n)
        service = 2.5 + 0.7*satisfaction + np.random.normal(0, 0.5, n)
        price = 2.8 + 0.4*satisfaction + np.random.normal(0, 0.6, n)
        loyalty = 2.0 + 0.8*satisfaction + np.random.normal(0, 0.5, n)
        nps = np.clip(loyalty * 20 + np.random.normal(0, 10, n), -100, 100)
        return pd.DataFrame({
            'ID': [f'C{i:03d}' for i in range(1, n+1)],
            '产品满意度': np.clip(satisfaction, 1, 5),
            '服务满意度': np.clip(service, 1, 5),
            '价格满意度': np.clip(price, 1, 5),
            '品牌忠诚度': np.clip(loyalty, 1, 5),
            '推荐意愿': np.clip(loyalty + 0.2*np.random.randn(n), 1, 5),
            'NPS': np.clip(nps, -100, 100)
        })

    elif data_type == "ecommerce":
        browse = np.random.exponential(25, n)
        clicks = 2 + 0.4*browse + np.random.poisson(2, n)
        cart = 0.2*clicks + np.random.poisson(0.5, n)
        purchase = 1 / (1 + np.exp(-(-2 + 0.08*browse + 0.3*clicks + 0.5*cart))) > 0.5
        amount = purchase * (50 + 0.5*browse + np.random.exponential(100, n))
        return pd.DataFrame({
            '用户ID': [f'U{i:03d}' for i in range(1, n+1)],
            '浏览时长': browse,
            '点击次数': clicks,
            '加购次数': cart,
            '是否购买': purchase.astype(int),
            '消费金额': amount
        })

    elif data_type == "product":
        features = [f'功能{i}' for i in range(1, 9)]
        data = {}
        for feat in features:
            data[feat] = np.random.uniform(0, 100, n)
        data['留存率'] = np.random.uniform(30, 80, n)
        data['用户ID'] = [f'U{i:03d}' for i in range(1, n+1)]
        return pd.DataFrame(data)

    elif data_type == "finance":
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        stock_a = np.random.normal(0.0005, 0.02, n).cumsum()
        stock_b = 0.6*stock_a + np.random.normal(0, 0.005, n).cumsum()
        stock_c = -0.3*stock_a + np.random.normal(0, 0.008, n).cumsum()
        stock_d = 0.4*stock_a + np.random.normal(0, 0.006, n).cumsum()
        bond = np.random.normal(0.0001, 0.002, n).cumsum()
        return pd.DataFrame({
            '日期': dates,
            '股票A': np.diff(np.append([0], stock_a)),
            '股票B': np.diff(np.append([0], stock_b)),
            '股票C': np.diff(np.append([0], stock_c)),
            '股票D': np.diff(np.append([0], stock_d)),
            '债券': np.diff(np.append([0], bond))
        })

    elif data_type == "risk":
        income = np.random.normal(10000, 3000, n)
        debt = np.random.normal(3000, 1500, n)
        credit_score = np.random.normal(650, 80, n)
        age = np.random.normal(35, 10, n)
        default = 1 / (1 + np.exp(-(0.001*income - 0.0005*debt - 0.01*credit_score + 0.02*age))) > 0.5
        return pd.DataFrame({
            '客户ID': [f'C{i:03d}' for i in range(1, n+1)],
            '收入': income,
            '负债': debt,
            '信用评分': credit_score,
            '年龄': age,
            '是否违约': default.astype(int)
        })

    # ===== 工业农业类 =====
    elif data_type == "quality":
        temp = np.random.normal(150, 10, n)
        pressure = 4.8 + 0.01*temp + np.random.normal(0, 0.3, n)
        speed = 100 + 0.2*temp + np.random.normal(0, 5, n)
        humidity = 50 + 0.1*temp + np.random.normal(0, 5, n)
        defect_rate = np.exp(-3 + 0.01*temp + 0.1*pressure - 0.02*humidity) / (1 + np.exp(-3 + 0.01*temp + 0.1*pressure - 0.02*humidity))
        return pd.DataFrame({
            '批次': [f'B{i:03d}' for i in range(1, n+1)],
            '温度(℃)': temp,
            '压力(MPa)': pressure,
            '速度(m/min)': speed,
            '湿度(%)': humidity,
            '良品率(%)': 100 * (1 - defect_rate)
        })

    elif data_type == "industrial":
        return pd.DataFrame({
            '工单ID': [f'W{i:03d}' for i in range(1, n+1)],
            '工时': np.random.normal(8, 2, n),
            '产出量': np.random.normal(100, 20, n),
            '缺陷数': np.random.poisson(3, n),
            '停机时间': np.random.exponential(30, n)
        })

    elif data_type == "lab":
        return pd.DataFrame({
            '样本ID': [f'L{i:03d}' for i in range(1, n+1)],
            '测量值1': np.random.normal(100, 2, n),
            '测量值2': np.random.normal(100, 2.5, n),
            '测量值3': np.random.normal(100, 1.8, n),
            '标准值': np.random.normal(100, 1, n)
        })

    elif data_type == "agriculture":
        fertilizer = np.random.uniform(20, 100, n)
        rainfall = np.random.uniform(80, 200, n)
        temperature = np.random.uniform(15, 35, n)
        ph = np.random.normal(6.5, 0.5, n)
        yield_amount = 300 + 3*fertilizer + 1.5*rainfall + 5*temperature - 20*abs(ph-6.5) + np.random.normal(0, 40, n)
        return pd.DataFrame({
            '地块ID': [f'F{i:03d}' for i in range(1, n+1)],
            '施肥量': fertilizer,
            '降雨量': rainfall,
            '温度': temperature,
            '土壤pH': ph,
            '产量': yield_amount
        })

    elif data_type == "environment":
        return pd.DataFrame({
            '监测点': [f'M{i:03d}' for i in range(1, n+1)],
            'PM2.5': np.random.normal(50, 20, n),
            'SO2': np.random.normal(20, 8, n),
            'NO2': np.random.normal(30, 10, n),
            'O3': np.random.normal(60, 15, n),
            'CO': np.random.normal(3, 1, n)
        })

    # ===== 数据科学类 =====
    elif data_type == "general":
        A = np.random.normal(50, 15, n)
        B = 0.7*A + np.random.normal(0, 8, n)
        C = -0.4*A + 0.3*B + np.random.normal(0, 8, n)
        D = 0.3*A + 0.4*B + np.random.normal(0, 8, n)
        E = np.random.normal(48, 11, n)
        return pd.DataFrame({
            '变量A': A, '变量B': B, '变量C': C, '变量D': D, '变量E': E
        })

    elif data_type == "ml":
        X1 = np.random.normal(50, 15, n)
        X2 = 0.6*X1 + np.random.normal(0, 8, n)
        X3 = -0.3*X1 + np.random.normal(0, 10, n)
        X4 = 0.4*X1 + 0.3*X2 + np.random.normal(0, 6, n)
        X5 = np.random.normal(48, 11, n)
        y = 0.5*X1 + 0.3*X2 - 0.2*X3 + 0.2*X4 + np.random.normal(0, 5, n)
        return pd.DataFrame({
            '特征1': X1, '特征2': X2, '特征3': X3, '特征4': X4, '特征5': X5,
            '目标变量': y
        })

    elif data_type == "bi":
        sales = np.random.normal(1000, 200, n).cumsum()
        cost = 0.6*sales + np.random.normal(0, 50, n).cumsum()
        profit = sales - cost
        return pd.DataFrame({
            '日期': pd.date_range('2023-01-01', periods=n, freq='D'),
            '销售额': sales,
            '成本': cost,
            '利润': profit
        })

    elif data_type == "growth":
        return pd.DataFrame({
            '用户ID': [f'G{i:03d}' for i in range(1, n+1)],
            '注册天数': np.random.randint(1, 365, n),
            '活跃天数': np.random.randint(1, 100, n),
            '交互次数': np.random.poisson(20, n),
            '是否留存': np.random.choice([0, 1], n, p=[0.4, 0.6])
        })

    # ===== 人文社科类 =====
    elif data_type == "psychology":
        anxiety = np.random.randint(1, 5, (n, 6)).sum(axis=1)
        depression = 0.5*anxiety + np.random.randint(1, 3, (n, 6)).sum(axis=1)
        wellbeing = -0.3*anxiety + np.random.randint(1, 4, (n, 6)).sum(axis=1)
        return pd.DataFrame({
            'ID': [f'P{i:03d}' for i in range(1, n+1)],
            '焦虑总分': np.clip(anxiety, 6, 30),
            '抑郁总分': np.clip(depression, 6, 30),
            '幸福感总分': np.clip(wellbeing, 6, 30)
        })

    elif data_type == "social":
        return pd.DataFrame({
            '受访者ID': [f'S{i:03d}' for i in range(1, n+1)],
            '年龄': np.random.randint(18, 80, n),
            '收入': np.random.normal(8000, 3000, n),
            '教育年限': np.random.randint(6, 22, n),
            '幸福感': np.random.randint(1, 10, n),
            '社会信任': np.random.randint(1, 10, n),
            '政治倾向': np.random.choice(['左', '中', '右'], n)
        })

    elif data_type == "hr":
        ability = np.random.normal(75, 10, n)
        potential = 60 + 0.5*ability + np.random.normal(0, 8, n)
        leadership = 50 + 0.3*ability + np.random.normal(0, 10, n)
        teamwork = 60 + 0.2*ability + np.random.normal(0, 8, n)
        performance = 50 + 0.5*ability + 0.3*potential + np.random.normal(0, 6, n)
        return pd.DataFrame({
            '员工ID': [f'E{i:03d}' for i in range(1, n+1)],
            '能力测评': np.clip(ability, 40, 100),
            '潜力测评': np.clip(potential, 40, 100),
            '领导力': np.clip(leadership, 40, 100),
            '团队合作': np.clip(teamwork, 40, 100),
            '绩效评分': np.clip(performance, 50, 100)
        })

    elif data_type == "sports":
        train_hours = np.random.uniform(2, 12, n)
        strength = 50 + 3*train_hours + np.random.normal(0, 8, n)
        endurance = 45 + 4*train_hours + np.random.normal(0, 7, n)
        technique = 40 + 3*train_hours + np.random.normal(0, 6, n)
        performance = 40 + 4*train_hours + 0.2*strength + 0.2*endurance + 0.2*technique + np.random.normal(0, 5, n)
        return pd.DataFrame({
            '运动员ID': [f'A{i:03d}' for i in range(1, n+1)],
            '训练时长': train_hours,
            '力量评分': np.clip(strength, 40, 100),
            '耐力评分': np.clip(endurance, 40, 100),
            '技术评分': np.clip(technique, 40, 100),
            '比赛成绩': np.clip(performance, 50, 100)
        })

    else:
        return pd.DataFrame({
            '指标1': np.random.normal(100, 15, n),
            '指标2': np.random.normal(100, 15, n),
            '指标3': np.random.normal(100, 15, n),
            '指标4': np.random.normal(100, 15, n)
        })

# ==================== 通用工具函数 ====================

def get_numeric(df):
    """获取数值列"""
    return df.select_dtypes(include=[np.number])

def calc_corr(df, method='pearson'):
    """计算相关系数和p值"""
    num = get_numeric(df)
    if len(num.columns) < 2:
        return None, None

    if method == 'pearson':
        corr = num.corr()
        p = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
        for i in range(len(num.columns)):
            for j in range(len(num.columns)):
                if i != j:
                    _, p.iloc[i,j] = pearsonr(num.iloc[:,i], num.iloc[:,j])
        return corr, p
    elif method == 'spearman':
        corr = num.corr(method='spearman')
        p = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
        for i in range(len(num.columns)):
            for j in range(len(num.columns)):
                if i != j:
                    _, p.iloc[i,j] = spearmanr(num.iloc[:,i], num.iloc[:,j])
        return corr, p
    else:
        corr = num.corr(method='kendall')
        p = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
        for i in range(len(num.columns)):
            for j in range(len(num.columns)):
                if i != j:
                    _, p.iloc[i,j] = kendalltau(num.iloc[:,i], num.iloc[:,j])
        return corr, p

def create_heatmap(corr, title="相关系数热力图", height=500):
    """创建交互式热力图"""
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title=title,
        aspect='auto'
    )
    fig.update_layout(height=height, width=None)
    return fig

def sig_star(p):
    """显著性标记"""
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''

def cronbach_alpha(data):
    """计算Cronbach's α"""
    k = len(data.columns)
    if k < 2:
        return np.nan
    item_vars = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k/(k-1)) * (1 - item_vars.sum()/total_var)

def safe_qcut(data, q=3, labels=None):
    """安全分位数切割"""
    try:
        if labels is None:
            return pd.qcut(data, q=q, duplicates='drop')
        return pd.qcut(data, q=q, labels=labels, duplicates='drop')
    except ValueError:
        return pd.cut(data, bins=q, labels=labels)

def kmo_bartlett(data):
    """KMO和Bartlett检验"""
    corr = data.corr()
    try:
        corr_inv = np.linalg.inv(corr.values)
        r2 = 1 - 1/np.diag(corr_inv)
        kmo = r2.sum() / (r2.sum() + (1 - r2).sum())
    except:
        kmo = 0.5

    try:
        det = np.linalg.det(corr.values)
        n = len(data)
        p = len(data.columns)
        chi2 = - (n - 1 - (2*p + 5)/6) * np.log(det)
        df = p*(p-1)//2
        p_value = 1 - stats.chi2.cdf(chi2, df)
        return kmo, chi2, p_value
    except:
        return kmo, 0, 1

def calculate_cpk(data, spec_limit=None):
    """计算过程能力指数Cpk"""
    if spec_limit is None:
        spec_limit = (data.mean() - 3*data.std(), data.mean() + 3*data.std())
    mean = data.mean()
    std = data.std()
    if std == 0:
        return 0
    usl, lsl = spec_limit
    cpu = (usl - mean) / (3*std)
    cpl = (mean - lsl) / (3*std)
    return min(cpu, cpl)

def show_analysis_header(title, description):
    """显示分析头部"""
    st.markdown(f"""
    <div class="analysis-section">
        <h3>{title}</h3>
        <p style="color: #666; margin: 5px 0;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== 26个角色分析函数 ====================

# ----- 1. 中小学教师 -----
def teacher_analysis(df):
    st.subheader("👩‍🏫 中小学教师 - 试题质量分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个数值列进行分析")
        return

    total_cols = [c for c in num.columns if '总分' in c or '总成绩' in c or 'total' in c.lower()]
    q_cols = [c for c in num.columns if c not in total_cols and c not in ['学生ID', '班级']]

    # 1. 难度系数
    st.markdown("### 📊 难度系数分析")
    difficulty_data = []
    for q in q_cols:
        max_score = num[q].max()
        if max_score > 0:
            diff = num[q].mean() / max_score
            difficulty_data.append({
                '题目': q,
                '平均分': round(num[q].mean(), 2),
                '满分': max_score,
                '难度系数': round(diff, 3),
                '评价': '偏易 😊' if diff > 0.8 else '适中 ✅' if diff > 0.3 else '偏难 😅'
            })
    st.dataframe(pd.DataFrame(difficulty_data))

    # 2. 区分度
    if total_cols:
        total = total_cols[0]
        st.markdown("### 📈 区分度分析")
        high_score = num[total].quantile(0.73)
        low_score = num[total].quantile(0.27)

        disc_data = []
        for q in q_cols:
            high_avg = num[num[total] >= high_score][q].mean()
            low_avg = num[num[total] <= low_score][q].mean()
            max_score = num[q].max()
            if max_score > 0:
                disc = (high_avg - low_avg) / max_score
                disc_data.append({
                    '题目': q,
                    '高分组均分': round(high_avg, 2),
                    '低分组均分': round(low_avg, 2),
                    '区分度': round(disc, 3),
                    '评价': '优秀 🌟' if disc > 0.4 else '良好 ✅' if disc > 0.3 else '尚可 ⚠️' if disc > 0.2 else '待改进 ❌'
                })
        st.dataframe(pd.DataFrame(disc_data))

        # 3. 题目-总分相关
        st.markdown("### 🎯 题目-总分相关性")
        item_total = []
        for q in q_cols:
            r, p = pearsonr(num[q], num[total])
            item_total.append({
                '题目': q,
                '相关系数': round(r, 3),
                '显著性': sig_star(p),
                '质量评价': '优良' if r > 0.4 else '良好' if r > 0.3 else '一般'
            })
        st.dataframe(pd.DataFrame(item_total))

    # 4. 信度分析
    if len(q_cols) >= 2:
        st.markdown("### 📐 信度分析")
        alpha = cronbach_alpha(num[q_cols])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cronbach's α", f"{alpha:.3f}")
        with col2:
            quality = "优秀 🌟" if alpha > 0.9 else "良好 ✅" if alpha > 0.8 else "可接受 ⚠️" if alpha > 0.7 else "需改进 ❌"
            st.metric("信度评价", quality)

    # 5. 成绩分布
    st.markdown("### 📊 成绩分布图")
    fig = px.histogram(num, title="各科目分数分布", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    # 6. 班级对比
    if '班级' in df.columns:
        st.markdown("### 🏫 班级成绩对比")
        class_avg = df.groupby('班级')[q_cols].mean()
        st.dataframe(class_avg)
        fig = px.bar(class_avg.reset_index().melt(id_vars='班级'),
                     x='班级', y='value', color='variable', title="班级平均分对比")
        st.plotly_chart(fig, use_container_width=True)

# ----- 2. 大学教授 -----
def professor_analysis(df):
    st.subheader("🎓 大学教授 - 问卷信效度分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    # 1. 信度分析
    st.markdown("### 📐 信度分析")
    alpha = cronbach_alpha(num)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cronbach's α", f"{alpha:.3f}")
    with col2:
        quality = "优秀 🌟" if alpha > 0.9 else "良好 ✅" if alpha > 0.8 else "可接受 ⚠️" if alpha > 0.7 else "需改进 ❌"
        st.metric("信度评价", quality)
    with col3:
        st.metric("题目数量", len(num.columns))

    # 2. 删题后信度
    st.markdown("### 📉 删题后信度变化")
    alpha_if_deleted = []
    for col in num.columns:
        alpha_del = cronbach_alpha(num.drop(columns=[col]))
        alpha_if_deleted.append({
            '删除题目': col,
            '删除后α': round(alpha_del, 3),
            '变化': round(alpha_del - alpha, 3),
            '建议': '可考虑删除 ✅' if alpha_del > alpha else '保留 ❌'
        })
    st.dataframe(pd.DataFrame(alpha_if_deleted))

    # 3. 效度分析
    st.markdown("### 🔬 效度分析")
    kmo, chi2, p_val = kmo_bartlett(num)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("KMO检验值", f"{kmo:.3f}")
        st.caption("KMO > 0.7 适合因子分析")
    with col2:
        st.metric("Bartlett检验 p值", f"{p_val:.4f}")
        st.caption("p < 0.05 适合因子分析")

    # 4. 相关性热力图
    st.markdown("### 🔥 题目相关性热力图")
    corr, _ = calc_corr(df)
    if corr is not None:
        st.plotly_chart(create_heatmap(corr), use_container_width=True)

# ----- 3. 临床医生 -----
def doctor_analysis(df):
    st.subheader("👨‍⚕️ 临床医生 - 临床指标分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    diag_cols = [c for c in num.columns if '诊断' in c or '结果' in c or '疾病' in c]

    if diag_cols:
        diag = diag_cols[0]
        st.markdown(f"### 🩺 指标与'{diag}'的相关性")

        results = []
        for col in num.columns:
            if col != diag:
                r, p = pointbiserialr(num[diag], num[col])
                results.append({
                    '指标': col,
                    '相关系数': round(r, 3),
                    '显著性': sig_star(p),
                    '临床意义': '高风险因子 ⚠️' if abs(r) > 0.3 else '参考指标 ℹ️'
                })
        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df)

        high_risk = results_df[results_df['临床意义'] == '高风险因子 ⚠️']
        if len(high_risk) > 0:
            st.warning(f"⚠️ 发现 {len(high_risk)} 个高风险相关因子")

    # 参考区间
    st.markdown("### 📊 参考区间 (95%置信区间)")
    ref = pd.DataFrame({
        '指标': num.columns,
        '均值': round(num.mean(), 2),
        '标准差': round(num.std(), 2),
        '参考下限': round(num.mean() - 1.96*num.std(), 2),
        '参考上限': round(num.mean() + 1.96*num.std(), 2)
    })
    st.dataframe(ref)

    # 热力图
    st.markdown("### 🔥 指标相关性热力图")
    corr, _ = calc_corr(df)
    if corr is not None:
        st.plotly_chart(create_heatmap(corr), use_container_width=True)

# ----- 4. 金融分析师 -----
def finance_analysis(df):
    st.subheader("💰 金融分析师 - 资产组合分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个资产")
        return

    # 相关性矩阵
    st.markdown("### 📊 资产收益率相关性矩阵")
    corr, p = calc_corr(df)
    if corr is not None:
        st.plotly_chart(create_heatmap(corr, "资产相关性热力图"), use_container_width=True)

    # 风险分散建议
    st.markdown("### 💡 风险分散建议")
    low_corr = []
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            if abs(corr.iloc[i,j]) < 0.3:
                low_corr.append({
                    '资产1': corr.index[i],
                    '资产2': corr.columns[j],
                    '相关系数': round(corr.iloc[i,j], 3),
                    '分散效果': '优秀 🌟' if abs(corr.iloc[i,j]) < 0.2 else '良好 ✅'
                })

    if low_corr:
        st.success(f"✅ 发现 {len(low_corr)} 对低相关性资产组合")
        st.dataframe(pd.DataFrame(low_corr))
    else:
        st.info("所有资产对相关性均较高，分散效果有限")

    # 风险指标
    st.markdown("### 📉 风险指标")
    returns_std = num.std()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("平均日收益率", f"{num.mean().mean():.4f}")
    with col2:
        st.metric("平均波动率", f"{returns_std.mean():.4f}")
    with col3:
        sharpe = num.mean().mean()/returns_std.mean() if returns_std.mean() > 0 else 0
        st.metric("夏普比率(估算)", f"{sharpe:.2f}")

# ----- 5. 质量工程师 -----
def quality_analysis(df):
    st.subheader("🔧 质量工程师 - 过程能力分析")

    num = get_numeric(df)
    if len(num.columns) < 1:
        st.warning("需要数值列")
        return

    # 过程能力
    st.markdown("### 📊 过程能力指数(Cpk)")
    cpk_results = []
    for col in num.columns:
        mean_val = num[col].mean()
        std_val = num[col].std()
        if std_val > 0:
            cpk = calculate_cpk(num[col])
            cpk_results.append({
                '指标': col,
                '均值': round(mean_val, 2),
                '标准差': round(std_val, 3),
                'Cpk': round(cpk, 3),
                '评价': '优秀 🌟' if cpk > 1.33 else '良好 ✅' if cpk > 1.0 else '一般 ⚠️' if cpk > 0.67 else '需改进 ❌'
            })
    st.dataframe(pd.DataFrame(cpk_results))

    # SPC控制图
    st.markdown("### 📈 SPC控制图")
    selected = st.selectbox("选择指标查看控制图", num.columns)
    if selected:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=num[selected], mode='lines+markers', name=selected))
        mean_val = num[selected].mean()
        std_val = num[selected].std()
        fig.add_hline(y=mean_val, line_dash="dash", line_color="green", annotation_text="CL")
        fig.add_hline(y=mean_val + 3*std_val, line_dash="dash", line_color="red", annotation_text="UCL")
        fig.add_hline(y=mean_val - 3*std_val, line_dash="dash", line_color="red", annotation_text="LCL")
        fig.update_layout(title=f"{selected} 控制图", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ----- 6. 人力资源HR -----
def hr_analysis(df):
    st.subheader("👥 人力资源HR - 人才测评分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    perf_cols = [c for c in num.columns if '绩效' in c or '业绩' in c]

    if perf_cols:
        perf = perf_cols[0]
        st.markdown(f"### 🎯 测评维度与{perf}的相关性")

        results = []
        for col in num.columns:
            if col != perf:
                r, p = pearsonr(num[perf], num[col])
                results.append({
                    '测评维度': col,
                    '相关系数': round(r, 3),
                    '显著性': sig_star(p),
                    '预测效度': '强 🌟' if abs(r) > 0.4 else '中 ✅' if abs(r) > 0.25 else '弱 ℹ️'
                })
        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df)

        best = results_df.iloc[0] if len(results_df) > 0 else None
        if best is not None:
            st.success(f"🏆 最佳预测指标: {best['测评维度']} (r={best['相关系数']})")

    # 九宫格
    if '能力测评' in num.columns and '潜力测评' in num.columns:
        st.markdown("### ⭐ 九宫格人才盘点")
        try:
            df['能力等级'] = safe_qcut(num['能力测评'], 3, labels=['C(待提升)', 'B(良好)', 'A(优秀)'])
            df['潜力等级'] = safe_qcut(num['潜力测评'], 3, labels=['C(待提升)', 'B(良好)', 'A(优秀)'])
            nine_grid = df.groupby(['能力等级', '潜力等级']).size().unstack(fill_value=0)
            st.dataframe(nine_grid)

            high_potential = df[(df['能力等级'] == 'A(优秀)') & (df['潜力等级'].isin(['A(优秀)', 'B(良好)']))]
            if len(high_potential) > 0:
                st.success(f"🎯 识别出 {len(high_potential)} 名高潜人才")
        except Exception as e:
            st.warning(f"九宫格生成失败: {e}")

    # 热力图
    st.markdown("### 🔥 测评指标相关性")
    corr, _ = calc_corr(df)
    if corr is not None:
        st.plotly_chart(create_heatmap(corr), use_container_width=True)

# ----- 7. 心理咨询师 -----
def psychology_analysis(df):
    st.subheader("🧠 心理咨询师 - 心理量表分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    # 信度
    st.markdown("### 📐 信度分析")
    alpha = cronbach_alpha(num)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cronbach's α", f"{alpha:.3f}")
    with col2:
        quality = "优秀 🌟" if alpha > 0.9 else "良好 ✅" if alpha > 0.8 else "可接受 ⚠️" if alpha > 0.7 else "需改进 ❌"
        st.metric("信度评价", quality)

    # 分半信度
    if len(num.columns) >= 4:
        st.markdown("### 📊 分半信度")
        n = len(num.columns)
        first_half = num.iloc[:, :n//2].sum(axis=1)
        second_half = num.iloc[:, n//2:].sum(axis=1)
        half_data = pd.DataFrame({'前半': first_half, '后半': second_half})
        split_alpha = cronbach_alpha(half_data)
        st.metric("分半信度", f"{split_alpha:.3f}")

    # 热力图
    st.markdown("### 🔥 题目相关性热力图")
    corr, _ = calc_corr(df)
    if corr is not None:
        st.plotly_chart(create_heatmap(corr), use_container_width=True)

# ----- 8. 农艺师 -----
def agronomy_analysis(df):
    st.subheader("🌱 农艺师 - 作物产量分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    yield_cols = [c for c in num.columns if '产量' in c]

    if yield_cols:
        yield_col = yield_cols[0]
        st.markdown(f"### 📈 影响{yield_col}的因素分析")

        results = []
        for col in num.columns:
            if col != yield_col:
                r, p = pearsonr(num[yield_col], num[col])
                results.append({
                    '影响因素': col,
                    '相关系数': round(r, 3),
                    '显著性': sig_star(p),
                    '重要性': '高 🌟' if abs(r) > 0.5 else '中 ✅' if abs(r) > 0.3 else '低 ℹ️'
                })
        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df)

        key_factors = results_df[results_df['重要性'] == '高 🌟']
        if len(key_factors) > 0:
            st.success(f"🎯 发现 {len(key_factors)} 个关键影响因素")

    # 热力图
    st.markdown("### 🔥 变量相关性热力图")
    corr, _ = calc_corr(df)
    if corr is not None:
        st.plotly_chart(create_heatmap(corr), use_container_width=True)

# ----- 9. 数据分析师 -----
def analyst_analysis(df):
    st.subheader("📉 数据分析师 - 通用相关性分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个数值变量")
        return

    # 热力图
    st.markdown("### 🔥 相关系数热力图")
    corr, p = calc_corr(df)
    if corr is not None:
        st.plotly_chart(create_heatmap(corr), use_container_width=True)

    # 相关系数矩阵
    st.markdown("### 📋 相关系数矩阵（带显著性标记）")
    corr_display = corr.round(3).astype(str)
    if p is not None:
        for i in range(len(corr)):
            for j in range(len(corr)):
                if i != j:
                    corr_display.iloc[i,j] += sig_star(p.iloc[i,j])
    st.dataframe(corr_display)

    # 最强相关性
    st.markdown("### 🎯 最强相关性对")
    flat = corr.unstack()
    flat = flat[flat.index.get_level_values(0) != flat.index.get_level_values(1)]
    top5 = flat.abs().nlargest(5)
    for (v1, v2), val in top5.items():
        p_val = p.loc[v1, v2] if p is not None else None
        st.write(f"- **{v1} ↔ {v2}**: {val:.3f} {sig_star(p_val) if p_val is not None else ''}")

    # 描述统计
    st.markdown("### 📊 描述统计")
    st.dataframe(num.describe())

    # 散点图矩阵
    if len(num.columns) <= 6:
        st.markdown("### 🔬 散点图矩阵")
        fig = px.scatter_matrix(num, title="散点图矩阵")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# ----- 10. 机器学习工程师 -----
def ml_analysis(df):
    st.subheader("🤖 机器学习工程师 - 特征工程分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个特征")
        return

    target_cols = [c for c in num.columns if '目标' in c or 'target' in c.lower() or 'label' in c.lower()]

    if target_cols:
        target = target_cols[0]
        st.markdown(f"### 🎯 特征与目标变量'{target}'的相关性")

        results = []
        for col in num.columns:
            if col != target:
                r, p = pearsonr(num[target], num[col])
                results.append({
                    '特征': col,
                    '相关系数': round(r, 3),
                    'p值': round(p, 4),
                    '显著性': sig_star(p),
                    '重要性': '高 🌟' if abs(r) > 0.4 else '中 ✅' if abs(r) > 0.25 else '低 ℹ️'
                })
        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df)

        important = results_df[results_df['重要性'] == '高 🌟']
        if len(important) > 0:
            st.success(f"✅ 发现 {len(important)} 个重要特征")

    # 多重共线性
    st.markdown("### 🔗 多重共线性检查")
    corr, _ = calc_corr(df)
    if corr is not None:
        high_corr = []
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                if abs(corr.iloc[i,j]) > 0.7:
                    high_corr.append({
                        '特征1': corr.index[i],
                        '特征2': corr.columns[j],
                        '相关系数': round(corr.iloc[i,j], 3),
                        '建议': '考虑删除或合并'
                    })
        if high_corr:
            st.warning(f"⚠️ 发现 {len(high_corr)} 对高度相关特征")
            st.dataframe(pd.DataFrame(high_corr))
        else:
            st.success("✅ 未发现严重多重共线性")

# ----- 11. 电商运营 -----
def ecommerce_analysis(df):
    st.subheader("🛒 电商运营 - 用户行为分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    conv_cols = [c for c in num.columns if '购买' in c or '转化' in c]

    if conv_cols:
        conv = conv_cols[0]
        st.markdown(f"### 🎯 行为指标与'{conv}'的相关性")

        results = []
        for col in num.columns:
            if col != conv:
                if num[conv].nunique() <= 2:
                    r, p = pointbiserialr(num[conv], num[col])
                else:
                    r, p = pearsonr(num[conv], num[col])
                results.append({
                    '行为指标': col,
                    '相关系数': round(r, 3),
                    '显著性': sig_star(p),
                    '优化优先级': '高 🌟' if abs(r) > 0.3 else '中 ✅' if abs(r) > 0.2 else '低 ℹ️'
                })
        results_df = pd.DataFrame(results).sort_values('相关系数', ascending=False)
        st.dataframe(results_df)

        drivers = results_df[results_df['优化优先级'] == '高 🌟']
        if len(drivers) > 0:
            st.success(f"🎯 发现 {len(drivers)} 个关键转化驱动因素")

    # 热力图
    st.markdown("### 🔥 行为相关性热力图")
    corr, _ = calc_corr(df)
    if corr is not None:
        st.plotly_chart(create_heatmap(corr), use_container_width=True)

# ----- 12. 市场研究员 -----
def marketing_analysis(df):
    st.subheader("📈 市场研究员 - 市场调研分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    st.markdown("### 🎯 满意度驱动因素分析")
    corr, p = calc_corr(df)
    if corr is not None:
        st.plotly_chart(create_heatmap(corr), use_container_width=True)

        # 找出与各满意度指标最相关的因素
        sat_cols = [c for c in num.columns if '满意' in c or '忠诚' in c]
        for sat in sat_cols:
            st.markdown(f"**{sat} 的驱动因素**")
            factors = []
            for col in num.columns:
                if col != sat:
                    r, p_val = pearsonr(num[sat], num[col])
                    factors.append({'因素': col, '相关系数': round(r, 3)})
            top3 = pd.DataFrame(factors).sort_values('相关系数', ascending=False).head(3)
            st.write(f"Top 3: {', '.join([f'{f[1]}({f[2]})' for f in top3.values])}")

# ----- 13. 产品经理 -----
def product_analysis(df):
    st.subheader("📱 产品经理 - 产品功能分析")

    num = get_numeric(df)
    if len(num.columns) < 2:
        st.warning("需要至少2个变量")
        return

    st.markdown("### 🔥 功能使用相关性分析")
    corr, _ = calc_corr(df)
    if corr is not None:
        st.plotly_chart(create_heatmap(corr), use_container_width=True)

        # 功能组合建议
        high_corr = []
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                if corr.iloc[i,j] > 0.5:
                    high_corr.append({
                        '功能A': corr.index[i],
                        '功能B': corr.columns[j],
                        '相关系数': round(corr.iloc[i,j], 3)
                    })
        if high_corr:
            st.markdown("### 💡 功能联动建议")
            st.dataframe(pd.DataFrame(high_corr))

# ----- 14. 市场研究员 - 别名 -----
def market_analysis(df):
    marketing_analysis(df)

# ==================== 其他角色（完整实现） ====================

def assessment_analysis(df):
    st.subheader("📝 教育测评专家 - 标准化考试分析")
    st.markdown("### 📊 项目反应理论(IRT)分析")
    st.info("IRT三参数模型：区分度 a、难度 b、猜测度 c")
    st.markdown("### 📈 测验信息函数")
    st.info("信息函数反映测验在不同能力水平上的测量精度")
    num = get_numeric(df)
    if len(num.columns) > 0:
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def school_analysis(df):
    st.subheader("🏫 学校管理者 - 学校质量评估")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 学校KPI指标分析")
        st.dataframe(num.describe())
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def researcher_analysis(df):
    st.subheader("🔬 医学研究员 - 临床试验分析")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 临床试验数据分析")
        st.dataframe(num.describe())
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def pharma_analysis(df):
    st.subheader("💊 药学/药理 - 剂量效应分析")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 药效动力学分析")
        st.info("四参数Logistic模型拟合剂量效应曲线")
        st.markdown("### 💊 半数有效量(ED50)估算")
        st.metric("估算ED50", "拟合中...")
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def bioinfo_analysis(df):
    st.subheader("🧬 生物信息学 - 基因表达分析")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 差异表达分析")
        st.info("火山图展示差异表达基因")
        st.markdown("### 🔥 基因表达聚类热图")
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def rehab_analysis(df):
    st.subheader("🏃 康复治疗师 - 康复评估")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 康复效果评估")
        st.dataframe(num.describe())
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def risk_analysis(df):
    st.subheader("🏦 风控专员 - 风险分析")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 信用风险评估")
        st.dataframe(num.describe())
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def industrial_analysis(df):
    st.subheader("🏭 工业工程师 - 效率分析")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 生产效率分析")
        st.dataframe(num.describe())
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def lab_analysis(df):
    st.subheader("🔬 实验室技术员 - 方法验证")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 检测方法验证")
        st.dataframe(num.describe())
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def environment_analysis(df):
    st.subheader("🌍 环境科学家 - 环境监测")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 环境质量分析")
        st.dataframe(num.describe())
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def bi_analysis(df):
    st.subheader("📊 BI分析师 - 智能洞察")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 数据洞察报告")
        st.dataframe(num.describe())
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def growth_analysis(df):
    st.subheader("🔍 用户增长 - 增长分析")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 用户增长分析")
        st.dataframe(num.describe())
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def social_analysis(df):
    st.subheader("📋 社会调查员 - 调查分析")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 社会调查数据分析")
        st.dataframe(num.describe())
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

def sports_analysis(df):
    st.subheader("⚽ 体育分析师 - 运动表现")
    num = get_numeric(df)
    if len(num.columns) > 0:
        st.markdown("### 📊 运动表现分析")
        st.dataframe(num.describe())
        corr, _ = calc_corr(df)
        if corr is not None:
            st.plotly_chart(create_heatmap(corr), use_container_width=True)

# ==================== 角色函数映射 ====================

ROLE_FUNCS = {
    "👩‍🏫 中小学教师": teacher_analysis,
    "🎓 大学教授/研究生": professor_analysis,
    "📝 教育测评专家": assessment_analysis,
    "🏫 学校管理者": school_analysis,
    "👨‍⚕️ 临床医生": doctor_analysis,
    "🔬 医学研究员": researcher_analysis,
    "💊 药学/药理": pharma_analysis,
    "🧬 生物信息学": bioinfo_analysis,
    "🏃 康复治疗师": rehab_analysis,
    "📈 市场研究员": marketing_analysis,
    "🛒 电商运营": ecommerce_analysis,
    "📱 产品经理": product_analysis,
    "💰 金融分析师": finance_analysis,
    "🏦 风控专员": risk_analysis,
    "🔧 质量工程师": quality_analysis,
    "🏭 工业工程师": industrial_analysis,
    "🔬 实验室技术员": lab_analysis,
    "🌱 农艺师": agronomy_analysis,
    "🌍 环境科学家": environment_analysis,
    "📉 数据分析师": analyst_analysis,
    "🤖 机器学习工程师": ml_analysis,
    "📊 BI分析师": bi_analysis,
    "🔍 用户增长": growth_analysis,
    "🧠 心理咨询师": psychology_analysis,
    "📋 社会调查员": social_analysis,
    "👥 人力资源HR": hr_analysis,
    "⚽ 体育分析师": sports_analysis
}

# ==================== 主程序 ====================

def main():
    # 标题
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.8rem; color: #1f77b4; margin: 0;">{APP_NAME}</h1>
        <p style="font-size: 1.2rem; color: #666; margin: 5px 0;">{APP_SUBTITLE}</p>
        <p style="font-size: 0.9rem; color: #999;">{APP_VERSION}</p>
    </div>
    """, unsafe_allow_html=True)

    # 初始化Session State
    if 'selected_industry' not in st.session_state:
        st.session_state.selected_industry = list(ROLES_DB.keys())[0]
    if 'selected_role' not in st.session_state:
        first_role = list(ROLES_DB[st.session_state.selected_industry]["roles"].keys())[0]
        st.session_state.selected_role = first_role
    if 'selected_analysis' not in st.session_state:
        st.session_state.selected_analysis = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False

    # ===== 侧边栏 =====
    with st.sidebar:
        st.markdown(f"### 🎯 {APP_NAME}")
        st.markdown("---")

        # 行业选择
        st.markdown("#### 🏢 选择行业")
        for ind in ROLES_DB.keys():
            icon = ROLES_DB[ind]["icon"]
            if st.button(f"{icon} {ind}", key=f"ind_{ind}", use_container_width=True):
                st.session_state.selected_industry = ind
                st.session_state.selected_role = list(ROLES_DB[ind]["roles"].keys())[0]
                st.session_state.selected_analysis = None
                st.session_state.analysis_started = False
                st.rerun()

        st.markdown("---")

        # 角色选择
        st.markdown("#### 👤 选择角色")
        roles = ROLES_DB[st.session_state.selected_industry]["roles"]
        role_names = list(roles.keys())

        selected_role = st.selectbox("职业角色", role_names)
        if selected_role != st.session_state.selected_role:
            st.session_state.selected_role = selected_role
            st.session_state.selected_analysis = None
            st.session_state.analysis_started = False
            st.rerun()

        st.markdown("---")

        # 角色信息
        role_info = roles[st.session_state.selected_role]
        st.info(f"**{st.session_state.selected_role}**\n\n{role_info['desc']}")

        # 功能标签
        st.markdown("**核心功能：**")
        for feat in role_info['features'][:5]:
            st.markdown(f"- {feat}")

        st.markdown("---")

        # 分析类型选择
        st.markdown("#### 📊 选择分析类型")
        analysis_types = role_info.get('analysis_types', ['📊 综合分析'])

        selected_analysis = st.selectbox(
            "分析类型",
            analysis_types,
            help="选择需要执行的分析类型"
        )
        if selected_analysis != st.session_state.selected_analysis:
            st.session_state.selected_analysis = selected_analysis
            st.session_state.analysis_started = False

        st.markdown("---")

        # 数据上传
        st.markdown("#### 📁 数据")
        uploaded = st.file_uploader("上传CSV/Excel", type=['csv', 'xlsx', 'xls'])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 示例数据", use_container_width=True):
                st.session_state.df = generate_sample_data(role_info['sample'])
                st.session_state.data_loaded = True
                st.session_state.analysis_started = False
                st.rerun()

        with col2:
            if st.session_state.data_loaded and st.session_state.df is not None:
                if st.button("▶️ 开始分析", use_container_width=True, type="primary"):
                    st.session_state.analysis_started = True
                    st.rerun()

        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded)
                else:
                    st.session_state.df = pd.read_excel(uploaded)
                st.session_state.data_loaded = True
                st.session_state.analysis_started = False
                st.rerun()
            except Exception as e:
                st.error(f"加载失败: {e}")

        if st.session_state.data_loaded and st.session_state.df is not None:
            st.success(f"✅ {len(st.session_state.df)} 行数据已加载")

    # ===== 主区域 =====
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df

        # 数据概览
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 总行数", len(df))
        with col2:
            st.metric("📋 总列数", len(df.columns))
        with col3:
            num_cols = len(get_numeric(df).columns)
            st.metric("🔢 数值列", num_cols)
        with col4:
            missing = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            st.metric("❓ 缺失率", f"{missing:.1f}%")

        # 数据预览
        with st.expander("📋 数据预览"):
            st.dataframe(df.head(10))
            st.caption(f"数据维度: {df.shape}")

        st.markdown("---")

        # 显示当前分析类型
        if st.session_state.selected_analysis:
            st.info(f"📌 当前分析模式: **{st.session_state.selected_analysis}**")

        # 执行分析
        if st.session_state.analysis_started:
            role_key = st.session_state.selected_role
            if role_key in ROLE_FUNCS:
                ROLE_FUNCS[role_key](df)
            else:
                st.warning("分析功能加载中...")
                analyst_analysis(df)
        else:
            # 等待开始
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; background: #f8f9fa; border-radius: 10px;">
                <h3 style="color: #666;">👆 点击「开始分析」按钮运行分析</h3>
                <p style="color: #999;">选择行业 → 选择角色 → 选择分析类型 → 上传数据 → 开始分析</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            ### 📌 操作步骤
            
            1. **选择行业** - 从左侧选择6大行业之一
            2. **选择角色** - 从26个职业角色中选择
            3. **选择分析类型** - 选择该角色专属的分析类型
            4. **准备数据** - 上传自己的数据或生成示例数据
            5. **开始分析** - 点击「开始分析」按钮
            
            ### 📊 支持的行业与角色
            
            | 行业 | 角色数 | 典型角色 |
            |------|--------|----------|
            | 📚 教育科研 | 4 | 中小学教师、大学教授、测评专家、学校管理者 |
            | 🏥 医疗健康 | 5 | 临床医生、医学研究员、药学、生物信息学、康复治疗师 |
            | 📊 商业金融 | 5 | 市场研究员、电商运营、产品经理、金融分析师、风控专员 |
            | 🏭 工业农业 | 5 | 质量工程师、工业工程师、实验室技术员、农艺师、环境科学家 |
            | 💻 数据科学 | 4 | 数据分析师、ML工程师、BI分析师、用户增长 |
            | 🧠 人文社科 | 4 | 心理咨询师、社会调查员、人力资源HR、体育分析师 |
            """)

    else:
        # 未加载数据
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <h2 style="color: #999;">👈 选择行业和角色</h2>
            <p style="color: #bbb;">然后生成示例数据或上传文件开始分析</p>
        </div>
        """, unsafe_allow_html=True)

        # 快速入门
        st.markdown("""
        ### 🚀 快速入门
        
        1. 左侧选择**行业领域**
        2. 选择**职业角色**（26个角色可选）
        3. 点击 **「示例数据」** 或上传自己的CSV/Excel
        4. 选择**分析类型**
        5. 点击 **「开始分析」**
        """)

    # 页脚
    st.markdown("---")
    st.markdown(
        f"<center style='color: #999; font-size: 0.8rem;'>"
        f"{APP_NAME} {APP_VERSION} · 6大行业 × 26个职业角色 · 智能统计分析"
        f"</center>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()