"""
CorrMatrix Explorer - 增强互动版
一个交互式相关性分析工具
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
from io import StringIO
import base64

# 页面配置
st.set_page_config(
    page_title="CorrMatrix Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.title("📊 CorrMatrix Explorer")
st.markdown("*交互式相关性分析工具 - 点击、拖动、探索你的数据*")

# 定义 matplotlib 可用的 colormap（用于表格背景渐变）
MATPLOTLIB_CMAPS = {
    'RdBu_r': 'RdBu_r',
    'Viridis': 'viridis',
    'Plasma': 'plasma',
    'Cividis': 'cividis',
    'Coolwarm': 'coolwarm'
}

# 侧边栏配置
with st.sidebar:
    st.header("🎮 控制面板")

    # 数据上传
    st.subheader("📁 数据上传")
    uploaded_file = st.file_uploader(
        "上传 CSV 或 Excel 文件",
        type=['csv', 'xlsx', 'xls'],
        help="支持 CSV 和 Excel 格式"
    )

    # 示例数据选项
    use_example = st.checkbox("使用示例数据", value=uploaded_file is None)

    if use_example:
        st.info("使用示例数据：员工满意度调查")

    st.divider()

    # 变量选择
    st.subheader("🔧 变量选择")

    # 初始化session state
    if 'selected_vars' not in st.session_state:
        st.session_state.selected_vars = []

    # 相关性设置
    st.subheader("📈 相关性设置")
    corr_method = st.selectbox(
        "相关系数类型",
        options=['pearson', 'spearman', 'kendall'],
        format_func=lambda x: {
            'pearson': 'Pearson (线性相关)',
            'spearman': 'Spearman (单调相关)',
            'kendall': 'Kendall (等级相关)'
        }[x]
    )

    # 阈值设置
    corr_threshold = st.slider(
        "相关性阈值 (|r| > )",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="只显示绝对值大于此阈值的相关系数"
    )

    # 显著性设置
    show_pvalues = st.checkbox("显示 p 值", value=True)
    sig_level = st.selectbox(
        "显著性水平",
        options=[0.05, 0.01, 0.001],
        format_func=lambda x: f"p < {x}"
    ) if show_pvalues else None

    st.divider()

    # 数据预处理
    st.subheader("⚙️ 数据预处理")
    handle_missing = st.selectbox(
        "缺失值处理",
        options=['drop', 'mean', 'median', 'ffill'],
        format_func=lambda x: {
            'drop': '删除缺失行',
            'mean': '用均值填充',
            'median': '用中位数填充',
            'ffill': '前向填充'
        }[x]
    )

    # 异常值处理
    handle_outliers = st.checkbox("检测并处理异常值 (Z-score方法)")
    if handle_outliers:
        outlier_threshold = st.slider(
            "Z-score 阈值",
            min_value=2.0,
            max_value=4.0,
            value=3.0,
            step=0.5
        )

    st.divider()

    # 可视化设置
    st.subheader("🎨 可视化设置")
    color_theme = st.selectbox(
        "颜色主题",
        options=['RdBu_r', 'Viridis', 'Plasma', 'Cividis', 'Coolwarm']
    )

    show_numbers = st.checkbox("在热力图上显示数值", value=True)

    st.divider()

    # 导出选项
    st.subheader("💾 导出")
    if st.button("导出相关系数矩阵 (CSV)"):
        if 'corr_matrix' in st.session_state:
            csv = st.session_state.corr_matrix.to_csv()
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="correlation_matrix.csv">下载 CSV</a>'
            st.markdown(href, unsafe_allow_html=True)


# 主区域
@st.cache_data
def load_data(file, use_example):
    """加载数据"""
    if use_example:
        # 生成示例数据
        np.random.seed(42)
        n = 200

        # 创建相关变量
        age = np.random.normal(35, 10, n)
        income = 30000 + age * 2000 + np.random.normal(0, 10000, n)
        satisfaction = 3 + 0.05 * age + np.random.normal(0, 0.5, n)
        work_hours = 40 + np.random.normal(0, 5, n)
        performance = 70 + 0.3 * age + np.random.normal(0, 10, n)
        years_company = np.random.exponential(5, n)
        salary_satisfaction = 2.5 + 0.0001 * income + np.random.normal(0, 0.5, n)

        df = pd.DataFrame({
            '年龄': np.clip(age, 20, 60),
            '月收入': np.clip(income, 20000, 80000),
            '工作满意度': np.clip(satisfaction, 1, 5),
            '每周工时': np.clip(work_hours, 30, 60),
            '绩效评分': np.clip(performance, 50, 100),
            '司龄': np.clip(years_company, 0, 20),
            '薪酬满意度': np.clip(salary_satisfaction, 1, 5)
        })

        # 添加一些缺失值
        for col in df.columns:
            if np.random.random() < 0.05:
                idx = np.random.choice(df.index, size=5, replace=False)
                df.loc[idx, col] = np.nan

        return df

    elif uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df

    return None


def preprocess_data(df, handle_missing, handle_outliers=False, outlier_threshold=3):
    """数据预处理"""
    df_processed = df.copy()

    # 只选择数值列
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed = df_processed[numeric_cols]

    # 处理缺失值
    if handle_missing == 'drop':
        df_processed = df_processed.dropna()
    elif handle_missing == 'mean':
        df_processed = df_processed.fillna(df_processed.mean())
    elif handle_missing == 'median':
        df_processed = df_processed.fillna(df_processed.median())
    elif handle_missing == 'ffill':
        df_processed = df_processed.fillna(method='ffill')

    # 处理异常值
    if handle_outliers:
        z_scores = np.abs(stats.zscore(df_processed))
        df_processed = df_processed[(z_scores < outlier_threshold).all(axis=1)]

    return df_processed


def calculate_correlations(df, method='pearson', threshold=0, show_pvalues=True, sig_level=0.05):
    """计算相关系数和p值"""
    corr_matrix = df.corr(method=method)

    # 应用阈值
    if threshold > 0:
        corr_matrix = corr_matrix.where(np.abs(corr_matrix) >= threshold, 0)

    p_matrix = None
    sig_matrix = None

    if show_pvalues:
        p_matrix = pd.DataFrame(np.zeros_like(corr_matrix),
                                index=corr_matrix.index,
                                columns=corr_matrix.columns)

        for i in range(len(df.columns)):
            for j in range(len(df.columns)):
                if method == 'pearson':
                    corr, p = stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
                elif method == 'spearman':
                    corr, p = stats.spearmanr(df.iloc[:, i], df.iloc[:, j])
                else:
                    corr, p = stats.kendalltau(df.iloc[:, i], df.iloc[:, j])
                p_matrix.iloc[i, j] = p

        # 创建显著性标记矩阵
        sig_matrix = p_matrix.map(lambda p:
                                  '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < sig_level else '')

    return corr_matrix, p_matrix, sig_matrix


def create_heatmap(corr_matrix, sig_matrix=None, show_numbers=True, color_scale='RdBu_r'):
    """创建交互式热力图"""
    fig = px.imshow(
        corr_matrix,
        text_auto=True if show_numbers and sig_matrix is None else False,
        color_continuous_scale=color_scale,
        aspect='auto',
        zmin=-1, zmax=1,
        title='相关系数热力图'
    )

    # 如果需要显示显著性标记，创建自定义文本
    if show_numbers and sig_matrix is not None:
        text = np.empty_like(corr_matrix, dtype=object)
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                if i == j:
                    text[i, j] = '1.00'
                else:
                    val = corr_matrix.iloc[i, j]
                    if pd.notna(val) and val != 0:
                        text_val = f'{val:.2f}'
                        sig = sig_matrix.iloc[i, j] if sig_matrix is not None else ''
                        text[i, j] = f'{text_val}{sig}'
                    else:
                        text[i, j] = ''
        fig.update_traces(text=text, texttemplate='%{text}')

    fig.update_layout(
        height=600,
        width=800,
        xaxis_title='变量',
        yaxis_title='变量',
        font=dict(size=12)
    )

    return fig


def create_clustered_heatmap(corr_matrix, color_scale='RdBu_r'):
    """创建聚类热力图"""
    try:
        # 层次聚类
        distance = squareform(1 - np.abs(corr_matrix))
        linkage = hierarchy.linkage(distance, method='average')

        # 重排行和列
        order = hierarchy.dendrogram(linkage, no_plot=True)['leaves']
        corr_clustered = corr_matrix.iloc[order, order]

        fig = px.imshow(
            corr_clustered,
            color_continuous_scale=color_scale,
            aspect='auto',
            zmin=-1, zmax=1,
            title='聚类热力图 (自动重排)'
        )

        fig.update_layout(height=600, width=800)
        return fig
    except Exception as e:
        st.warning(f"聚类热力图生成失败: {e}")
        return create_heatmap(corr_matrix, None, False, color_scale)


def create_scatter_matrix(df, corr_method='pearson'):
    """创建散点图矩阵"""
    fig = px.scatter_matrix(
        df,
        dimensions=df.columns,
        title=f'散点图矩阵 ({corr_method} 相关系数)',
        labels={col: col for col in df.columns}
    )

    fig.update_traces(diagonal_visible=False)
    fig.update_layout(height=800, width=1000)

    return fig


# 加载数据
df_raw = load_data(uploaded_file, use_example)

if df_raw is not None:
    # 显示数据概览
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总行数", len(df_raw))
    with col2:
        st.metric("总列数", len(df_raw.columns))
    with col3:
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
        st.metric("数值列数", len(numeric_cols))
    with col4:
        missing_pct = (df_raw.isnull().sum().sum() / (len(df_raw) * len(df_raw.columns))) * 100
        st.metric("缺失值比例", f"{missing_pct:.1f}%")

    # 数据预览
    with st.expander("📋 数据预览", expanded=False):
        st.dataframe(df_raw.head(10))
        st.caption(f"数据维度: {df_raw.shape}")

    # 数据预处理
    df_processed = preprocess_data(df_raw, handle_missing, handle_outliers,
                                   outlier_threshold if handle_outliers else 3)

    if len(df_processed.columns) < 2:
        st.error("至少需要2个数值列才能进行相关性分析")
        st.stop()

    # 变量选择
    all_vars = df_processed.columns.tolist()

    # 获取侧边栏的变量选择
    with st.sidebar:
        st.session_state.selected_vars = st.multiselect(
            "选择要分析的变量",
            options=all_vars,
            default=all_vars[:min(5, len(all_vars))],
            help="至少选择2个变量"
        )

    selected_vars = st.session_state.selected_vars

    if len(selected_vars) < 2:
        st.warning("请至少选择2个变量进行分析")
        st.stop()

    # 筛选数据
    df_selected = df_processed[selected_vars]

    # 计算相关性
    corr_matrix, p_matrix, sig_matrix = calculate_correlations(
        df_selected, corr_method, corr_threshold, show_pvalues, sig_level
    )

    # 存储到session state供导出使用
    st.session_state.corr_matrix = corr_matrix

    # 主内容区域 - 两列布局
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # 标签页
        tab1, tab2, tab3, tab4 = st.tabs(["📊 热力图", "🔬 散点图矩阵", "🌲 聚类热力图", "📈 相关性详情"])

        with tab1:
            fig = create_heatmap(corr_matrix, sig_matrix if show_pvalues else None,
                                 show_numbers, color_theme)
            st.plotly_chart(fig, use_container_width=True)

            # 点击交互说明
            st.info("💡 **提示**: 在右侧选择变量对查看详细散点图分析")

        with tab2:
            if len(selected_vars) <= 10:
                fig = create_scatter_matrix(df_selected, corr_method)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"变量过多 ({len(selected_vars)}个)，散点图矩阵可能难以阅读。建议减少变量数量。")

        with tab3:
            fig = create_clustered_heatmap(corr_matrix, color_theme)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            # 显示相关性详情表格 - 修复：使用正确的 matplotlib colormap
            st.subheader("相关系数矩阵")
            # 转换颜色主题为 matplotlib 兼容格式
            mpl_cmap = MATPLOTLIB_CMAPS.get(color_theme, 'RdBu_r')
            st.dataframe(corr_matrix.style.background_gradient(cmap=mpl_cmap, vmin=-1, vmax=1))

            if show_pvalues and p_matrix is not None:
                st.subheader("p值矩阵")
                st.dataframe(p_matrix.style.background_gradient(cmap='Reds', vmin=0, vmax=0.1))

    with right_col:
        st.subheader("🔍 交互分析")

        # 变量对选择
        st.markdown("### 选择变量对")
        var1 = st.selectbox("变量 X", selected_vars, key='var1')
        var2 = st.selectbox("变量 Y", selected_vars, key='var2')

        if var1 and var2 and var1 != var2:
            # 获取相关系数
            corr_val = corr_matrix.loc[var1, var2]
            if show_pvalues and p_matrix is not None:
                p_val = p_matrix.loc[var1, var2]

            # 显示统计信息
            st.markdown("### 📊 统计结果")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("相关系数", f"{corr_val:.3f}")
            with col2:
                if show_pvalues and p_matrix is not None:
                    sig_text = ""
                    if p_val < 0.001:
                        sig_text = "***"
                    elif p_val < 0.01:
                        sig_text = "**"
                    elif p_val < 0.05:
                        sig_text = "*"
                    st.metric("p值", f"{p_val:.4f} {sig_text}")

            # 解释相关性强度
            abs_corr = abs(corr_val)
            if abs_corr >= 0.7:
                strength = "非常强的"
            elif abs_corr >= 0.5:
                strength = "较强的"
            elif abs_corr >= 0.3:
                strength = "中等的"
            elif abs_corr >= 0.1:
                strength = "较弱的"
            else:
                strength = "极弱的"

            direction = "正相关" if corr_val > 0 else "负相关"
            st.info(f"**解释**: {var1} 和 {var2} 之间存在 {strength}{direction} (r={corr_val:.3f})")

            # 散点图
            st.markdown("### 📈 散点图")
            fig = px.scatter(
                df_selected,
                x=var1,
                y=var2,
                title=f"{var1} vs {var2}",
                trendline="ols",
                trendline_color_override="red"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # 描述性统计
            st.markdown("### 📋 描述性统计")
            desc_df = pd.DataFrame({
                '统计量': ['样本量', '均值', '标准差', '最小值', '最大值'],
                var1: [
                    len(df_selected[var1].dropna()),
                    df_selected[var1].mean(),
                    df_selected[var1].std(),
                    df_selected[var1].min(),
                    df_selected[var1].max()
                ],
                var2: [
                    len(df_selected[var2].dropna()),
                    df_selected[var2].mean(),
                    df_selected[var2].std(),
                    df_selected[var2].min(),
                    df_selected[var2].max()
                ]
            })
            st.dataframe(desc_df)

        # 全局统计
        st.markdown("### 📊 全局统计")

        # 最强相关性
        corr_flat = corr_matrix.unstack()
        corr_flat = corr_flat[corr_flat.index.get_level_values(0) != corr_flat.index.get_level_values(1)]
        if len(corr_flat) > 0:
            strongest = corr_flat.abs().idxmax()
            strongest_val = corr_flat[strongest]
            st.info(f"**最强相关性**: {strongest[0]} ↔ {strongest[1]} = {strongest_val:.3f}")

        # 相关性分布
        st.markdown("### 📊 相关性分布")
        corr_values = corr_flat.values
        fig = px.histogram(
            corr_values,
            nbins=20,
            title="相关系数分布",
            labels={'value': '相关系数', 'count': '频数'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

else:
    # 未上传数据时的提示
    st.info("👈 请从侧边栏上传 CSV/Excel 文件，或勾选「使用示例数据」开始探索")

    # 显示功能说明
    st.markdown("""
    ### ✨ 功能特点

    - **交互式热力图**: 点击格子查看详细分析
    - **多种相关系数**: Pearson, Spearman, Kendall
    - **动态变量选择**: 自由组合分析变量
    - **数据预处理**: 缺失值处理、异常值检测
    - **显著性检验**: 自动标注 p 值显著性
    - **散点图矩阵**: 多变量关系可视化
    - **聚类热力图**: 自动重排显示相关结构
    - **一键导出**: 下载相关系数矩阵

    ### 📁 支持的文件格式
    - CSV 文件 (.csv)
    - Excel 文件 (.xlsx, .xls)

    ### 💡 使用建议
    1. 上传包含数值型变量的数据
    2. 至少选择2个变量进行分析
    3. 使用阈值滑块筛选强相关性
    4. 在右侧选择变量对进行深入分析
    """)

# 页脚
st.markdown("---")
st.markdown(
    "<center>CorrMatrix Explorer - 让数据分析变得简单有趣</center>",
    unsafe_allow_html=True
)