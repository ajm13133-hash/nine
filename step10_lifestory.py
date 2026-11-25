# -*- coding: utf-8 -*-
"""步骤10（新版·单文件版）：「政策概念生命史漫游」交互平台
输入任意政策词，生成它的专属传记页面：
诞生 → 生命曲线 → 朋友圈变迁 → 语义画像 → RAG生命故事
另含「全景可视化」「LDA主题互动」「智能问答（RAG检索）」页面。

运行方式（终端中）：
    streamlit run step10_lifestory.py
"""
import json
import os
import re

import jieba
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title='政策概念生命史漫游', layout='wide')

ERAS = [('1954—1977', lambda y: y <= 1977),
        ('1978—2012', lambda y: 1978 <= y <= 2012),
        ('2013—2026', lambda y: y >= 2013)]
ERA_COLORS = ['#E45756', '#4C78A8', '#54A24B']

QUERY_STOP = {'哪些', '什么', '如何', '怎么', '怎样', '方面', '请问',
              '一下', '我们', '你们', '它们', '他们', '进行', '以及',
              '报告', '中有', '政府工作'}


# ================= 红白主题样式 =================
def inject_css():
    st.markdown("""
    <style>
    h1 { color: #B80000; font-weight: 800; }
    h2, h3 { color: #B80000; }
    section[data-testid="stSidebar"] { background-color: #FCF3F3; }
    div[data-testid="stMetricValue"] { color: #C00000; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #666; }
    .stProgress > div > div > div > div { background-color: #C00000; }
    hr { border-color: #EACACA; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { color: #8a4b4a; }
    </style>
    """, unsafe_allow_html=True)


# ================= RAG 检索 / 生命史生成 =================

class Corpus:
    """语料与TF-IDF索引（懒加载，只建一次）"""
    loaded = False
    stopwords = None
    reports = None
    paras = None       # [(year, text)]
    vec = None
    X = None


def _load():
    if Corpus.loaded:
        return
    Corpus.stopwords = set(w.strip() for w in open('stopwords.txt', encoding='utf-8'))
    Corpus.reports = json.load(open('data/gov_reports.json', encoding='utf-8'))
    Corpus.paras = []
    for r in Corpus.reports:
        for p in re.split(r'[\n。]', r['text']):
            p = p.strip()
            if len(p) >= 40:
                Corpus.paras.append((r['year'], p))
    Corpus.vec = TfidfVectorizer(tokenizer=tokenize, max_features=8000)
    Corpus.X = Corpus.vec.fit_transform([p for _, p in Corpus.paras])
    Corpus.loaded = True


def tokenize(text):
    return [w for w in jieba.lcut(text)
            if len(w) > 1 and w not in Corpus.stopwords and w not in QUERY_STOP]


def retrieve(question, top_k=3):
    """通用RAG检索：TF-IDF相似度 + 词表对齐 + 原文包含兜底"""
    _load()
    q_words = tokenize(question)
    if not q_words:
        return []
    # 词表对齐：不在TF-IDF词表中的查询词（如被jieba整词切分的"生态文明"）
    # 用搜索引擎模式再切小，避免查询向量成为零向量
    aligned = []
    for w in q_words:
        if w in Corpus.vec.vocabulary_:
            aligned.append(w)
        else:
            aligned += [x for x in jieba.lcut_for_search(w)
                        if len(x) > 1 and x in Corpus.vec.vocabulary_]
    hits = []
    if aligned:
        sims = cosine_similarity(Corpus.vec.transform([' '.join(aligned)]),
                                 Corpus.X)[0]
        for i in np.argsort(sims)[::-1][:50]:
            if sims[i] <= 0.02:
                break
            cover = sum(1 for w in q_words if w in Corpus.paras[i][1]) / len(q_words)
            if cover >= 0.2:
                hits.append((Corpus.paras[i][0], Corpus.paras[i][1],
                             float(sims[i])))
            if len(hits) >= top_k:
                break
    if not hits:
        # 兜底：原文直接包含任一查询词即算命中，取最新的几段
        cands = [(y, t) for y, t in Corpus.paras
                 if any(w in t for w in q_words)]
        hits = [(y, t, None) for y, t in cands[-top_k:]]
    return hits


def _sentences_with(word, year):
    """某年报告中包含该词的句子"""
    for r in Corpus.reports:
        if r['year'] == year:
            return [s.strip() for s in re.split(r'[。；]', r['text'])
                    if word in s and 15 < len(s.strip()) < 150]
    return []


def era_of(year):
    for era, cond in ERAS:
        if cond(year):
            return era
    return ''


def word_profile(word):
    """一个词的生命档案：诞生、峰值、现状、各时代伙伴"""
    _load()
    freq = {}  # year -> 每万字频次
    total = 0
    for r in Corpus.reports:
        c = r['text'].count(word)
        if c:
            freq[r['year']] = c / max(len(r['text']) / 10000, 1)
            total += c
    if not freq:
        return None
    years = sorted(freq)
    birth, peak = years[0], max(freq, key=freq.get)
    recent = years[-1]

    # 各时代窗口共现伙伴（把查询词加入jieba词典后再分词，避免被切开）
    jieba.add_word(word)
    partners = {era: {} for era, _ in ERAS}
    for r in Corpus.reports:
        if word not in r['text']:
            continue
        words = [w for w in jieba.lcut(r['text'])
                 if len(w) > 1 and w not in Corpus.stopwords]
        for era, cond in ERAS:
            if not cond(r['year']):
                continue
            co = partners[era]
            for i, w in enumerate(words):
                if w == word:
                    for j in range(max(0, i - 10), min(len(words), i + 11)):
                        nb = words[j]
                        if nb != word:
                            co[nb] = co.get(nb, 0) + 1
    partners = {era: sorted(co.items(), key=lambda x: x[1],
                            reverse=True)[:6]
                for era, co in partners.items()}

    return {
        'word': word, 'freq': freq, 'years': years, 'total': total,
        'birth_year': birth,
        'birth_sentence': (_sentences_with(word, birth) or [''])[0],
        'peak_year': peak, 'peak_freq': freq[peak],
        'peak_sentence': (_sentences_with(word, peak) or [''])[0],
        'recent_year': recent,
        'recent_sentence': (_sentences_with(word, recent) or [''])[0],
        'partners': partners,
    }


def life_story(word):
    """组装生命故事（RAG：检索原文 + 模板生成）"""
    prof = word_profile(word)
    if not prof:
        return None
    f = prof['freq']
    trend = []
    ys = prof['years']
    if len(ys) > 1:
        first_half = np.mean([f[y] for y in ys[:len(ys) // 2]])
        second_half = np.mean([f[y] for y in ys[len(ys) // 2:]])
        trend = '上升' if second_half > first_half * 1.3 else (
            '下降' if second_half < first_half * 0.7 else '平稳')

    p = prof['partners']
    story = []
    story.append(f"【诞生】「{word}」最早出现在{prof['birth_year']}年的"
                 f"《政府工作报告》中。当年的原文写道："
                 f"“{prof['birth_sentence']}”。")
    story.append(f"【历程】此后数十年间，该词共在{len(ys)}篇报告中出现，"
                 f"使用强度整体呈{trend}趋势，"
                 f"于{prof['peak_year']}年达到峰值"
                 f"（每万字{prof['peak_freq']:.1f}次）。当年的表述是："
                 f"“{prof['peak_sentence']}”。")
    era_desc = []
    for era, _ in ERAS:
        if p[era]:
            top = '、'.join(w for w, _ in p[era][:3])
            era_desc.append(f"{era}时期常与“{top}”等词结伴出现")
    if era_desc:
        story.append("【朋友圈变迁】" + "；".join(era_desc) +
                     "，反映出该词的政策语境随时代不断迁移。")
    story.append(f"【当下】在{prof['recent_year']}年的报告中，"
                 f"它的最新表述是：“{prof['recent_sentence']}”。")
    return {'profile': prof, 'story': story, 'trend': trend}


# ================= 以下是 Streamlit 界面 =================

@st.cache_resource
def load_models():
    """分时代训练词向量（只训练一次）"""
    seg = json.load(open('data/segmented.json', encoding='utf-8'))
    models = {}
    for era, cond in ERAS:
        sents = [w for y, w in seg.items() if cond(int(y))]
        models[era] = Word2Vec(sents, vector_size=100, window=5,
                               min_count=2, sg=1, epochs=30, seed=42)
    return models


@st.cache_data
def get_profile(word):
    return word_profile(word)


inject_css()
models = load_models()

st.title('政策概念的生命史漫游')
st.markdown('**——1954—2026年国务院《政府工作报告》文本挖掘交互平台**')
st.caption('数据：1954—2026年历年国务院《政府工作报告》（58篇，约122万字）｜'
           '方法：词频统计 · 共词网络 · Word2Vec · LDA · t-SNE/UMAP · SVM/KMeans · RAG')

# ============ 侧栏 ============
st.sidebar.header('🔍 输入一个政策词')
quick = ['生态文明', '市场经济', '创新', '就业', '高质量', '乡村振兴',
         '改革开放', '科技', '民生', '双循环']
word = st.sidebar.text_input('搜索任意政策关键词', value='生态文明')
st.sidebar.caption('试试这些：' + '、'.join(quick))

page = st.sidebar.radio('功能',
                        ['📖 生命史漫游', '🌊 全景可视化',
                         '🕸 LDA主题互动', '❓ 智能问答'])

st.sidebar.divider()
with st.sidebar.expander('ℹ️ 关于本实验'):
    st.markdown(
        '**语料**：1954—2026年历年国务院《政府工作报告》共58篇、'
        '约122万字，采集自维基文库与国家发展和改革委员会官网。\n\n'
        '**分析流水线**：网页采集 → 文本清洗 → jieba分词 → '
        '词频/TF-IDF → 共词网络 → Word2Vec词嵌入 → LDA主题模型 → '
        't-SNE/UMAP降维 → SVM/KMeans分类聚类 → RAG检索增强生成。\n\n'
        '**技术栈**：Python · Streamlit · jieba · gensim · '
        'scikit-learn · NetworkX · pyLDAvis · Matplotlib。')

# ============ 全景页 ============
if page == '🌊 全景可视化':
    st.subheader('全景可视化：实验各步骤的分析成果')
    st.markdown('这里汇总了实验步骤3—9的全部可视化成果，'
                '从不同角度刻画58篇政府工作报告的文本特征与时代变迁。')
    tabs = st.tabs(['关键词河流', '共词网络', '语义漂移',
                    '主题演变', '降维与轨迹', '分类评估'])
    with tabs[0]:
        st.image('figures/keyword_river.png',
                 caption='关键词河流图（步骤3：词频统计与排名）',
                 use_container_width=True)
        st.markdown('**怎么读**：每年纵向一列是该年词频排名前列的关键词，'
                    '色带连接同一关键词在不同年份的名次，色带越高（排名越靠前）、'
                    '越宽，说明该词当年的政策地位越重。可以清楚看到"计划"在早期'
                    '长期霸榜后逐渐让位，"高质量""创新"等新词近年快速登顶。')
    with tabs[1]:
        st.image('figures/coword_network.png',
                 caption='高频词共词网络（步骤4：共词分析）',
                 use_container_width=True)
        st.markdown('**怎么读**：节点为高频词，两词在同一句中共同出现即连边，'
                    '节点越大、越靠近中心，度中心性越高。"企业""加快""发展"等'
                    '处于网络核心，是贯穿报告的高频动作与对象。')
        st.image('figures/coword_eras.png',
                 caption='分时代共词网络对比', use_container_width=True)
        st.markdown('**怎么读**：同一方法分别作用于三个时代，'
                    '可以对比不同时期政策话语的"核心议题圈"如何更替。')
    with tabs[2]:
        st.image('figures/semantic_drift.png',
                 caption='关键词语义画像（步骤5：Word2Vec词嵌入）',
                 use_container_width=True)
        st.markdown('**怎么读**：对每个时代分别训练Word2Vec模型，'
                    '取目标词语义距离最近的"邻居"。同一个词在不同年代的邻居不同，'
                    '即为语义漂移——例如"市场"早年与"投机倒把"为伍，'
                    '近年则与"产权保护"相邻。')
    with tabs[3]:
        st.image('figures/topic_river.png',
                 caption='LDA主题强度河流图（步骤7：主题模型）',
                 use_container_width=True)
        st.markdown('**怎么读**：色带宽度表示该主题在当年报告中的相对强度，'
                    '可以看到"计划经济"主题收窄、"创新发展"主题在2013年后'
                    '显著扩张的主干切换过程。')
        st.image('figures/lda_topics.png',
                 caption='LDA各主题高概率词', use_container_width=True)
    with tabs[4]:
        c1, c2 = st.columns(2)
        c1.image('figures/tsne.png', caption='t-SNE降维（按时代着色）',
                 use_container_width=True)
        c2.image('figures/umap.png', caption='UMAP降维（按时代着色）',
                 use_container_width=True)
        st.markdown('**怎么读**：每篇报告被表示为高维TF-IDF向量后压缩到二维。'
                    '点按时代着色后自然分成三簇，说明三个时期的报告语言风格'
                    '差异显著、各自内部高度一致。')
        st.image('figures/tsne_trajectory.png',
                 caption='政策语言演变轨迹（步骤8：降维与聚类）',
                 use_container_width=True)
        st.markdown('**怎么读**：在t-SNE平面上用箭头连接相邻年份。'
                    '轨迹总体单向延伸、很少回头，直观展示了政策语言'
                    '持续单向演进的路径。')
    with tabs[5]:
        st.image('figures/classify_timeline.png',
                 caption='时代分类误判时间轴（步骤9：文本分类）',
                 use_container_width=True)
        st.markdown('**怎么读**：用线性SVM按语言特征预测每篇报告所属时代。'
                    '被打上红色叉号的年份是"分错"的报告，集中在1978、2009等'
                    '历史转折期——转折年份的语言本身就带有两个时代的混合特征。')
        st.image('figures/classify_cm.png',
                 caption='分类混淆矩阵', use_container_width=True)
    st.stop()

# ============ LDA互动页 ============
if page == '🕸 LDA主题互动':
    st.subheader('LDA主题模型交互可视化（pyLDAvis）')
    st.markdown(
        '这是步骤7 LDA主题模型的交互式结果：**左侧**每个气泡代表一个主题，'
        '气泡大小是该主题在全部语料中的占比，气泡间的距离反映主题相似度；'
        '**点击任意气泡**，右侧会显示该主题最具代表性的词语及权重，'
        '拖动 λ 滑块可以调整"词频—区分度"的权衡。')
    lda_path = 'figures/lda_vis.html'
    if os.path.exists(lda_path):
        with open(lda_path, encoding='utf-8') as f:
            components.html(f.read(), height=820, scrolling=True)
    else:
        st.warning('未找到 figures/lda_vis.html，请先运行 step07_lda.py。')
    st.stop()

# ============ 问答页 ============
if page == '❓ 智能问答':
    st.subheader('基于RAG的语料问答')
    st.markdown(
        '输入问题后，系统会将问题分词并转为TF-IDF向量，'
        '与全部报告段落（约1.2万段）计算余弦相似度，'
        '返回最相关的原文段落及出处年份——这就是检索增强生成（RAG）'
        '中的"检索"环节；接入大模型后即可基于这些原文生成连贯回答。')
    q = st.text_input('向历年政府工作报告提问',
                      value='科技创新方面取得了哪些成就？')
    st.caption('示例：生态文明方面的建设 / 乡村振兴的措施 / 就业政策如何变化')
    if q:
        hits = retrieve(q, top_k=3)
        if hits:
            for i, (year, text, score) in enumerate(hits, 1):
                if score is None:
                    st.markdown(f'**{i}. 【{year}年报告】**（按原文直接匹配）')
                else:
                    st.markdown(f'**{i}. 【{year}年报告】**（相关度 {score:.2f}）')
                st.info(text)
        else:
            st.warning('未检索到足够相关的内容。')
    st.stop()

# ============ 生命史漫游主页 ============
prof = get_profile(word)
if prof is None:
    st.error(f'「{word}」未在任何年份的报告中出现，换个词试试（如：'
             + '、'.join(quick) + '）')
    st.stop()

st.header(f'📖 「{word}」的生命史')
st.markdown('像查阅一个人的传记一样，从**诞生、起伏、交游、语义到自述**五个维度，'
            '还原这个政策概念在58篇政府工作报告中的完整一生。')

c1, c2, c3, c4 = st.columns(4)
c1.metric('诞生年份', f"{prof['birth_year']} 年")
c2.metric('峰值年份', f"{prof['peak_year']} 年",
          f"{prof['peak_freq']:.1f} 次/万字")
c3.metric('出现报告数', f"{len(prof['years'])} / 58 篇")
c4.metric('累计出现', f"{prof['total']} 次")

st.divider()

# ---------- 第一章：诞生 ----------
st.subheader('第一章 · 诞生')
st.markdown(f"**{prof['birth_year']}年**（{era_of(prof['birth_year'])}时期），"
            f"《政府工作报告》首次提到「{word}」：")
st.success(f"“{prof['birth_sentence']}”")
st.caption('一个词何时"出生"在报告中，往往对应着它何时进入国家政策的视野。')

# ---------- 第二章：生命曲线 ----------
st.subheader('第二章 · 生命曲线（词频涨落）')
years_all = sorted(prof['freq'])
fig, ax = plt.subplots(figsize=(11, 3.6))
freqs = [prof['freq'][y] for y in years_all]
colors = [ERA_COLORS[0 if y <= 1977 else (1 if y <= 2012 else 2)]
          for y in years_all]
ax.bar(years_all, freqs, color=colors, alpha=0.85)
ax.axvline(1977.5, color='#999', ls='--', lw=1)
ax.axvline(2012.5, color='#999', ls='--', lw=1)
ax.set_ylabel('每万字词频')
ax.spines[['top', 'right']].set_visible(False)
st.pyplot(fig)
st.caption('红=1954—1977，蓝=1978—2012，绿=2013—2026')
era_avg = [(era, np.mean([prof['freq'][y] for y in years_all if cond(y)]))
           for era, cond in ERAS
           if any(cond(y) for y in years_all)]
if len(era_avg) >= 2:
    top_era = max(era_avg, key=lambda x: x[1])
    st.markdown(f"📈 **解读**：「{word}」在 **{prof['peak_year']}年** 最活跃；"
                f"分时代看，平均使用强度最高的是 **{top_era[0]}** 时期"
                f"（每万字 {top_era[1]:.1f} 次），"
                f"说明这一概念在该阶段的政策议程中地位最重。")

# ---------- 第三章：朋友圈 ----------
st.subheader('第三章 · 朋友圈变迁（±10词窗口共现）')
st.markdown('统计该词**前后各10个词**的窗口内最高频的共现词。'
            '"和谁一起出现"记录了它的政策语境：朋友圈从行政指令词汇'
            '变成市场、法治词汇，本身就是一部微缩的改革史。')
cols = st.columns(3)
for col, (era, _) in zip(cols, ERAS):
    top = prof['partners'][era]
    with col:
        fign, axn = plt.subplots(figsize=(4.2, 4.2))
        if not top:
            axn.text(0.5, 0.5, '此时代尚未出现', ha='center', va='center',
                     fontsize=12, color='#999')
            axn.axis('off')
        else:
            G = nx.Graph()
            for w, c in top:
                G.add_edge(word, w, weight=c)
            pos = nx.spring_layout(G, k=1.8, seed=42)
            widths = [G[u][v]['weight'] / top[0][1] * 5 for u, v in G.edges]
            sizes = [1500 if n == word else 650 for n in G.nodes]
            ncs = ['#C00000' if n == word else '#E8A0A0' for n in G.nodes]
            nx.draw(G, pos, ax=axn, with_labels=True, font_size=9,
                    font_family='SimHei', node_size=sizes, node_color=ncs,
                    width=widths, edge_color='#AAAAAA', font_color='white')
        axn.set_title(era, fontsize=11)
        st.pyplot(fign)

# ---------- 第四章：语义画像 ----------
st.subheader('第四章 · 语义画像（分时代Word2Vec语义邻居）')
st.markdown('对每个时代单独训练Word2Vec模型，取出与该词**上下文最相似**的词。'
            '邻居的更替即"语义漂移"：如"市场"在1954—1977年与"投机倒把"为伍，'
            '到2013—2026年则与"产权保护"相邻——词还是同一个词，'
            '含义已经完成了时代转身。')
cols = st.columns(3)
for col, (era, _) in zip(cols, ERAS):
    m = models[era]
    with col:
        st.markdown(f'**{era}**')
        if word in m.wv:
            sims = m.wv.most_similar(word, topn=6)
            for t, s in sims:
                st.progress(max(min(s, 1.0), 0.0),
                            text=f'{t}（{s:.2f}）')
        else:
            st.markdown('*该词在此时代尚未出现或频次过低*')

st.divider()

# ---------- 第五章：生命故事（RAG） ----------
st.subheader('第五章 · 生命故事（RAG自动生成）')
st.markdown('把前面检索到的关键原文段落按"诞生—历程—朋友圈—当下"的传记结构'
            '自动组装成文：')
story = life_story(word)
for para in story['story']:
    st.write(para)
st.caption('以上故事由检索到的报告原文段落组装而成；'
           '接入大模型后可进一步润色为连贯传记。')
