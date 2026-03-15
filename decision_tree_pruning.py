"""
Decision Tree — Diabetes vs No Diabetes
A relatable 2-class example with step-by-step calculations.
Developed by: Venugopal Adep
"""
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd

DIABETES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

st.set_page_config(layout="wide", page_title="Decision Tree: Diabetes Prediction", page_icon="🩺")

st.markdown("""
<style>
    .main-header { font-size: 42px !important; font-weight: bold; color: #4B0082; text-align: center; margin-bottom: 30px; }
    .tab-subheader { font-size: 28px !important; font-weight: bold; color: #8A2BE2; margin-top: 20px; margin-bottom: 20px; }
    .content-text { font-size: 18px !important; line-height: 1.6; }
    .highlight { background-color: #E6E6FA; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    .relatable { background: linear-gradient(135deg, #f5f0ff 0%, #e8f4fd 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #8A2BE2; margin: 15px 0; }
    .quiz-question { background-color: #F0E6FA; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid #8A2BE2; }
    .explanation { background-color: #E6F3FF; padding: 10px; border-radius: 5px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🩺 Decision Tree: Diabetes or No Diabetes? 🩺</h1>", unsafe_allow_html=True)
st.markdown("<p class='content-text'><strong>Predict if a patient has diabetes using health measures (glucose, BMI, age, etc.). Simple, relatable, with step-by-step calculations. — Venugopal Adep</strong></p>", unsafe_allow_html=True)


# ─── Formulas for 2-class classification ───────────────────────────────────
def gini_impurity(p0, p1):
    """G = 1 - (p₀² + p₁²)"""
    return 1 - (p0**2 + p1**2)


def entropy(p0, p1):
    """H = -(p₀·log₂(p₀) + p₁·log₂(p₁)), with 0·log(0)=0"""
    def safe_log(p):
        return 0 if p <= 0 else p * np.log2(p)
    return -(safe_log(p0) + safe_log(p1))


def get_node_calculations(clf, class_names, feature_names):
    """Extract per-node calculations for 2-class tree (Gini, Entropy, Information Gain)."""
    tree = clf.tree_
    calculations = []
    for node_id in range(tree.node_count):
        value = tree.value[node_id][0]
        n_total = value.sum()
        n0, n1 = value[0], value[1]
        p0 = n0 / n_total if n_total > 0 else 0
        p1 = n1 / n_total if n_total > 0 else 0
        gini = gini_impurity(p0, p1)
        ent = entropy(p0, p1)
        impurity = tree.impurity[node_id]
        is_leaf = tree.children_left[node_id] == -1
        split_feature_idx = tree.feature[node_id]
        feat_name = feature_names[split_feature_idx] if split_feature_idx >= 0 else None
        split_threshold = tree.threshold[node_id]
        left_id = tree.children_left[node_id]
        right_id = tree.children_right[node_id]
        n_left = tree.n_node_samples[left_id] if left_id >= 0 else 0
        n_right = tree.n_node_samples[right_id] if right_id >= 0 else 0
        imp_left = tree.impurity[left_id] if left_id >= 0 else 0
        imp_right = tree.impurity[right_id] if right_id >= 0 else 0
        info_gain = impurity - ((n_left / n_total) * imp_left + (n_right / n_total) * imp_right) if not is_leaf else 0
        calculations.append({
            "node_id": node_id, "n0": int(n0), "n1": int(n1), "n_total": int(n_total),
            "p0": p0, "p1": p1, "gini": gini, "entropy": ent, "impurity": impurity,
            "is_leaf": is_leaf, "split_feature": feat_name, "split_threshold": split_threshold,
            "class_names": class_names, "n_left": n_left, "n_right": n_right,
            "imp_left": imp_left, "imp_right": imp_right, "info_gain": info_gain,
        })
    return calculations


def plot_decision_tree(clf, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names, ax=ax)
    st.pyplot(fig)


# ─── Sidebar: Single source of parameters (shared by Tree & Calculations) ───
with st.sidebar:
    st.header("🩺 Diabetes Prediction")
    st.markdown("**Dataset:** Pima Indians Diabetes (768 patients)")
    with st.expander("What do these features mean?"):
        st.caption("**Glucose** = blood sugar · **BMI** = body mass index · **BloodPressure** = diastolic · **Insulin** = 2-h serum insulin · **Pedigree** = diabetes family history")
    st.markdown("**Pruning parameters**")
    criterion = st.selectbox("Splitting criterion", ['gini', 'entropy'])
    max_depth = st.slider("Maximum depth", 1, 10, 4)
    min_samples_split = st.slider("Minimum samples split", 2, 20, 2)
    min_samples_leaf = st.slider("Minimum samples leaf", 1, 20, 1)
    ccp_alpha = st.slider("ccp_alpha", 0.0, 0.1, 0.0, 0.001)
    with st.expander("📋 What do these features mean?"):
        st.caption("**Glucose** = blood sugar level | **BMI** = body mass index | **Blood Pressure** = diastolic (mm Hg) | **Insulin** = 2-hour serum insulin | **Age** = years | **Pregnancies** = number | **SkinThickness** = triceps (mm) | **Pedigree** = diabetes pedigree function")

# Load Diabetes data (Pima Indians: Glucose, BMI, Age, etc. → Diabetes or No Diabetes)
try:
    df = pd.read_csv(DIABETES_URL, header=None,
                    names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age', 'Outcome'])
except Exception:
    # Fallback: synthetic diabetes-like data if URL fails
    from sklearn.datasets import make_classification
    X_syn, y_syn = make_classification(n_samples=768, n_features=8, n_informative=5, n_redundant=2, random_state=42)
    feature_names = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin', 'SkinThickness', 'Pedigree', 'Pregnancies']
    X, y = X_syn, y_syn
    class_names = np.array(['No Diabetes', 'Diabetes'])
else:
    X = df.drop(columns='Outcome').values
    y = df['Outcome'].values
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']
    class_names = np.array(['No Diabetes', 'Diabetes'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha)
clf.fit(X_train, y_train)


# ─── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎓 Learn", "🌳 2-Class Tree", "📐 Calculations", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Learn: Diabetes or No Diabetes?</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='relatable'>
    <p class='content-text'>
    <b>Imagine a clinic</b> with 768 patients. For each patient we have: <b>Glucose</b> (blood sugar), <b>BMI</b>, <b>Age</b>, <b>Blood Pressure</b>, <b>Insulin</b>, and a few more health measures.
    The question: <b>Does this patient have diabetes?</b> (Yes or No — a 2-class prediction.)
    </p>
    </div>
    <p class='content-text'>
    A <b>decision tree</b> learns rules like: <i>"If Glucose > 127 and BMI > 29, then likely Diabetes."</i> It asks yes/no questions to split patients into groups that become more and more similar.
    </p>
    <p class='content-text'>
    <b>Key formulas (2 classes — Diabetes vs No Diabetes):</b><br>
    • <b>Gini impurity</b>: G = 1 − (p₀² + p₁²) &nbsp; — how mixed is this group? (0 = all same class)<br>
    • <b>Entropy</b>: H = −(p₀·log₂(p₀) + p₁·log₂(p₁)) &nbsp; — information content<br>
    • <b>Information Gain</b>: How much does a split clean up the mix? = Impurity(before) − weighted Impurity(after)
    </p>
    <p class='content-text'>
    The tree picks the question (split) that <b>maximizes information gain</b>. Open the <b>📐 Calculations</b> tab to see the exact math for each node.
    </p>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 class='tab-subheader'>🌳 Diabetes Prediction Tree</h2>", unsafe_allow_html=True)
    st.info("💡 Each node asks a question (e.g. Glucose ≤ 127?). Adjust parameters in the **sidebar** to see how the tree changes.")

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.markdown(f"""
    <div class='highlight'>
    <p class='content-text'>
    <strong>Accuracy:</strong> {accuracy:.2f} | <strong>Precision:</strong> {precision:.2f} |
    <strong>Recall:</strong> {recall:.2f} | <strong>F1 Score:</strong> {f1:.2f}
    </p>
    </div>
    """, unsafe_allow_html=True)

    plot_decision_tree(clf, feature_names, class_names)

    fi_df = pd.DataFrame({'feature': feature_names, 'importance': clf.feature_importances_}).sort_values('importance', ascending=False)
    fig = px.bar(fi_df, x='importance', y='feature', orientation='h', title='Which health measures matter most? (Feature Importance)')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<h2 class='tab-subheader'>📐 Step-by-Step Calculations (Diabetes vs No Diabetes)</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='content-text'>
    At each node we count: <i>How many patients have Diabetes? How many don't?</i> Then we compute impurity and information gain.<br>
    <b>Formulas:</b> Gini G = 1 − (p₀² + p₁²) &nbsp; | &nbsp; Entropy H = −(p₀·log₂(p₀) + p₁·log₂(p₁)) &nbsp; | &nbsp; Info Gain = Impurity(parent) − weighted Impurity(children)
    </p>
    """, unsafe_allow_html=True)

    calcs = get_node_calculations(clf, class_names, feature_names)
    for c in calcs:
        c0, c1 = c['class_names'][0], c['class_names'][1]
        title = f"Node {c['node_id']}: {c['n0']} {c0} + {c['n1']} {c1} = {c['n_total']} samples"
        if not c['is_leaf']:
            title += f" → Split: {c['split_feature']} ≤ {c['split_threshold']:.3f}"
        else:
            title += " (Leaf)"
        with st.expander(title):
            st.markdown(f"**1. Class counts:** {c0}={c['n0']}, {c1}={c['n1']}")
            st.markdown(f"**2. Proportions:** p₀ = {c['n0']}/{c['n_total']} = {c['p0']:.4f}, &nbsp; p₁ = {c['n1']}/{c['n_total']} = {c['p1']:.4f}")
            st.markdown(f"**3. Gini:** G = 1 − (p₀² + p₁²) = 1 − ({c['p0']:.4f}² + {c['p1']:.4f}²) = **{c['gini']:.4f}**")
            st.markdown(f"**4. Entropy:** H = −(p₀·log₂(p₀) + p₁·log₂(p₁)) = **{c['entropy']:.4f}**")
            if not c['is_leaf']:
                w_left = c['n_left'] / c['n_total']
                w_right = c['n_right'] / c['n_total']
                st.markdown(f"**5. Information Gain:** IG = {c['impurity']:.4f} − ({w_left:.3f}×{c['imp_left']:.4f} + {w_right:.3f}×{c['imp_right']:.4f}) = **{c['info_gain']:.4f}**")

with tab4:
    st.markdown("<h2 class='tab-subheader'>🧠 Quiz</h2>", unsafe_allow_html=True)
    questions = [
        {"q": "In the diabetes example, if a node has 80 No Diabetes and 20 Diabetes patients, what does a low Gini impurity mean?", "opts": ["The group is very mixed", "The group is mostly one class (fairly pure)", "We need more data", "Gini doesn't apply here"], "correct": 1, "exp": "Low Gini = mostly one class. Here 80% No Diabetes means the node is fairly pure."},
        {"q": "For Gini: G = 1 − (p₀² + p₁²). When is G = 0?", "opts": ["Never", "When both classes are 50-50", "When the node is pure (all one class)", "When p₀ = p₁"], "correct": 2, "exp": "G = 0 when the node is pure — e.g. all patients have Diabetes or all don't."},
        {"q": "What does the tree try to maximize when choosing a split (e.g. Glucose ≤ 127)?", "opts": ["Tree depth", "Information Gain", "Number of leaves", "Number of patients"], "correct": 1, "exp": "Information Gain = how much the split cleans up the mix. Higher = better split."},
    ]
    score = 0
    for i, q in enumerate(questions):
        st.markdown(f"<div class='quiz-question'><p class='content-text'><strong>Q{i+1}:</strong> {q['q']}</p></div>", unsafe_allow_html=True)
        ans = st.radio("Choose:", q['opts'], key=f"q{i}")
        if st.button("Check", key=f"c{i}"):
            if q['opts'].index(ans) == q['correct']:
                st.success("Correct!")
                score += 1
            else:
                st.error("Incorrect.")
            st.info(q['exp'])
        st.markdown("---")
    if st.button("Show Final Score"):
        st.markdown(f"**Score: {score}/{len(questions)}**")
