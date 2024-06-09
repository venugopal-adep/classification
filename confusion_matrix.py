import streamlit as st
import plotly.graph_objects as go
import random

st.set_page_config(page_title='Confusion Matrix Demo', layout='wide')

st.title('Understanding Confusion Matrix')
st.write('In this demo, we will explore the confusion matrix in the context of a medical test for a disease.')

st.write('Imagine you are a doctor testing patients for a disease. The test can either come back positive or negative. However, no test is perfect. Sometimes it can incorrectly identify a healthy person as sick (False Positive) or a sick person as healthy (False Negative).')

tp = tn = fp = fn = 0

def generate_random_matrix():
    global tp, tn, fp, fn
    tp = random.randint(0, 100)
    tn = random.randint(0, 100)
    fp = random.randint(0, 100)
    fn = random.randint(0, 100)

if st.sidebar.button('Generate Random Confusion Matrix'):
    generate_random_matrix()

CM = [[f'FN<br>{fn}', f'TN<br>{tn}'],
      [f'TP<br>{tp}', f'FP<br>{fp}']]

fig = go.Figure(data=go.Heatmap(z=[[fn, tn], [tp, fp]], 
                                text=CM,
                                texttemplate="%{text}",
                                textfont={"size":20},
                                x=['Predicted Negative', 'Predicted Positive'],
                                y=['Actual Positive', 'Actual Negative'],
                                hoverongaps = False,
                                colorscale='Viridis'))

fig.update_layout(width=800, height=400, title_text='Confusion Matrix')                       
st.plotly_chart(fig)

total = tp + tn + fp + fn
accuracy = (tp + tn) / total if total > 0 else 0
precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

metric = st.sidebar.selectbox('Select Metric', ('Accuracy', 'Precision', 'Recall', 'F1 Score'))

if metric == 'Accuracy':
    st.subheader('Accuracy')
    st.write('Accuracy is the proportion of correct predictions (both true positives and true negatives) among the total number of cases examined.')
    st.sidebar.latex(r'''Accuracy = \frac{TP+TN}{TP+FP+FN+TN}''')
    st.sidebar.write(f'Accuracy: {accuracy:.2f}')
elif metric == 'Precision':
    st.subheader('Precision')
    st.write('Precision is the proportion of true positives among all positive predictions. It answers the question: "Of all the patients we predicted to have the disease, how many actually have it?"')
    st.sidebar.latex(r'''Precision = \frac{TP}{TP+FP}''')
    st.sidebar.write(f'Precision: {precision:.2f}')
elif metric == 'Recall':
    st.subheader('Recall')
    st.write('Recall is the proportion of true positives among all actual positives. It answers the question: "Of all the patients who actually have the disease, how many did we correctly identify?"')
    st.sidebar.latex(r'''Recall = \frac{TP}{TP+FN}''')
    st.sidebar.write(f'Recall: {recall:.2f}')
else:
    st.subheader('F1 Score')
    st.write('F1 Score is the harmonic mean of precision and recall. It provides a balanced evaluation of the model\'s performance.')
    st.sidebar.latex(r'''F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}''')
    st.sidebar.write(f'F1 Score: {f1:.2f}')