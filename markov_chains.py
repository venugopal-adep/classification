import streamlit as st
import numpy as np
import plotly.graph_objects as go
import networkx as nx

# Set page config
st.set_page_config(layout="wide", page_title="Markov Chains Explorer", page_icon="🔗")

# Custom CSS (unchanged, omitted for brevity)
st.markdown("""
<style>
    # ... (keep the existing CSS)
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🔗 Markov Chains Explorer 🔗</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the Markov Chains Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's explore Markov chains, their state transitions, and long-term behavior.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What are Markov Chains?</p>", unsafe_allow_html=True)
st.markdown("""
<p class='small-font'>
Markov chains are mathematical systems that transition from one state to another according to certain probabilistic rules. Key points:

- The probability of moving to the next state depends only on the current state (Markov property).
- Represented by a transition matrix where each element (i,j) is the probability of moving from state i to state j.
- The sum of probabilities in each row of the transition matrix must equal 1.
- Long-term behavior is described by the stationary distribution.

Markov chains have applications in physics, biology, economics, and many other fields.
</p>
""", unsafe_allow_html=True)

# Tabs with custom styling
tab1, tab2, tab3, tab4 = st.tabs(["🔄 State Transitions", "📊 Long-term Behavior", "🎲 Interactive Simulation", "🧠 Quiz"])

def plot_markov_chain(transition_matrix):
    n_states = len(transition_matrix)
    G = nx.DiGraph()

    # Add nodes
    for i in range(n_states):
        G.add_node(i)

    # Add edges
    for i in range(n_states):
        for j in range(n_states):
            if transition_matrix[i][j] > 0:
                G.add_edge(i, j, weight=transition_matrix[i][j])

    # Create positions
    pos = nx.spring_layout(G)

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=50,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # Color node points by the number of connections
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'State {node}<br># of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Markov Chain State Diagram',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    return fig

with tab1:
    st.markdown("<p class='medium-font'>Visualizing State Transitions</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Adjust the transition probabilities to see how it affects the Markov chain.
        </p>
        """, unsafe_allow_html=True)

        n_states = st.slider("Number of states", 2, 5, 3, 1)
        
        transition_matrix = []
        for i in range(n_states):
            row = []
            remaining = 1.0
            for j in range(n_states - 1):
                if j == n_states - 2:
                    prob = remaining
                else:
                    prob = st.slider(f"P(State {i} → State {j})", 0.0, remaining, remaining / (n_states - j), 0.01)
                row.append(prob)
                remaining -= prob
            row.append(remaining)
            transition_matrix.append(row)

        st.markdown("Transition Matrix:")
        st.write(np.round(transition_matrix, 2))

    with col2:
        fig = plot_markov_chain(transition_matrix)
        st.plotly_chart(fig)

with tab2:
    st.markdown("<p class='medium-font'>Long-term Behavior</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Observe how the state probabilities converge to the stationary distribution over time.
        </p>
        """, unsafe_allow_html=True)

        n_steps = st.slider("Number of steps", 1, 100, 20, 1)
        initial_state = st.selectbox("Initial state", range(n_states))

        # Calculate state probabilities over time
        state_probs = np.zeros((n_steps, n_states))
        state_probs[0, initial_state] = 1
        for i in range(1, n_steps):
            state_probs[i] = np.dot(state_probs[i-1], transition_matrix)

        # Calculate stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(np.array(transition_matrix).T)
        stationary_dist = eigenvectors[:, np.isclose(eigenvalues, 1)].real
        stationary_dist = stationary_dist / np.sum(stationary_dist)

        st.markdown("Stationary Distribution:")
        st.write(np.round(stationary_dist.flatten(), 4))

    with col2:
        fig = go.Figure()
        for i in range(n_states):
            fig.add_trace(go.Scatter(x=list(range(n_steps)), y=state_probs[:, i],
                                     mode='lines', name=f'State {i}'))
        
        fig.add_trace(go.Scatter(x=[0, n_steps-1], y=[stationary_dist[i][0], stationary_dist[i][0]],
                                 mode='lines', name=f'Stationary State {i}', line=dict(dash='dash')))

        fig.update_layout(title='State Probabilities Over Time',
                          xaxis_title='Step',
                          yaxis_title='Probability',
                          yaxis_range=[0, 1])
        
        st.plotly_chart(fig)

with tab3:
    st.markdown("<p class='medium-font'>Interactive Markov Chain Simulation</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Simulate a random walk through the Markov chain and observe the results.
        </p>
        """, unsafe_allow_html=True)

        n_simulations = st.slider("Number of simulations", 1, 1000, 100, 1)
        n_steps = st.slider("Steps per simulation", 1, 100, 20, 1, key='sim_steps')

        if st.button("Run Simulation"):
            results = np.zeros((n_simulations, n_steps))
            for i in range(n_simulations):
                state = np.random.choice(n_states)
                for j in range(n_steps):
                    results[i, j] = state
                    state = np.random.choice(n_states, p=transition_matrix[state])

            state_counts = np.sum(results == np.arange(n_states)[:, None, None], axis=(1, 2))
            empirical_dist = state_counts / (n_simulations * n_steps)

            st.markdown("Empirical Distribution:")
            st.write(np.round(empirical_dist, 4))

    with col2:
        if 'results' in locals():
            fig = go.Figure()
            for i in range(n_states):
                fig.add_trace(go.Box(y=np.sum(results == i, axis=1) / n_steps, name=f'State {i}'))

            fig.update_layout(title='Distribution of State Occupancy',
                              yaxis_title='Proportion of Time in State',
                              yaxis_range=[0, 1])
            
            st.plotly_chart(fig)

with tab4:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What is the Markov property?",
            "options": [
                "The future state depends on all past states",
                "The future state depends only on the current state",
                "The future state is always random",
                "The future state is always the same as the current state"
            ],
            "correct": 1,
            "explanation": "The Markov property states that the probability of moving to the next state depends only on the current state, not on the sequence of events that preceded it."
        },
        {
            "question": "What does the stationary distribution of a Markov chain represent?",
            "options": [
                "The initial state of the system",
                "The most common state of the system",
                "The long-term probabilities of being in each state",
                "The transition probabilities between states"
            ],
            "correct": 2,
            "explanation": "The stationary distribution represents the long-term probabilities of being in each state, regardless of the initial state, assuming the Markov chain is irreducible and aperiodic."
        },
        {
            "question": "In a transition matrix of a Markov chain, what must be true about each row?",
            "options": [
                "The sum must equal 0",
                "The sum must equal 1",
                "All elements must be positive",
                "All elements must be less than 1"
            ],
            "correct": 1,
            "explanation": "In a transition matrix, each row represents the probabilities of transitioning from one state to all possible states. These probabilities must sum to 1, as they represent all possible outcomes."
        }
    ]

    score = 0
    for i, q in enumerate(questions):
        st.markdown(f"<p class='small-font'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
        user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
        
        if st.button("Check Answer", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! 🎉")
                score += 1
            else:
                st.error("Incorrect. Try again! 🤔")
            st.info(q['explanation'])
        st.markdown("---")

    if st.button("Show Final Score"):
        st.markdown(f"<p class='big-font'>Your score: {score}/{len(questions)}</p>", unsafe_allow_html=True)
        if score == len(questions):
            st.balloons()

# Conclusion
st.markdown("<p class='big-font'>Congratulations! 🎊</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>You've explored Markov chains through interactive examples and simulations. These concepts are crucial in understanding stochastic processes and have wide-ranging applications in various fields. Keep exploring and applying these concepts in different scenarios!</p>", unsafe_allow_html=True)