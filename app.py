import numpy as np
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib

from cognitive_modelling import CognitiveModelling

st.set_page_config(layout="wide", page_title="Cognitive Modelling")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Cognitive Modelling')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

full_file = st.file_uploader("Choose a file with your data for cognitive modelling in .xlsx format",
                                 type=[".xls", ".xlsx"])

if full_file is not None:

    matrix = pd.read_excel(full_file, header=0, index_col=0).to_numpy()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Check dimensions of the cognitive matrix!")
    cmds = CognitiveModelling(matrix)

    col_01, col_02 = st.columns(2)

    with col_01:
        st.write("### Static plot")
        st.pyplot(cmds.show_graph_static())

    with col_02:
        st.write("### Interactive plot")
        cmds.show_graph_html()
        with open("network.html", "r", encoding="utf-8") as f:
            source_code_html = f.read()
            components.html(source_code_html, height=700)

    st.write("## Stability Analysis")
    odd_cycles = cmds.check_system_stability_structural()

    st.write("#### The system is **structurally unstable**" if len(odd_cycles)
             else "#### The system is **structurally stable**")

    if len(odd_cycles):
        with st.expander(f"See odd cycles (total number: {len(odd_cycles)})"):
            st.write(["->".join(elem.astype(str)) for elem in odd_cycles])

    egnvl, rho = cmds.check_system_stability_numerical()
    col_lat, col_stab = st.columns(2)
    with col_lat:
        st.write("#### Eigenvalues")
        for i, elem in enumerate(egnvl):
            st.latex(f"\\lambda_{i+1} = {round(np.real(elem), 4)} {'+' if round(np.imag(elem),4) >= 0 else ''} {round(np.imag(elem),4)}\\cdot i")
    with col_stab:
        st.write("#### Spectral radius of matrix: ")
        st.latex(f"\\rho(A) = {round(rho, 4)}")
        st.write(f"The system is **{'not ' if rho >= 1 else ''}stable** in terms of value")
        st.write(f"The system is **{'not ' if rho > 1 else ''}stable** in terms of disturbance")

    st.write("## Impulse simulation")
    q_vec = []
    col_mod_1, col_mod_2 = st.columns([1,4])
    with col_mod_1:
        for i in range(cmds.cognitive_matrix.shape[0]):
            q_vec.append(st.number_input(f"q_{i+1}", value=1 if not i else 0))
        t = st.number_input("Number of iterations", value=10)
    with col_mod_2:
        start_modelling = st.button("Start modelling")
        if start_modelling:
            with plt.style.context("seaborn-whitegrid", "seaborn"):
                model_output = cmds.impulse_modelling(t, np.array(q_vec))
                fig, ax = plt.subplots(figsize=(12,8))
                ax.plot(model_output, "-o")
                ax.legend([f"$x_{i + 1}$" for i in range(model_output.shape[0])])
                ax.set_xlabel("$t$")
                ax.set_ylabel("$X_t$")
                st.pyplot(fig)
