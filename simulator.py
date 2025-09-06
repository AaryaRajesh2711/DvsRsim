import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from collections import Counter
import math

st.set_page_config(page_title="DNA vs RNA Environment Simulator", layout="wide")

# --------------------------- Helper physics/chem functions ---------------------------
R = 8.31446261815324  # J/mol/K

@st.cache_data
def arrhenius_rate(A, Ea, T):
    return A * np.exp(-Ea / (R * T))

def ph_factor_rna(pH):
    alpha = 0.12
    return 1.0 + alpha * (pH - 5.0) ** 2

def dna_ph_factor(pH):
    alpha = 0.02
    return 1.0 + alpha * (pH - 7.0) ** 2

def uv_damage_rate(uv_flux, cross_section=1e-20):
    return uv_flux * cross_section

# --------------------------- Population & replication ---------------------------

def mutate_sequence(seq, mu, alphabet):
    seq_list = list(seq)
    for i in range(len(seq_list)):
        if np.random.rand() < mu:
            choices = [b for b in alphabet if b != seq_list[i]]
            seq_list[i] = np.random.choice(choices)
    return ''.join(seq_list)

def replicate_population(pop_counter, copy_rate, mu, carry_capacity, alphabet):
    new_pop = Counter()
    for seq, count in pop_counter.items():
        if count > 0:
            offspring = np.random.poisson(copy_rate, size=count).sum()
        else:
            offspring = 0
        for _ in range(offspring):
            mutated = mutate_sequence(seq, mu, alphabet)
            new_pop[mutated] += 1
    if sum(new_pop.values()) > carry_capacity:
        items = list(new_pop.items())
        seqs, counts = zip(*items)
        probs = np.array(counts) / sum(counts)
        sampled = np.random.multinomial(carry_capacity, probs)
        capped = Counter({s: int(c) for s, c in zip(seqs, sampled) if c > 0})
        return capped
    return new_pop

# --------------------------- Metrics ---------------------------

def shannon_entropy(pop_counter):
    total = sum(pop_counter.values())
    if total == 0:
        return 0.0
    freqs = np.array([v / total for v in pop_counter.values()])
    return -np.sum(freqs * np.log2(freqs + 1e-12))

# --------------------------- Simulation engine ---------------------------

def simulate(env, seq0, polymer_type, params, sim_steps=100, record_every=1, energy_source=None):
    T = env['T']
    pH = env['pH']
    uv_flux = env['uv_flux']

    k_h_base = arrhenius_rate(params['A_hyd'], params['Ea_hyd'], T)
    if polymer_type == 'RNA':
        k_h = k_h_base * ph_factor_rna(pH)
    else:
        k_h = k_h_base * dna_ph_factor(pH)

    k_uv_site = uv_damage_rate(uv_flux, params.get('cross_section', 1e-20))
    seq_len = len(seq0)
    k_uv = k_uv_site * seq_len

    # Energy pulses modifier
    energy_multiplier = 1.0
    if energy_source == "Lightning":
        energy_multiplier = 1.5
    elif energy_source == "Geothermal":
        energy_multiplier = 1.2
    elif energy_source == "Cosmic radiation":
        energy_multiplier = 2.0

    k_h *= energy_multiplier
    k_uv *= energy_multiplier

    pop = Counter({seq0: params.get('N0', 100)})
    records = []

    for t in range(sim_steps + 1):
        if t % record_every == 0:
            total = sum(pop.values())
            master_freq = pop.get(seq0, 0) / total if total > 0 else 0
            entropy = shannon_entropy(pop)
            mean_len = np.mean([len(s) for s in pop.elements()]) if total > 0 else 0
            records.append({'t': t, 'total': total, 'master_freq': master_freq, 'entropy': entropy, 'mean_len': mean_len, 'k_h': k_h, 'k_uv': k_uv})

        if sum(pop.values()) == 0:
            break
        survive_prob = math.exp(-(k_h + k_uv))
        survived = Counter()
        for seq, count in pop.items():
            survivors = np.random.binomial(count, survive_prob)
            if survivors > 0:
                survived[seq] = survivors
        pop = survived

        pop = replicate_population(pop, params.get('copy_rate', 1.0), params.get('mu_base', 0.01), params.get('carry_capacity', 1000), params.get('alphabet', 'ACGT' if polymer_type=='DNA' else 'ACGU'))

    df = pd.DataFrame(records)
    return df, pop

# --------------------------- Streamlit UI ---------------------------

st.title("DNA vs RNA Planetary Environment Simulator")
st.write("Explore nucleic-acid stability and evolutionary potential across environments.")

with st.sidebar:
    st.header("Environment")
    # Quick planet presets
    planet_preset = st.radio("Choose Planet Preset", ["Custom", "Mars surface", "Europa ocean", "Titan lake", "Hot spring", "Deep space"])

    if planet_preset == "Mars surface":
        T_c, pH, uv_flux = -30, 7.0, 5e12
    elif planet_preset == "Europa ocean":
        T_c, pH, uv_flux = -5, 8.0, 1e8
    elif planet_preset == "Titan lake":
        T_c, pH, uv_flux = -179, 6.5, 1e6
    elif planet_preset == "Hot spring":
        T_c, pH, uv_flux = 90, 5.0, 1e11
    elif planet_preset == "Deep space":
        T_c, pH, uv_flux = -270, 7.0, 1e3
    else:
        T_c = st.slider("Temperature (°C)", min_value=-50.0, max_value=150.0, value=25.0)
        pH = st.slider("pH", min_value=0.0, max_value=14.0, value=7.0)
        uv_flux = st.number_input("UV flux (photons/cm^2/s)", min_value=0.0, value=1e12, format="%.0f")

    st.header("Energy source")
    energy_source = st.selectbox("Energy pulses", ["None", "Lightning", "Geothermal", "Cosmic radiation"])

    st.header("Polymer & sequence")
    polymer_type = st.selectbox("Polymer type", ['RNA','DNA'])
    seq_choice = st.radio("Initial sequence", ['Random','Custom'])
    seq_len = st.slider("Sequence length (for random)", 4, 60, 12)
    if seq_choice == 'Custom':
        seq0 = st.text_input("Enter sequence (A/C/G/T/U)", value='A'*12)
    else:
        alphabet = 'ACGU' if polymer_type=='RNA' else 'ACGT'
        seq0 = ''.join(np.random.choice(list(alphabet), seq_len))

    st.header("Replication & population")
    copy_rate = st.number_input("Copy rate (mean offspring per cycle)", min_value=0.0, value=1.0)
    mu_base = st.number_input("Per-base replication error (mu)", min_value=0.0, max_value=1.0, value=0.01, step=0.001)
    carry_capacity = st.number_input("Carrying capacity", min_value=10, value=1000)
    N0 = st.number_input("Initial population (N0)", min_value=1, value=100)

    st.header("Chemical kinetics (hydrolysis)")
    A_hyd = st.number_input("A (s^-1) prefactor", value=1e6, format="%.5g")
    Ea_kJ = st.number_input("Ea (kJ/mol)", value=120.0)
    Ea_hyd = Ea_kJ * 1000.0

    st.header("UV damage")
    cross_section = st.number_input("UV cross-section (cm^2 per site)", value=1e-20, format="%.0e")

    st.header("Simulation")
    steps = st.number_input("Generations / cycles", min_value=1, value=200)
    record_every = st.number_input("Record every N steps", min_value=1, value=1)

    run = st.button("Run simulation")

# Assemble env and params
env = {'T': T_c + 273.15, 'pH': pH, 'uv_flux': uv_flux}
params = {'A_hyd': A_hyd, 'Ea_hyd': Ea_hyd, 'cross_section': cross_section,
          'copy_rate': copy_rate, 'mu_base': mu_base, 'carry_capacity': carry_capacity,
          'N0': N0, 'alphabet': 'ACGU' if polymer_type=='RNA' else 'ACGT'}

if run:
    with st.spinner("Simulating..."):
        df, pop = simulate(env, seq0, polymer_type, params, sim_steps=int(steps), record_every=int(record_every), energy_source=energy_source if energy_source!="None" else None)

    col1, col2 = st.columns((2,1))
    with col1:
        st.subheader("Time series")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['master_freq'], name='Master seq freq'))
        fig.add_trace(go.Scatter(x=df['t'], y=df['entropy'], name='Shannon entropy'))
        fig.update_layout(xaxis_title='Generation', yaxis_title='Value')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Heatmap: predicted half-life (T vs pH)")
        Ts = np.linspace(max(0, T_c-60), T_c+60, 60)
        pHs = np.linspace(0,14,60)
        half_grid = np.zeros((len(pHs), len(Ts)))
        for i, phv in enumerate(pHs):
            for j, Tv in enumerate(Ts):
                k_base = arrhenius_rate(A_hyd, Ea_hyd, Tv+273.15)
                if polymer_type == 'RNA':
                    k_eff = k_base * ph_factor_rna(phv)
                else:
                    k_eff = k_base * dna_ph_factor(phv)
                if k_eff <= 0: half_grid[i,j] = np.nan
                else:
                    half_grid[i,j] = np.log10(np.log(2)/k_eff)
        heat_df = pd.DataFrame(half_grid, index=np.round(pHs,2), columns=np.round(Ts,2))
        hm = px.imshow(heat_df, labels=dict(x='T (°C)', y='pH', color='log10 half-life (s)'),
                       x=heat_df.columns, y=heat_df.index, aspect='auto')
        st.plotly_chart(hm, use_container_width=True)

        st.subheader("Population composition (top sequences)")
        top = pop.most_common(25)
        if top:
            pop_df = pd.DataFrame(top, columns=['sequence','count'])
            bar = px.bar(pop_df, x='sequence', y='count')
            st.plotly_chart(bar, use_container_width=True)
        else:
            st.write("No population left (extinct)")

    with col2:
        st.subheader("Final summary")
        st.write(f"Total population: {sum(pop.values())}")
        st.write(f"Master sequence ({seq0}) count: {pop.get(seq0,0)}")
        st.write(f"Shannon entropy: {shannon_entropy(pop):.3f}")

        csv = df.to_csv(index=False).encode()
        st.download_button("Download time series CSV", csv, file_name='simulation_timeseries.csv', mime='text/csv')

        pop_df_export = pd.DataFrame(pop.items(), columns=['sequence','count'])
        st.download_button("Download final population CSV", pop_df_export.to_csv(index=False).encode(), file_name='final_population.csv')

    st.success("Simulation finished")

st.markdown("---")
st.subheader("Notes & presets")
st.markdown("""
- Quick presets available for Mars, Europa, Titan, hot springs, and deep space.
- Energy pulses (lightning, geothermal, cosmic radiation) can modify degradation and replication rates.
- Use custom inputs or upload exoplanet data for further exploration.
""")
