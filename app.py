import streamlit as st
import os
import subprocess
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import threading
import queue
import time
import io
import zipfile
import math
import numpy as np
import re

import py3Dmol
import streamlit.components.v1 as components
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


POTCAR_PATH_FILE = "potcar_path.txt"
VASP_COMMAND_FILE = "vasp_command.txt"



def create_potcar(structure, vasp_potentials_folder):
    if not vasp_potentials_folder or not os.path.isdir(vasp_potentials_folder):
        st.error("VASP potentials directory is not set or does not exist.")
        return None

    elements = [site.specie.symbol for site in structure]
    unique_elements = sorted(set(elements), key=elements.index)

    potcar_content = []
    st.session_state.generated_files['potcar_details'] = []

    for element in unique_elements:
        potential_found = False
        potential_paths_to_try = [
            os.path.join(vasp_potentials_folder, element),
            os.path.join(vasp_potentials_folder, f"{element}_sv"),
            os.path.join(vasp_potentials_folder, f"{element}_pv"),
            os.path.join(vasp_potentials_folder, f"{element}_sv_GW")
        ]
        for pot_dir in potential_paths_to_try:
            potcar_path = os.path.join(pot_dir, 'POTCAR')
            if os.path.exists(potcar_path):
                with open(potcar_path, 'r') as f:
                    potcar_content.append(f.read())
                potential_found = True
                st.session_state.generated_files['potcar_details'].append(
                    f"✓ Found potential for '{element}' at: {pot_dir}")
                break
        if not potential_found:
            error_message = f"✗ Potential for element '{element}' not found in the specified directory or its common variants (_sv, _pv)."
            st.error(error_message)
            st.session_state.generated_files['potcar_details'].append(error_message)
            return None
    return "".join(potcar_content)


def create_incar(encut):
    return f"""SYSTEM = Convergence Test
PREC   = Accurate
ENCUT  = {encut}
ISMEAR = 0
SIGMA  = 0.05
IBRION = -1
NELM   = 150
LWAVE  = .FALSE.
LCHARG = .FALSE.
"""


def create_kpoints(kx, ky, kz):
    return f"""Automatic K-Mesh
0
Gamma
{kx} {ky} {kz}
0 0 0
"""


def view_structure(structure, height=400, width=400):
    cif_writer = CifWriter(structure)
    cif_string = str(cif_writer)

    view = py3Dmol.view(width=width, height=height)
    view.addModel(cif_string, 'cif')
    view.setStyle({'sphere': {'scale': 0.3}})

    view.addUnitCell()
    view.zoomTo()

    return view._make_html()


def get_kpoints_from_kspacing(structure, k_spacing):
    if not structure or k_spacing <= 0:
        return 0, 0, 0
    recip_lattice = structure.lattice.reciprocal_lattice
    a_star, b_star, c_star = recip_lattice.abc

    ka = a_star / k_spacing
    kb = b_star / k_spacing
    kc = c_star / k_spacing

    ka = int(ka) + 1 if (ka - int(ka) > 0.4) else int(ka)
    kb = int(kb) + 1 if (kb - int(kb) > 0.4) else int(kb)
    kc = int(kc) + 1 if (kc - int(kc) > 0.4) else int(kc)

    return max(1, ka), max(1, kb), max(1, kc)




def run_vasp_in_thread(calc_params, work_dir, log_queue, stop_event):
    mode = calc_params['mode']
    command = calc_params['command']
    incar_template = calc_params['incar']

    output_filename = f"{mode.upper()}_CONVERGENCE_TEST.txt"
    output_file_path = os.path.join(work_dir, output_filename)

    header = ("# ENCUT[eV] Total_Energy[eV] Time/step(min)\n" if mode == 'encut'
              else "# k_spacing[A^-1] k_a k_b k_c Total_Energy[eV] Time/step(min)\n")

    try:
        with open(output_file_path, 'w') as f:
            f.write(header)

        if mode == 'encut':
            loop_values = range(calc_params['max'], calc_params['min'] + (-1 if calc_params['step'] < 0 else 1),
                                calc_params['step'])
        else: 
            loop_values = np.arange(calc_params['start'], calc_params['end'] + calc_params['step'], calc_params['step'])

        previous_k_grid = None

        for i, val in enumerate(loop_values):
            if stop_event.is_set():
                log_queue.put("--- Calculation stopped by user. ---")
                break

            if mode == 'encut':
                encut = val
                log_queue.put({'type': 'progress', 'step': i, 'value': val})
                log_queue.put(f"--- Starting calculation for ENCUT = {encut} eV ---")
                incar_content = re.sub(r"ENCUT\s*=\s*\d+", f"ENCUT = {encut}", incar_template)
                with open(os.path.join(work_dir, "INCAR"), "w") as f:
                    f.write(incar_content)
            else:  
                k_spacing = val
                ka, kb, kc = get_kpoints_from_kspacing(calc_params['structure'], k_spacing)
                current_k_grid = (ka, kb, kc)
                log_queue.put({'type': 'progress', 'step': i, 'value': val, 'k_grid': f"{ka}x{kb}x{kc}"})

                if current_k_grid == previous_k_grid:
                    log_queue.put(
                        f"--- Skipping k-spacing {k_spacing:.4f} Å⁻¹ as k-grid {ka}x{kb}x{kc} is unchanged. ---")
                    continue
                previous_k_grid = current_k_grid

                log_queue.put(f"--- Starting calculation for k-spacing {k_spacing:.4f} Å⁻¹ (grid: {ka}x{kb}x{kc}) ---")
                kpoints_content = create_kpoints(ka, kb, kc)
                with open(os.path.join(work_dir, "KPOINTS"), "w") as f:
                    f.write(kpoints_content)
                with open(os.path.join(work_dir, "INCAR"), "w") as f:
                    f.write(incar_template)  

            if os.path.exists(os.path.join(work_dir, "WAVECAR")):
                os.remove(os.path.join(work_dir, "WAVECAR"))

            start_time = time.time()
            process = subprocess.Popen(command, shell=True, cwd=work_dir,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in iter(process.stdout.readline, ''):
                log_queue.put(line.strip())
            process.stdout.close()
            return_code = process.wait()
            end_time = time.time()
            elapsed_time_min = (end_time - start_time) / 60.0

            if return_code != 0:
                log_queue.put(f"ERROR: VASP calculation failed with return code {return_code}.")
                break

            total_energy = None
            with open(os.path.join(work_dir, "OUTCAR"), 'r') as outcar:
                for line in outcar:
                    if "free  energy   TOTEN" in line:
                        total_energy = float(line.split()[-2])

            if total_energy:
                log_queue.put(f"SUCCESS: Energy = {total_energy:.5f} eV, Time = {elapsed_time_min:.2f} min")
                data_point = {'type': 'data', 'energy': total_energy, 'time': elapsed_time_min}
                if mode == 'encut':
                    data_point['encut'] = val
                    line_to_write = f"{val} {total_energy} {elapsed_time_min}\n"
                else:
                    data_point.update({'k_spacing': val, 'k_grid': f"{ka}x{kb}x{kc}"})
                    line_to_write = f"{val} {ka} {kb} {kc} {total_energy} {elapsed_time_min}\n"

                with open(output_file_path, 'a') as f:
                    f.write(line_to_write)
                log_queue.put(data_point)
            else:
                log_queue.put(f"ERROR: Could not parse total energy from OUTCAR.")
                break

        if not stop_event.is_set():
            log_queue.put("--- All calculations finished ---")
    except Exception as e:
        log_queue.put(f"An error occurred in the calculation thread: {e}")
    finally:
        log_queue.put("THREAD_FINISHED")



st.set_page_config(page_title="VASP Convergence Workflow", layout="wide")
st.title("VASP Convergence Workflow")


default_potcar_path = ""
if os.path.exists(POTCAR_PATH_FILE):
    with open(POTCAR_PATH_FILE, 'r') as f:
        default_potcar_path = f.read().strip()

default_vasp_command = "mpirun -np 4 vasp_gpu_std"
if os.path.exists(VASP_COMMAND_FILE):
    with open(VASP_COMMAND_FILE, 'r') as f:
        default_vasp_command = f.read().strip()

if 'calculation_running' not in st.session_state:
    st.session_state.calculation_running = False
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'vasp_potentials_path' not in st.session_state:
    st.session_state.vasp_potentials_path = default_potcar_path
if 'vasp_command' not in st.session_state:
    st.session_state.vasp_command = default_vasp_command
if 'results_list' not in st.session_state:
    st.session_state.results_list = []
if 'generated_files' not in st.session_state:
    st.session_state.generated_files = {}
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if 'structure' not in st.session_state:
    st.session_state.structure = None
if 'work_dir' not in st.session_state:
    st.session_state.work_dir = os.getcwd()
if 'progress_text' not in st.session_state:
    st.session_state.progress_text = ""
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'total_steps' not in st.session_state:
    st.session_state.total_steps = 0
if 'incar_content' not in st.session_state:
    st.session_state.incar_content = create_incar(520)


with st.sidebar:
    st.header("1. VASP Potentials Path")
    st.session_state.vasp_potentials_path = st.text_input("Path", st.session_state.vasp_potentials_path,
                                                          label_visibility="collapsed")
    if st.button("💾 Save Path Permanently"):
        with open(POTCAR_PATH_FILE, 'w') as f:
            f.write(st.session_state.vasp_potentials_path)
        st.success("Path saved.")

    st.header("2. VASP Execution Command")
    st.session_state.vasp_command = st.sidebar.text_input("VASP execution command", st.session_state.vasp_command,
                                                          label_visibility="collapsed")
    if st.sidebar.button("💾 Save Command Permanently"):
        with open(VASP_COMMAND_FILE, 'w') as f:
            f.write(st.session_state.vasp_command)
        st.sidebar.success("Command saved.")

    st.header("3. Upload Structure (Optional)")
    st.info("If a `POSCAR` file isn't found in your project folder, upload one here.")
    uploaded_poscar_sidebar = st.file_uploader("Upload", type=['POSCAR', 'vasp', 'contcar'],
                                               label_visibility="collapsed")


tab1, tab2, tab3 = st.tabs(["➡️ Run Workflow", "🖥️ Live Console", "📊 Live Results"])

with tab1:
    st.header("1. Define Working Directory")
    st.session_state.work_dir = st.text_input("Project Folder Path", value=st.session_state.work_dir)

    auto_poscar_path = os.path.join(st.session_state.work_dir, "POSCAR")
    if os.path.exists(auto_poscar_path):
        try:
            st.session_state.structure = Structure.from_file(auto_poscar_path)
            if "last_loaded" not in st.session_state or st.session_state.last_loaded != auto_poscar_path:
                st.success(f"Loaded `POSCAR` from `{st.session_state.work_dir}`.")
                #st.session_state.generated_files = {}
                st.session_state.last_loaded = auto_poscar_path
                #st.rerun()
        except Exception as e:
            st.error(f"Error parsing `POSCAR`: {e}")
            st.session_state.structure = None
    elif uploaded_poscar_sidebar:
        try:
            poscar_content = uploaded_poscar_sidebar.getvalue().decode("utf-8")
            st.session_state.structure = Structure.from_str(poscar_content, fmt="poscar")
            st.success(f"Loaded structure from: `{uploaded_poscar_sidebar.name}`.")
            st.session_state.generated_files = {}
            st.session_state.last_loaded = uploaded_poscar_sidebar.name
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")
            st.session_state.structure = None

    st.info(f"**Calculation Location:** `{st.session_state.work_dir}`")
    st.divider()
    st.header("Run or Configure the Convergence Test Below")
    start_test = st.button("🚀 Start Convergence Test", type="secondary",
                           disabled=st.session_state.calculation_running or not st.session_state.structure)
    if st.session_state.structure:
        vis_col, info_col = st.columns([1.2, 1])
        with info_col:
            st.markdown("##### Structure Details")
            s = st.session_state.structure
            st.write(f"**Formula:** `{s.composition.reduced_formula}`")
            st.write(f"**Atoms:** {s.num_sites}")
            st.write(f"**Lattice:**")
            st.markdown(f"&nbsp;&nbsp;a={s.lattice.a:.4f}Å, b={s.lattice.b:.4f}Å, c={s.lattice.c:.4f}Å")
            st.markdown(f"&nbsp;&nbsp;α={s.lattice.alpha:.2f}°, β={s.lattice.beta:.2f}°, γ={s.lattice.gamma:.2f}°")
        with vis_col:
            components.html(view_structure(s, height=350, width=500), height=360)
        st.divider()

    st.header("Convergence Test Setup")
    calc_mode = st.radio("Select Convergence Test:", ["ENCUT", "K-Point Sampling"], horizontal=True,
                         key="calc_mode_select")

    param_col1, param_col2 = st.columns([2, 2])
    with param_col1:
        if calc_mode == "ENCUT":
            st.subheader("ENCUT Parameters")
            encut_max = st.number_input("Max ENCUT (eV)", value=700, step=25)
            encut_min = st.number_input("Min ENCUT (eV)", value=350, step=25)
            encut_step = st.number_input("ENCUT Step (eV)", value=-50, step=10)
            st.subheader("K-Point Mesh for ENCUT Test")
            k_col1, k_col2, k_col3 = st.columns(3)
            kx = k_col1.number_input("$k_x$", min_value=1, value=2, step=1)
            ky = k_col2.number_input("$k_y$", min_value=1, value=2, step=1)
            kz = k_col3.number_input("$k_z$", min_value=1, value=2, step=1)
        else:
            st.subheader("K-Spacing Parameters")
            k_start = st.number_input("Start k-spacing (Å⁻¹)", value=0.6, format="%.3f")
            k_end = st.number_input("End k-spacing (Å⁻¹)", value=0.1, format="%.3f")
            k_step = st.number_input("k-spacing Step", value=-0.025, format="%.4f")
            encut_for_ktest = st.number_input("ENCUT for K-Point Test (eV)", value=520)

            grid_col1, grid_col2 = st.columns(2)
            with grid_col1:
                ks, ke, ksc = get_kpoints_from_kspacing(st.session_state.structure, k_start)
                st.markdown(f"**Start Grid:** `{ks} x {ke} x {ksc}`")
            with grid_col2:
                ks, ke, ksc = get_kpoints_from_kspacing(st.session_state.structure, k_end)
                st.markdown(f"**End Grid:** `{ks} x {ke} x {ksc}`")

    with param_col2:
        st.subheader("INCAR Parameters")
        st.info(
            f"You can **modified or add new lines** into the following **INCAR** file. When you click on the button to start the convergence test,"
            f"the settings set here will be used.")
        st.session_state.incar_content = st.text_area("Edit INCAR file", value=st.session_state.incar_content,
                                                      height=250)

    st.divider()
    st.header("File Generation")

    exec_col1, exec_col2 = st.columns(2)

    with exec_col1:
        if st.button("⚙️ Generate Input Files", disabled=(st.session_state.structure is None)):
            structure = st.session_state.structure
            st.session_state.generated_files['poscar'] = structure.to(fmt="poscar")
            st.session_state.generated_files['potcar'] = create_potcar(structure, st.session_state.vasp_potentials_path)
            if calc_mode == "ENCUT":
                st.session_state.generated_files['kpoints'] = create_kpoints(kx, ky, kz)

                st.session_state.generated_files['incar'] = st.session_state.incar_content
            else:
                ka, kb, kc = get_kpoints_from_kspacing(structure, k_start)
                st.session_state.generated_files['kpoints'] = create_kpoints(ka, kb, kc)
                st.session_state.generated_files['incar'] = st.session_state.incar_content
            st.success("Input files generated in memory.")

    if st.session_state.generated_files:
        if 'potcar_details' in st.session_state.generated_files:
            with st.expander("POTCAR Generation Details"):
                for detail in st.session_state.generated_files['potcar_details']:
                    st.text(detail)

        dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
        dl_col1.download_button("Download POSCAR", st.session_state.generated_files.get('poscar', ''), "POSCAR",
                                type='primary')
        dl_col2.download_button("Download POTCAR", st.session_state.generated_files.get('potcar', ''), "POTCAR",
                                type='primary')
        dl_col3.download_button("Download KPOINTS", st.session_state.generated_files.get('kpoints', ''), "KPOINTS",
                                type='primary')
        dl_col4.download_button("Download INCAR", st.session_state.generated_files.get('incar', ''), "INCAR",
                                type='primary')

    st.write("---")
    st.markdown("""
        <style>
        div.stButton > button[kind="primary"] {
            background-color: #0099ff; color: white; font-size: 16px; font-weight: bold;
            padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
        }
        div.stButton > button[kind="primary"]:active, div.stButton > button[kind="primary"]:focus {
            background-color: #007acc !important; color: white !important; box-shadow: none !important;
        }

        div.stButton > download_button[kind="secondary"] {
            background-color: #dc3545; color: white; font-size: 16px; font-weight: bold;
            padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
        }

        div.stButton > button[kind="secondary"] {
            background-color: #dc3545; color: white; font-size: 16px; font-weight: bold;
            padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
        }
        div.stButton > button[kind="secondary"]:active, div.stButton > button[kind="secondary"]:focus {
            background-color: #c82333 !important; color: white !important; box-shadow: none !important;
        }

        div.stButton > button[kind="tertiary"] {
            background-color: #6f42c1; color: white; font-size: 16px; font-weight: bold;
            padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
        }
        div.stButton > button[kind="tertiary"]:active, div.stButton > button[kind="tertiary"]:focus {
            background-color: #5a2d91 !important; color: white !important; box-shadow: none !important;
        }

        div[data-testid="stDataFrameContainer"] table td { font-size: 16px !important; }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    if start_test:
        with st.spinner("Running the calculation test..."):
            os.makedirs(st.session_state.work_dir, exist_ok=True)
            with open(os.path.join(st.session_state.work_dir, "POSCAR"), "w") as f:
                f.write(st.session_state.structure.to(fmt="poscar"))

            potcar_content = create_potcar(st.session_state.structure, st.session_state.vasp_potentials_path)
            if not potcar_content:
                st.error("POTCAR generation failed. Cannot start calculation.")
            else:
                with open(os.path.join(st.session_state.work_dir, "POTCAR"), "w") as f:
                    f.write(potcar_content)

                if calc_mode == 'ENCUT':
                    with open(os.path.join(st.session_state.work_dir, "KPOINTS"), "w") as f:
                        f.write(create_kpoints(kx, ky, kz))
                    calc_params = {'mode': 'encut', 'command': st.session_state.vasp_command, 'max': encut_max,
                                   'min': encut_min, 'step': encut_step, 'incar': st.session_state.incar_content}
                    total_steps = len(range(encut_max, encut_min + (-1 if encut_step < 0 else 1), encut_step))
                else:
                    calc_params = {'mode': 'kpoint', 'command': st.session_state.vasp_command, 'start': k_start,
                                   'end': k_end, 'step': k_step, 'structure': st.session_state.structure,
                                   'incar': st.session_state.incar_content}
                    total_steps = len(np.arange(k_start, k_end + k_step, k_step))

                st.session_state.calculation_running = True
                st.session_state.log_messages = []
                st.session_state.results_list = []
                st.session_state.stop_event.clear()
                st.session_state.progress = 0
                st.session_state.total_steps = total_steps

                st.info(f"Starting {calc_mode} test in `{st.session_state.work_dir}`.")
                thread = threading.Thread(target=run_vasp_in_thread, args=(
                    calc_params, st.session_state.work_dir, st.session_state.log_queue, st.session_state.stop_event))
                thread.start()
                st.write('Running')
                # time.sleep(1)
                st.rerun()


rerun_needed = not st.session_state.log_queue.empty()
while not st.session_state.log_queue.empty():
    message = st.session_state.log_queue.get()

    if isinstance(message, dict):
        if message.get('type') == 'data':
            st.session_state.results_list.append(message)
        elif message.get('type') == 'progress':
            st.session_state.progress = message['step']
            val_str = f"{message['value']:.4f}" if isinstance(message['value'], float) else str(message['value'])
            grid_str = f" (Grid: {message['k_grid']})" if 'k_grid' in message else ""
            st.session_state.progress_text = f"Step {message['step'] + 1}/{st.session_state.total_steps}: Current Value = {val_str}{grid_str}"

    elif message == "THREAD_FINISHED":
        st.session_state.calculation_running = False
        st.success("Calculation thread finished.")
    else:
        st.session_state.log_messages.append(str(message))

with tab2:
    st.header("Live Calculation Output")
    if st.session_state.calculation_running:
        if st.button("🛑 Stop Calculation", type="primary"):
            st.session_state.stop_event.set()
            st.warning("Stop signal sent. Calculation will terminate after the current step.")

        progress_value = (
                st.session_state.progress / st.session_state.total_steps) if st.session_state.total_steps > 0 else 0
        st.progress(progress_value, text=st.session_state.progress_text)

    st.text_area("Log", "\n".join(st.session_state.log_messages), height=500, key="log_area")

with tab3:
    st.header("Final Convergence Results")
    if st.session_state.results_list:
        df_plot = pd.DataFrame(st.session_state.results_list)

        if 'energy' in df_plot.columns and not df_plot.empty:
            if 'encut' in df_plot.columns:
                df_plot = df_plot.sort_values(by='encut', ascending=False)
                reference_energy = df_plot.iloc[0]['energy']
                x_axis, title_text = 'encut', 'ENCUT Convergence'
                x_label = 'Energy Cutoff (eV)'
                xaxis_opts = {}
            else:
                df_plot = df_plot.sort_values(by='k_spacing', ascending=True)
                reference_energy = df_plot.iloc[0]['energy']
                df_plot['k_label'] = df_plot['k_spacing'].round(4).astype(str) + " (" + df_plot['k_grid'] + ")"
                x_axis, title_text = 'k_label', 'K-Point Convergence'
                x_label = 'K-Spacing (Å⁻¹) and Grid'
                xaxis_opts = {'autorange': 'reversed'}

            df_plot['Energy_Diff_per_Atom'] = np.abs(
                (df_plot['energy'] - reference_energy) / len(st.session_state.structure))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_plot[x_axis],
                y=df_plot['Energy_Diff_per_Atom'],
                name='ΔE / atom',
                mode='lines+markers',
                yaxis='y1',
                line=dict(width=3),  
                marker=dict(size=12) 
            ))
            fig.add_trace(go.Scatter(
                x=df_plot[x_axis],
                y=df_plot['time'],
                name='Time / step',
                mode='lines+markers',
                yaxis='y2',
                line=dict(width=3, dash='dash', color='green'), 
                marker=dict(size=10, symbol='square', color='green') 
            ))

            fig.update_layout(
                height=600,
                title=dict(text=title_text, font=dict(size=28, color='black')),
                xaxis_title=x_label,
                xaxis=dict(
                    title_font=dict(size=20, color='black'),
                    tickfont=dict(size=19, color='black'),
                    **xaxis_opts
                ),
                yaxis=dict(
                    title='Abs. ΔE per Atom (eV)',
                    color='blue',
                    title_font=dict(size=20, color='blue'),
                    tickfont=dict(size=19, color='blue')
                ),
                yaxis2=dict(
                    title='Time / step (min)',
                    overlaying='y',
                    side='right',
                    color='green',
                    title_font=dict(size=20, color='green'),
                    tickfont=dict(size=16, color='green')
                ),
                legend=dict(
                    y=1.1,
                    x=0.5,
                    xanchor='center',
                    orientation="h",
                    font=dict(size=18, color='black')
                ),
                font=dict(size=16, color='black')  
            )
            fig.add_hline(
                y=0.001,
                line_dash="dash",
                line_color="red",
                annotation_text="1 meV/atom threshold",
                annotation_font_size=18
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_plot)
    else:
        st.info("Results will appear here as calculations complete.")

if st.session_state.calculation_running or rerun_needed:
    time.sleep(1)
    st.rerun()
