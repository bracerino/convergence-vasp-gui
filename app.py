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



def parse_oszicar(oszicar_content):
    lines = oszicar_content.strip().split('\n')

    optimization_steps = []
    ionic_steps = []
    energies = []
    electronic_steps = []
    ncg_steps = []
    de_values = []
    methods_used = []
    elec_steps_per_ionic = []

    current_ionic_step = 0
    current_ionic_elec_count = 0

    for line in lines:
        line = line.strip()

        ionic_match = re.match(r'^\s*(\d+)\s+F=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
        if ionic_match:
            if current_ionic_step > 0:
                elec_steps_per_ionic.append(current_ionic_elec_count)

            current_ionic_step = int(ionic_match.group(1))
            energy = float(ionic_match.group(2))
            ionic_steps.append(current_ionic_step)
            energies.append(energy)
            current_ionic_elec_count = 0
            continue

        electronic_match = re.match(
            r'^\s*(DAV|RMM):\s*(\d+)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(\d+)',
            line
        )
        if electronic_match:
            method = electronic_match.group(1)
            elec_step = int(electronic_match.group(2))
            energy = float(electronic_match.group(3))
            de = float(electronic_match.group(4)) if electronic_match.group(4) else 0
            d_eps = float(electronic_match.group(5)) if electronic_match.group(5) else 0
            ncg = int(electronic_match.group(6)) if electronic_match.group(6) else 0

            optimization_steps.append(len(optimization_steps) + 1)
            electronic_steps.append(f"Ionic {current_ionic_step}, {method} {elec_step}")
            methods_used.append(method)
            de_values.append(abs(de))
            ncg_steps.append(ncg)

            current_ionic_elec_count += 1

    if current_ionic_step > 0:
        elec_steps_per_ionic.append(current_ionic_elec_count)

    return (optimization_steps, electronic_steps, ionic_steps, energies,
            ncg_steps, de_values, methods_used, elec_steps_per_ionic)


def parse_incar(incar_content):
    ediff = 1E-4

    lines = incar_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        ediff_match = re.search(r'EDIFF\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line, re.IGNORECASE)
        if ediff_match:
            ediff = float(ediff_match.group(1))

    return ediff


def identify_file_type(filename):
    filename_lower = filename.lower()
    if 'incar' in filename_lower:
        return 'INCAR'
    elif 'oszicar' in filename_lower:
        return 'OSZICAR'
    else:
        return 'UNKNOWN'


def create_convergence_tab():
    st.header("VASP Optimization Convergence Analysis")
    st.info("Upload your VASP files (INCAR and/or OSZICAR) to analyze energy convergence during optimization. **Only OSZICAR is required.**")

    st.subheader("Upload VASP Files")
    uploaded_files = st.file_uploader(
        "Upload INCAR and/or OSZICAR)",
        accept_multiple_files=True,
        key="vasp_files_upload",
        help="Upload INCAR and/or OSZICAR files. Files are automatically recognized by name."
    )

    if uploaded_files:
        incar_content = None
        oszicar_content = None
        incar_found = False
        oszicar_found = False

        for uploaded_file in uploaded_files:
            file_type = identify_file_type(uploaded_file.name)

            if file_type == 'INCAR':
                incar_content = uploaded_file.getvalue().decode("utf-8")
                incar_found = True
                st.success(f"✅ INCAR file detected: {uploaded_file.name}")
            elif file_type == 'OSZICAR':
                oszicar_content = uploaded_file.getvalue().decode("utf-8")
                oszicar_found = True
                st.success(f"✅ OSZICAR file detected: {uploaded_file.name}")
            else:
                st.warning(f"⚠️ Unknown file type: {uploaded_file.name} (will be ignored)")

        if not oszicar_found:
            st.error("❌ No OSZICAR file detected. Please upload an OSZICAR file to analyze convergence.")
            st.info("💡 Make sure your file name contains 'OSZICAR' or 'oszicar'")
            return

        try:
            if incar_found and incar_content:
                ediff = parse_incar(incar_content)
                ediff_source = "from INCAR file"
            else:
                ediff = 1E-4
                ediff_source = "default value (no INCAR file provided)"

            (optimization_steps, electronic_steps, ionic_steps, energies,
             ncg_steps, de_values, methods_used, elec_steps_per_ionic) = parse_oszicar(oszicar_content)

            if not ionic_steps or not energies:
                st.error("❌ No valid energy data found in OSZICAR file. Please check the file format.")
                return

            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Ionic Steps Found:** {len(ionic_steps)}")
            with col2:
                st.info(f"**EDIFF:** {ediff:.2E} eV")
            with col3:
                st.info(f"**EDIFF Source:** {ediff_source}")

            energy_diffs = []
            if len(energies) > 1:
                for i in range(1, len(energies)):
                    energy_diffs.append(abs(energies[i] - energies[i - 1]))
                energy_diffs.insert(0, float('inf'))
            else:
                energy_diffs = [float('inf')]

            df = pd.DataFrame({
                'Ionic Step': ionic_steps,
                'Energy (eV)': energies,
                'Energy Difference (eV)': energy_diffs,
                'Electronic Steps': elec_steps_per_ionic
            })

            st.subheader("Energy Convergence Analysis")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=ionic_steps,
                y=energies,
                name='Total Energy',
                mode='lines+markers',
                line=dict(width=3, color='blue'),
                marker=dict(size=8),
                yaxis='y1',
                hovertemplate='Step: %{x}<br>Energy: %{y:.6f} eV<extra></extra>'
            ))

            if len(energy_diffs) > 1:
                valid_diffs = [diff for diff in energy_diffs[1:] if diff != float('inf')]
                if valid_diffs:
                    fig.add_trace(go.Scatter(
                        x=ionic_steps[1:],
                        y=valid_diffs,
                        name='|ΔE|',
                        mode='lines+markers',
                        line=dict(width=2, color='red', dash='dash'),
                        marker=dict(size=6, symbol='square'),
                        yaxis='y2',
                        hovertemplate='Step: %{x}<br>|ΔE|: %{y:.2E} eV<extra></extra>'
                    ))

            if len(energy_diffs) > 1:
                fig.add_hline(
                    y=ediff,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"EDIFF = {ediff:.2E} eV",
                    annotation_position="top right",
                    annotation_font_size=18,
                    yref="y2"
                )

            fig.update_layout(
                title=dict(
                    text="VASP Optimization Convergence",
                    font=dict(size=24, color='black')
                ),
                xaxis=dict(
                    title='Ionic Step',
                    title_font=dict(size=20, color='black'),
                    tickfont=dict(size=20, color='black'),
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title='Total Energy (eV)',
                    side='left',
                    title_font=dict(size=20, color='blue'),
                    tickfont=dict(size=20, color='blue')
                ),
                yaxis2=dict(
                    title='|Energy Difference| (eV)',
                    side='right',
                    overlaying='y',
                    title_font=dict(size=20, color='red'),
                    tickfont=dict(size=20, color='red'),
                    tickformat='.1E'
                ),
                legend=dict(
                    x=0.5,
                    y=-0.2,
                    xanchor='center',
                    orientation='h',
                    bgcolor='rgba(255,255,255,0.8)',
                    borderwidth=1,
                    font=dict(size=18)
                ),
                height=600,
                plot_bgcolor='white',
                font=dict(size=20, color='black')
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Electronic Convergence Efficiency")

            col_eff1, col_eff2 = st.columns(2)

            with col_eff1:
                if elec_steps_per_ionic:
                    fig_elec = go.Figure()

                    fig_elec.add_trace(go.Bar(
                        x=ionic_steps,
                        y=elec_steps_per_ionic,
                        name='Electronic Steps',
                        marker_color='purple',
                        hovertemplate='Ionic Step: %{x}<br>Electronic Steps: %{y}<extra></extra>'
                    ))

                    fig_elec.update_layout(
                        title=dict(
                            text="Electronic Steps per Ionic Step",
                            font=dict(size=18, color='black')
                        ),
                        xaxis=dict(
                            title='Ionic Step',
                            title_font=dict(size=14, color='black'),
                            tickfont=dict(size=12, color='black')
                        ),
                        yaxis=dict(
                            title='Electronic Steps',
                            title_font=dict(size=14, color='purple'),
                            tickfont=dict(size=12, color='purple')
                        ),
                        height=400,
                        plot_bgcolor='white'
                    )

                    st.plotly_chart(fig_elec, use_container_width=True)

            with col_eff2:
                if methods_used:
                    method_counts = pd.Series(methods_used).value_counts()

                    fig_methods = go.Figure(data=[go.Pie(
                        labels=method_counts.index,
                        values=method_counts.values,
                        hole=0.3,
                        hovertemplate='Method: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                    )])

                    fig_methods.update_layout(
                        title=dict(
                            text="Electronic Method Usage",
                            font=dict(size=18, color='black')
                        ),
                        height=400,
                        font=dict(size=12)
                    )

                    st.plotly_chart(fig_methods, use_container_width=True)

            st.subheader("Convergence Analysis Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="Total Ionic Steps",
                    value=len(ionic_steps)
                )

            with col2:
                final_energy = energies[-1] if energies else 0
                st.metric(
                    label="Final Energy (eV)",
                    value=f"{final_energy:.6f}"
                )

            with col3:
                if len(energy_diffs) > 1:
                    final_diff = energy_diffs[-1] if energy_diffs[-1] != float('inf') else 0
                    st.metric(
                        label="Final |ΔE| (eV)",
                        value=f"{final_diff:.2E}" if final_diff != float('inf') else "N/A"
                    )
                else:
                    st.metric(
                        label="Final |ΔE| (eV)",
                        value="N/A"
                    )

            with col4:
                if len(energy_diffs) > 1:
                    final_diff = energy_diffs[-1] if energy_diffs[-1] != float('inf') else 0
                    energy_converged = final_diff < ediff if final_diff != float('inf') else False
                    st.metric(
                        label="Energy Converged",
                        value="Yes" if energy_converged else "No"
                    )
                else:
                    st.metric(
                        label="Energy Converged",
                        value="Unknown"
                    )

            if len(energy_diffs) > 1:
                valid_diffs = [diff for diff in energy_diffs[1:] if diff != float('inf')]
                if valid_diffs:
                    converged_steps = [i + 2 for i, diff in enumerate(valid_diffs) if diff < ediff]
                    if converged_steps:
                        st.success(f"🎯 Energy convergence first achieved at ionic step: **{converged_steps[0]}**")
                    else:
                        st.warning(f"⚠️ Energy difference has not reached EDIFF threshold of {ediff:.2E} eV")

            if elec_steps_per_ionic:
                avg_elec_steps = np.mean(elec_steps_per_ionic)
                max_elec_steps = max(elec_steps_per_ionic)
                min_elec_steps = min(elec_steps_per_ionic)

                st.info(f"📊 **Electronic Convergence Efficiency:** Avg: {avg_elec_steps:.1f} steps/ionic, "
                        f"Range: {min_elec_steps}-{max_elec_steps} steps")

            if methods_used:
                method_counts = pd.Series(methods_used).value_counts()
                method_summary = ", ".join([f"{method}: {count}" for method, count in method_counts.items()])
                st.info(f"🔧 **Methods Used:** {method_summary} electronic steps")

            st.subheader("Detailed Data")

            df_display = df.copy()
            df_display['Energy (eV)'] = df_display['Energy (eV)'].apply(lambda x: f"{x:.6f}")
            df_display['Energy Difference (eV)'] = df_display['Energy Difference (eV)'].apply(
                lambda x: f"{x:.2E}" if x != float('inf') else "N/A"
            )

            st.dataframe(df_display, use_container_width=True)

            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Convergence Data (CSV)",
                data=csv_data,
                file_name="vasp_convergence_data.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.error("Please ensure your files are in the correct VASP format.")

    else:
        st.info("📁 Please upload your VASP files to begin analysis.")

        with st.expander("📋 Expected File Formats and Naming"):
            st.markdown("""
            **File Naming Convention:**
            - Files containing 'INCAR' or 'incar' in the name will be recognized as INCAR files
            - Files containing 'OSZICAR' or 'oszicar' in the name will be recognized as OSZICAR files

            **Examples of valid file names:**
            - `INCAR`, `incar`, `INCAR.txt`, `my_incar_file.txt`
            - `OSZICAR`, `oszicar`, `OSZICAR.txt`, `calculation_oszicar.txt`

            **INCAR file should contain (optional):**
            ```
            SYSTEM = Your System Name
            PREC   = Accurate
            ENCUT  = 520
            EDIFF  = 1E-5
            IBRION = 2
            ...
            ```

            **OSZICAR file should contain optimization steps like (required):**
            ```
            DAV:   1     0.107357121467E+05    0.10736E+05   -0.33896E+05  5440   0.240E+03
            DAV:   2     0.182815773570E+04   -0.89076E+04   -0.85277E+04  5440   0.677E+02
            ...
               1 F= -.53549513E+03 E0= -.53549513E+03  d E =-.535495E+03
            DAV:   1    -0.210829461167E+04   -0.15728E+04   -0.61866E+04  5740   0.686E+02
            ...
               2 F= 0.14480323E+04 E0= 0.14480554E+04  d E =0.198353E+04
            ```

            **Note:** Only OSZICAR file is required for analysis. INCAR is optional and used to read custom EDIFF values.
            """)



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


import select
import fcntl
import os
import signal
import psutil


def run_vasp_in_thread(calc_params, work_dir, log_queue, stop_event):
    mode = calc_params['mode']
    command = calc_params['command']
    incar_template = calc_params['incar']

    output_filename = f"{mode.upper()}_CONVERGENCE_TEST.txt"
    output_file_path = os.path.join(work_dir, output_filename)

    header = ("# ENCUT[eV] Total_Energy[eV] Time/step(min)\n" if mode == 'encut'
              else "# k_spacing[A^-1] k_a k_b k_c Total_Energy[eV] Time/step(min)\n")

    current_process = None

    def set_non_blocking(fd):
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def terminate_process(process):
        if process and process.poll() is None:
            log_queue.put("--- Terminating VASP process and all children ---")
            try:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    log_queue.put("--- Sent SIGTERM to process group ---")
                except (OSError, ProcessLookupError):
                    process.terminate()
                    log_queue.put("--- Sent SIGTERM to main process ---")
                try:
                    process.wait(timeout=1)
                    log_queue.put("--- Process terminated gracefully ---")
                except subprocess.TimeoutExpired:
                    log_queue.put("--- Force killing process group ---")
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        process.kill()
                    process.wait()
                    log_queue.put("--- Process killed ---")
            except Exception as e:
                log_queue.put(f"Error terminating process: {e}")
                try:
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        child.kill()
                    parent.kill()
                    log_queue.put("--- Used psutil to kill process tree ---")
                except:
                    pass

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

            current_process = subprocess.Popen(command, shell=True, cwd=work_dir,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, bufsize=1, preexec_fn=os.setsid)
            set_non_blocking(current_process.stdout.fileno())

            output_buffer = ""

            while current_process.poll() is None:
                if stop_event.is_set():
                    terminate_process(current_process)
                    log_queue.put("--- Calculation stopped by user during VASP execution ---")
                    return

                try:
                    ready, _, _ = select.select([current_process.stdout], [], [], 0.1)  # 0.1 second timeout

                    if ready:
                        try:
                            chunk = current_process.stdout.read(1024)  # Read in chunks
                            if chunk:
                                output_buffer += chunk
                                # Process complete lines
                                while '\n' in output_buffer:
                                    line, output_buffer = output_buffer.split('\n', 1)
                                    if line.strip():
                                        log_queue.put(line.strip())
                        except (BlockingIOError, OSError):
                            pass

                except (select.error, ValueError):
                    break
                time.sleep(0.05)  # Very short sleep - 50ms
            if output_buffer.strip():
                for line in output_buffer.strip().split('\n'):
                    if line.strip():
                        log_queue.put(line.strip())
            if stop_event.is_set():
                log_queue.put("--- Calculation stopped by user ---")
                return

            return_code = current_process.returncode
            current_process = None

            end_time = time.time()
            elapsed_time_min = (end_time - start_time) / 60.0

            if return_code != 0:
                log_queue.put(f"ERROR: VASP calculation failed with return code {return_code}.")
                break

            total_energy = None
            outcar_path = os.path.join(work_dir, "OUTCAR")
            if os.path.exists(outcar_path):
                with open(outcar_path, 'r') as outcar:
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
        if current_process:
            terminate_process(current_process)
    finally:
        if current_process:
            terminate_process(current_process)
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
    st.sidebar.info(f"❤️🫶 **[Donations always appreciated!](https://buymeacoffee.com/bracerino)**")
    st.header("3. Upload Structure (Optional)")
    st.info("If a `POSCAR` file isn't found in your project folder, upload one here.")
    uploaded_poscar_sidebar = st.file_uploader("Upload", type=['POSCAR', 'vasp', 'contcar'],
                                               label_visibility="collapsed")

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem !important;
        color: #1e3a8a !important;
        font-weight: bold !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 25px !important;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["➡️ Run Workflow", "🖥️ Live Console", "📊 Live Results","📈 OSZICAR Analysis"])

with tab1:
    st.header("1. Define Working Directory")
    st.session_state.work_dir = st.text_input("Project Folder Path", value=st.session_state.work_dir)
    if st.session_state.calculation_running:
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                with st.spinner("Calculation in progress..."):
                    st.success("✅ VASP calculations are running")
                    if st.session_state.progress_text:
                        st.write(f"📈 {st.session_state.progress_text}")
                    st.write("👀 **Switch to 'Live Console or Live Results' tab for detailed output**")
                    if st.button("🛑 Stop Calculation", key="stop_main"):
                        st.session_state.stop_event.set()
        st.divider()
    auto_poscar_path = os.path.join(st.session_state.work_dir, "POSCAR")
    if os.path.exists(auto_poscar_path):
        try:
            st.session_state.structure = Structure.from_file(auto_poscar_path)
            if "last_loaded" not in st.session_state or st.session_state.last_loaded != auto_poscar_path:
                st.success(f"Loaded `POSCAR` from `{st.session_state.work_dir}`.")
                # st.session_state.generated_files = {}
                st.session_state.last_loaded = auto_poscar_path
                # st.rerun()
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

with tab4:
    create_convergence_tab()
if st.session_state.calculation_running or rerun_needed:
    time.sleep(1)
    st.rerun()
