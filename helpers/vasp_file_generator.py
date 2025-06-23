import streamlit as st
import os
import subprocess
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import threading
import queue
import time
import numpy as np
import re
from pymatgen.core import Structure

def parse_xdatcar(xdatcar_content):
    lines = xdatcar_content.strip().split('\n')
    lattice_params = []
    
    i = 0
    while i < len(lines):
        if 'Direct configuration=' in lines[i]:
            config_num = int(lines[i].split('=')[1].strip())
            
            lattice_vectors = []
            
            try:
                for offset in [5, 4, 3]:
                    line_idx = i - offset
                    if line_idx >= 0:
                        line = lines[line_idx].strip()
                        parts = line.split()
                        if len(parts) == 3:
                            vector = [float(x) for x in parts]
                            lattice_vectors.append(vector)
                
                if len(lattice_vectors) == 3:
                    lattice_matrix = np.array(lattice_vectors)
                    
                    a_vec = lattice_matrix[0]
                    b_vec = lattice_matrix[1] 
                    c_vec = lattice_matrix[2]
                    
                    a = np.linalg.norm(a_vec)
                    b = np.linalg.norm(b_vec)
                    c = np.linalg.norm(c_vec)
                    

                    alpha = np.degrees(np.arccos(np.clip(np.dot(b_vec, c_vec) / (b * c), -1.0, 1.0)))
                    beta = np.degrees(np.arccos(np.clip(np.dot(a_vec, c_vec) / (a * c), -1.0, 1.0)))
                    gamma = np.degrees(np.arccos(np.clip(np.dot(a_vec, b_vec) / (a * b), -1.0, 1.0)))
                    
                    volume = np.abs(np.linalg.det(lattice_matrix))
                    
                    lattice_params.append({
                        'config': config_num, 
                        'a': a, 
                        'b': b, 
                        'c': c, 
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'volume': volume
                    })
            except (ValueError, IndexError):
                pass
        i += 1
    
    return lattice_params

def parse_oszicar_geom(oszicar_content):
    lines = oszicar_content.strip().split('\n')
    energies = []
    ionic_steps = []
    scf_per_ionic = []
    current_scf_step = 0
    
    current_ionic_step = 0
    current_scf_count = 0
    
    for line in lines:
        line = line.strip()
        
        ionic_match = re.match(r'^\s*(\d+)\s+F=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
        if ionic_match:
            if current_ionic_step > 0:
                scf_per_ionic.append(current_scf_count)
            
            step = int(ionic_match.group(1))
            energy = float(ionic_match.group(2))
            ionic_steps.append(step)
            energies.append(energy)
            current_ionic_step = step
            current_scf_count = 0
            continue
        
        electronic_match = re.match(r'^\s*(DAV|RMM):\s*(\d+)', line)
        if electronic_match and current_ionic_step > 0:
            current_scf_count += 1
            current_scf_step = int(electronic_match.group(2))
    
    if current_ionic_step > 0:
        scf_per_ionic.append(current_scf_count)
    
    return ionic_steps, energies, scf_per_ionic, current_ionic_step, current_scf_step

def parse_incar_nsw(incar_content):
    nsw = 50
    lines = incar_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        nsw_match = re.search(r'NSW\s*=\s*(\d+)', line, re.IGNORECASE)
        if nsw_match:
            nsw = int(nsw_match.group(1))
    return nsw

def parse_incar_nelm(incar_content):
    nelm = 60  # default VASP value
    lines = incar_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        nelm_match = re.search(r'NELM\s*=\s*(\d+)', line, re.IGNORECASE)
        if nelm_match:
            nelm = int(nelm_match.group(1))
            break
    return nelm

def parse_incar_ibrion(incar_content):
    ibrion = 2
    lines = incar_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        ibrion_match = re.search(r'IBRION\s*=\s*(-?\d+)', line, re.IGNORECASE)
        if ibrion_match:
            ibrion = int(ibrion_match.group(1))
            break
    return ibrion
    ibrion = 2
    lines = incar_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        ibrion_match = re.search(r'IBRION\s*=\s*(-?\d+)', line, re.IGNORECASE)
        if ibrion_match:
            ibrion = int(ibrion_match.group(1))
            break
    return ibrion

def create_single_point_incar():
    return """SYSTEM = Single Point Energy Calculation
PREC   = Accurate
ENCUT  = 520
ISMEAR = 0
SIGMA  = 0.05
IBRION = -1
NELM   = 150
LWAVE  = .FALSE.
LCHARG = .FALSE.
"""

def create_geometry_opt_incar(isif=3):
    return f"""SYSTEM = Geometry Optimization
PREC   = Accurate
ENCUT  = 520
ISMEAR = 0
SIGMA  = 0.05
IBRION = 2
NSW    = 50
EDIFFG = -0.01
ISIF   = {isif}
NELM   = 150
LWAVE  = .FALSE.
LCHARG = .FALSE.
"""

def create_bash_script(vasp_binary, nsw_ibrion2, nsw_ibrion1, num_processors):
    return f"""#!/bin/bash
INPUT_DIR="./"
NUMBER_OF_PROCESSORS={num_processors}
VASP_BINARY_NAME="{vasp_binary}"
cd "$INPUT_DIR"
input_file="INCAR"
search_text="IBRION"
replacement_text="IBRION = 2"
search_text_2="NSW"
replacement_text_2="NSW = {nsw_ibrion2}"
sed -i "/$search_text/c\\\\$replacement_text" "$input_file"
sed -i "/$search_text_2/c\\\\$replacement_text_2" "$input_file"
mpirun -np $NUMBER_OF_PROCESSORS $VASP_BINARY_NAME
echo "GEOM. OPTIMIZATION FINISHED WITH IBRION = 2"
if [[ -f "CONTCAR" ]]; then
    if [[ -f "POSCAR" ]]; then
        mv "POSCAR" "POSCAR_old"
    fi
    mv "CONTCAR" "POSCAR"
    echo "=================================================================================================================================="
    echo "Files renamed successfully!"
    echo "=================================================================================================================================="
else
    echo "=================================================================================================================================="
    echo "CONTCAR does not exist in the directory. No files renamed."
    echo "=================================================================================================================================="
fi
replacement_text="IBRION = 1"
replacement_text_2="NSW = {nsw_ibrion1}"
sed -i "/$search_text/c\\\\$replacement_text" "$input_file"
sed -i "/$search_text_2/c\\\\$replacement_text_2" "$input_file"
mpirun -np $NUMBER_OF_PROCESSORS $VASP_BINARY_NAME
echo "GEOM. OPTIMIZATION FINISHED WITH IBRION = 1"
"""

def run_geometry_optimization(work_dir, vasp_command, incar_content, log_queue, stop_event):
    os.makedirs(work_dir, exist_ok=True)
    
    with open(os.path.join(work_dir, "INCAR"), "w") as f:
        f.write(incar_content)
    
    if os.path.exists(os.path.join(work_dir, "WAVECAR")):
        os.remove(os.path.join(work_dir, "WAVECAR"))
    
    log_queue.put("Starting geometry optimization...")
    
    try:
        process = subprocess.Popen(
            vasp_command, 
            shell=True, 
            cwd=work_dir,
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            bufsize=1,
            preexec_fn=os.setsid
        )
        
        while process.poll() is None:
            if stop_event.is_set():
                process.terminate()
                log_queue.put("Calculation stopped by user")
                return
            
            line = process.stdout.readline()
            if line:
                log_queue.put(line.strip())
            
            oszicar_path = os.path.join(work_dir, "OSZICAR")
            xdatcar_path = os.path.join(work_dir, "XDATCAR")
            
            if os.path.exists(oszicar_path):
                try:
                    with open(oszicar_path, 'r') as f:
                        oszicar_content = f.read()
                    ionic_steps, energies, scf_per_ionic, current_ionic, current_scf = parse_oszicar_geom(oszicar_content)
                    if ionic_steps and energies:
                        log_queue.put({
                            'type': 'energy_data',
                            'steps': ionic_steps,
                            'energies': energies,
                            'scf_per_ionic': scf_per_ionic,
                            'current_ionic': current_ionic,
                            'current_scf': current_scf
                        })
                except:
                    pass
            
            if os.path.exists(xdatcar_path):
                try:
                    with open(xdatcar_path, 'r') as f:
                        xdatcar_content = f.read()
                    lattice_data = parse_xdatcar(xdatcar_content)
                    if lattice_data:
                        log_queue.put({
                            'type': 'lattice_data',
                            'data': lattice_data
                        })
                except:
                    pass
            
            time.sleep(1)
        
        return_code = process.returncode
        if return_code == 0:
            log_queue.put("Geometry optimization completed successfully!")
        else:
            log_queue.put(f"Geometry optimization failed with return code {return_code}")
    
    except Exception as e:
        log_queue.put(f"Error during calculation: {str(e)}")
    finally:
        log_queue.put("GEOM_THREAD_FINISHED")

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

def create_file_generator_tab():
    st.header("VASP File Generator & Geometry Optimization")
    
    if 'structure' not in st.session_state or st.session_state.structure is None:
        st.error("No structure loaded. Please upload a structure file first.")
        return
    
    structure = st.session_state.structure
    tab_s1, tab_s2, tab_s3, tab_s4, tab_s5 = st.tabs(["Calculation Type", 'Geometry Optimization Settings', "K-Points Settings",
        'Bash Script Settings (For Geometry Optimization)', 'Structure Information'])
    with tab_s5:
        st.subheader("Structure Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Formula:** {structure.composition.reduced_formula}")
            st.write(f"**Number of atoms:** {structure.num_sites}")
        with col2:
            st.write(f"**Lattice parameters:**")
            st.write(f"a={structure.lattice.a:.4f}Å, b={structure.lattice.b:.4f}Å, c={structure.lattice.c:.4f}Å")
        

    with tab_s1:
        st.subheader("Calculation Type")
        calc_type = st.radio(
            "Select calculation type:",
            ["Geometry Optimization", "Single Point Energy"],
            horizontal=True
        )
    
    isif_value = 3
    with tab_s2:
        if calc_type == "Geometry Optimization":
            st.subheader("Optimization Settings")
            optimization_type = st.selectbox(
                "What to optimize:",
                [
                    "All (atoms + cell shape + volume) - ISIF=3",
                    "Atoms only (fixed cell) - ISIF=2", 
                    "Cell volume only (fixed atoms + shape) - ISIF=7",
                    "Atoms + volume (fixed cell shape) - ISIF=4"
                ],
                key="optimization_type_select"
            )
            
            if "ISIF=2" in optimization_type:
                isif_value = 2
            elif "ISIF=7" in optimization_type:
                isif_value = 7
            elif "ISIF=4" in optimization_type:
                isif_value = 4
            else:
                isif_value = 3
    
        with tab_s3:
            st.subheader("K-Points Settings")
            k_col1, k_col2, k_col3 = st.columns(3)
            kx = k_col1.number_input("k_x", min_value=1, value=2, step=1)
            ky = k_col2.number_input("k_y", min_value=1, value=2, step=1)
            kz = k_col3.number_input("k_z", min_value=1, value=2, step=1)
            
            kpoints_content = f"""Automatic K-Mesh
0
Gamma
{kx} {ky} {kz}
0 0 0
"""
    
    if calc_type == "Single Point Energy":
        incar_content = create_single_point_incar()
    else:
        incar_content = create_geometry_opt_incar(isif_value)
    
    if calc_type == "Geometry Optimization":
        with tab_s4:
            st.subheader("Bash Script Settings")
            bash_col1, bash_col2, bash_col3, bash_col4 = st.columns(4)
            vasp_binary = bash_col1.text_input("VASP Binary", value="vasp_std")
            num_processors = bash_col2.number_input("Number of Processors", min_value=1, value=10)
            nsw_ibrion2 = bash_col3.number_input("NSW for IBRION=2", min_value=1, value=10)
            nsw_ibrion1 = bash_col4.number_input("NSW for IBRION=1", min_value=1, value=100)
            
            bash_script_content = create_bash_script(vasp_binary, nsw_ibrion2, nsw_ibrion1, num_processors)
    
    st.subheader("File Preview and Editing")
    
    tab2, tab1 = st.tabs(["INCAR", "KPOINTS"])
    
    with tab1:
        st.text("KPOINTS Preview:")
        edited_kpoints = st.text_area("Edit KPOINTS:", value=kpoints_content, height=150, key="kpoints_edit")
    
    with tab2:
        st.text("INCAR Preview:")
        edited_incar = st.text_area("Edit INCAR:", value=incar_content, height=300, key=f"incar_edit_{calc_type}_{isif_value}")
    
    tab_ss1, tab_ss2 = st.tabs(["Generate Input Files", 'Analyse Geometry Optimization or Run VASP',])
    with tab_ss1:
        st.subheader("Generate and Download Files")
        st.info("The files will be automatically created in the set folder.")
        work_dir_files = st.text_input("Save files to directory:", value=st.session_state.work_dir, key="file_save_dir")
        

        col_gen1, col_gen2 = st.columns(2)
        
        with col_gen1:
            generate_button = st.button("⚙️ Generate All Files", type="primary")
        
        with col_gen2:
            propagate_button = st.button("🔄 Propagate to All Subdirectories", type="tertiary")
        
        if generate_button:
            poscar_content = structure.to(fmt="poscar")
            potcar_content = create_potcar(structure, st.session_state.vasp_potentials_path)
            
            if potcar_content:
                st.session_state.file_generator = {
                    'poscar': poscar_content,
                    'potcar': potcar_content,
                    'kpoints': edited_kpoints,
                    'incar': edited_incar
                }
                
                if calc_type == "Geometry Optimization":
                    st.session_state.file_generator['bash_script'] = bash_script_content

                try:
                    os.makedirs(work_dir_files, exist_ok=True)
                    

                    with open(os.path.join(work_dir_files, "POSCAR"), "w") as f:
                        f.write(poscar_content)
                    
                    with open(os.path.join(work_dir_files, "POTCAR"), "w") as f:
                        f.write(potcar_content)
                    
                    with open(os.path.join(work_dir_files, "KPOINTS"), "w") as f:
                        f.write(edited_kpoints)

                    with open(os.path.join(work_dir_files, "INCAR"), "w") as f:
                        f.write(edited_incar)
                    
                    files_created = ["POSCAR", "POTCAR", "KPOINTS", "INCAR"]
                    
                    if calc_type == "Geometry Optimization":
                        with open(os.path.join(work_dir_files, "run_geom_opt.sh"), "w") as f:
                            f.write(bash_script_content)
                        os.chmod(os.path.join(work_dir_files, "run_geom_opt.sh"), 0o755)
                        files_created.append("run_geom_opt.sh")
                    
                    st.toast(f"✅ Files created in `{work_dir_files}`", icon="✅")
                    for file in files_created:
                        st.toast(f"Created: {file}", icon="📄")
                        
                except Exception as e:
                    st.toast(f"❌ Error creating files: {str(e)}", icon="❌")
                
                st.toast("All files generated successfully!", icon="🎉")
            else:
                st.toast("Failed to generate POTCAR. Check your potentials path.", icon="❌")

        if propagate_button:
            try:
                subdirs_processed = 0
                subdirs_failed = 0

                for root, dirs, files in os.walk(work_dir_files):
                    if 'POSCAR' in files and root != work_dir_files: 
                        subdir_name = os.path.relpath(root, work_dir_files)
                        poscar_path = os.path.join(root, 'POSCAR')
                        
                        try:
                            sub_structure = Structure.from_file(poscar_path)
                            sub_potcar_content = create_potcar(sub_structure, st.session_state.vasp_potentials_path)
                            
                            if sub_potcar_content:
                                with open(os.path.join(root, "POTCAR"), "w") as f:
                                    f.write(sub_potcar_content)
                                
                                with open(os.path.join(root, "KPOINTS"), "w") as f:
                                    f.write(edited_kpoints)
                                
                                with open(os.path.join(root, "INCAR"), "w") as f:
                                    f.write(edited_incar)

                                if calc_type == "Geometry Optimization":
                                    with open(os.path.join(root, "run_geom_opt.sh"), "w") as f:
                                        f.write(bash_script_content)
                                    os.chmod(os.path.join(root, "run_geom_opt.sh"), 0o755)
                                
                                subdirs_processed += 1
                                st.toast(f"✅ Created files in: {subdir_name}", icon="📁")
                            else:
                                subdirs_failed += 1
                                st.toast(f"❌ Failed to create POTCAR for: {subdir_name}", icon="⚠️")
                                
                        except Exception as e:
                            subdirs_failed += 1
                            st.toast(f"❌ Error in {subdir_name}: {str(e)}", icon="❌")
                
                if subdirs_processed > 0:
                    nelm_value = 60  # default
                    nsw_value = parse_incar_nsw(edited_incar)
                    
                    for line in edited_incar.split('\n'):
                        line = line.strip()
                        if line.startswith('#') or not line:
                            continue
                        nelm_match = re.search(r'NELM\s*=\s*(\d+)', line, re.IGNORECASE)
                        if nelm_match:
                            nelm_value = int(nelm_match.group(1))
                            break
                    
                    vasp_command = st.session_state.vasp_command

                    batch_script = f'''#!/bin/bash

    vasp_run="{vasp_command}"
    directory_path="{work_dir_files}"

    # Signal handler for SIGINT
    function handle_sigint() {{
        echo "Script interrupted by user, exiting."
        exit 1
    }}

    # Trap SIGINT (Control+C) and call handle_sigint
    trap handle_sigint SIGINT

    # Create directory list if it doesn't exist
    if [ ! -f "directory_list.txt" ]; then
        search_dir="$directory_path"
        dir_list="directory_list.txt"
        touch "$dir_list"

        for dir in "$search_dir"/*; do
            if [ -d "$dir" ] && [ -f "$dir/POSCAR" ]; then
                echo "FOUND DIR: $(basename "$dir")"
                echo "$(basename "$dir")" >> "$dir_list"
            fi
        done

        # Sort the directory list
        temp_file=$(mktemp)
        sort -V "$dir_list" > "$temp_file"
        mv "$temp_file" "$dir_list"
    else
        echo "The file 'directory_list.txt' already exists. Continuing with it."
    fi

    start_dir=$(pwd)
    output_file="${{directory_path}}/VASP_BATCH_RESULTS.txt"

    # Create output file with headers
    if [ ! -f "$output_file" ]; then
        echo "Creating results file: $output_file"
        echo -e "#FOLDER_NAME\\tSTATUS\\tIONIC_STEPS\\tSCF_CYCLES\\tFINAL_ENERGY[eV]\\tFINAL_a[Å]\\tFINAL_b[Å]\\tFINAL_c[Å]\\tFINAL_VOLUME[Ų]\\tTIME[min]\\tESTIMATED_MAX_TIME[min]" > "$output_file"
    fi

    mapfile -t dir_names < "directory_list.txt"

    echo "Starting VASP calculations for ${{#dir_names[@]}} directories..."

    # Function to extract lattice parameters from XDATCAR
    extract_lattice_params() {{
        local xdatcar_file="$1"
        if [ -f "$xdatcar_file" ]; then
            # Get the last lattice configuration
            local last_config=$(grep -n "Direct configuration=" "$xdatcar_file" | tail -1 | cut -d: -f1)
            if [ ! -z "$last_config" ]; then
                # Get lattice vectors (3 lines before the configuration line, after element names)
                local lattice_start=$((last_config - 5))
                local lattice_end=$((last_config - 3))
                
                # Extract lattice vectors
                local a_vec=$(sed -n "${{lattice_start}}p" "$xdatcar_file")
                local b_vec=$(sed -n "$((lattice_start + 1))p" "$xdatcar_file")
                local c_vec=$(sed -n "$((lattice_start + 2))p" "$xdatcar_file")
                
                # Calculate lattice parameters
                local a=$(echo "$a_vec" | awk '{{print sqrt($1*$1 + $2*$2 + $3*$3)}}')
                local b=$(echo "$b_vec" | awk '{{print sqrt($1*$1 + $2*$2 + $3*$3)}}')
                local c=$(echo "$c_vec" | awk '{{print sqrt($1*$1 + $2*$2 + $3*$3)}}')
                
                # Calculate volume (determinant of lattice matrix)
                local volume=$(echo "$a_vec $b_vec $c_vec" | awk '{{
                    a1=$1; a2=$2; a3=$3;
                    b1=$4; b2=$5; b3=$6;
                    c1=$7; c2=$8; c3=$9;
                    vol = a1*(b2*c3-b3*c2) - a2*(b1*c3-b3*c1) + a3*(b1*c2-b2*c1);
                    if(vol<0) vol=-vol;
                    print vol
                }}')
                
                echo "$a $b $c $volume"
            else
                echo "N/A N/A N/A N/A"
            fi
        else
            echo "N/A N/A N/A N/A"
        fi
    }}

    # Loop over directories
    for dir_name in "${{dir_names[@]}}"; do
        folder_path="${{directory_path}}/$dir_name"
        if [ -d "$folder_path" ]; then
            cd "$folder_path"
            echo ""
            echo "===================================== STARTING VASP: $dir_name ====================================="
            echo "Path: $folder_path"
            echo "=============================================================================================="
            
            start_time=$(date +%s)
            scf_start_time=$start_time
            
            # Estimate maximum time based on INCAR parameters
            NELM={nelm_value}
            NSW={nsw_value}
            
            # Run VASP
            timeout 24h $vasp_run
            vasp_exit_code=$?
            
            end_time=$(date +%s)
            duration=$(echo "scale=2; ($end_time - $start_time) / 60" | bc -l)
            
            echo "===================================== CALCULATIONS FINISHED ====================================="
            echo "Duration: $duration minutes"
            echo "========================================================================================="
            
            # Initialize variables
            status="UNKNOWN"
            ionic_steps=0
            scf_cycles=0
            final_energy="N/A"
            lattice_params="N/A N/A N/A N/A"
            estimated_max_time="N/A"
            
            # Check OSZICAR for convergence
            if [ -f "OSZICAR" ]; then
                # Get ionic steps and final energy
                ionic_steps=$(grep -c "F=" "OSZICAR" || echo "0")
                if [ $ionic_steps -gt 0 ]; then
                    final_energy=$(grep "F=" "OSZICAR" | tail -1 | awk '{{print $3}}')
                fi
                
                # Get electronic steps from last ionic step
                last_ionic_block=$(grep -n "F=" "OSZICAR" | tail -1 | cut -d: -f1)
                if [ ! -z "$last_ionic_block" ] && [ $last_ionic_block -gt 1 ]; then
                    prev_ionic_block=$(grep -n "F=" "OSZICAR" | tail -2 | head -1 | cut -d: -f1)
                    if [ -z "$prev_ionic_block" ]; then
                        prev_ionic_block=1
                    fi
                    scf_cycles=$(sed -n "${{prev_ionic_block}},${{last_ionic_block}}p" "OSZICAR" | grep -c "DAV\\|RMM" || echo "0")
                else
                    scf_cycles=$(grep -c "DAV\\|RMM" "OSZICAR" || echo "0")
                fi
                
                # Estimate time per SCF cycle
                if [ $scf_cycles -gt 0 ] && [ $ionic_steps -gt 0 ]; then
                    time_per_scf=$(echo "scale=4; $duration / ($scf_cycles * $ionic_steps)" | bc -l)
                    estimated_max_time=$(echo "scale=2; $time_per_scf * $NELM * $NSW" | bc -l)
                fi
                
                # Determine convergence status
                if [ -f "CONTCAR" ]; then
                    # Check if calculation reached NSW limit
                    if [ $ionic_steps -eq $NSW ]; then
                        status="MAX_IONIC_REACHED"
                    elif [ $scf_cycles -eq $NELM ]; then
                        status="MAX_SCF_REACHED"
                    else
                        status="CONVERGED"
                    fi
                else
                    status="FAILED"
                fi
                
                # Handle timeout
                if [ $vasp_exit_code -eq 124 ]; then
                    status="TIMEOUT_24H"
                fi
                
            else
                status="NO_OSZICAR"
            fi
            
            # Extract lattice parameters from XDATCAR
            if [ -f "XDATCAR" ]; then
                lattice_params=$(extract_lattice_params "XDATCAR")
            fi
            
            cd "$start_dir" || exit
            
            # Write results to output file
            echo -e "$dir_name\\t$status\\t$ionic_steps\\t$scf_cycles\\t$final_energy\\t$lattice_params\\t$duration\\t$estimated_max_time" >> "$output_file"
            
            # Remove completed directory from list
            sed -i "/^$dir_name$/d" "directory_list.txt"
            
            echo "Results for $dir_name written to $output_file"
            echo ""
        fi
    done

    # Cleanup
    rm -f "directory_list.txt"
    echo ""
    echo "=============================================="
    echo "ALL CALCULATIONS FINISHED!"
    echo "Results saved in: $output_file"
    echo "=============================================="

    # Generate summary statistics
    echo ""
    echo "SUMMARY:"
    total_dirs=$(wc -l < "$output_file")
    total_dirs=$((total_dirs - 1))  # Subtract header line
    converged=$(grep -c "CONVERGED" "$output_file" || echo "0")
    failed=$(grep -c -E "FAILED|NO_OSZICAR|TIMEOUT" "$output_file" || echo "0")
    max_reached=$(grep -c -E "MAX_.*_REACHED" "$output_file" || echo "0")

    echo "Total directories processed: $total_dirs"
    echo "Converged: $converged"
    echo "Max steps/cycles reached: $max_reached"
    echo "Failed: $failed"
    '''
                    
                    batch_script_path = os.path.join(work_dir_files, "run_batch_vasp.sh")
                    with open(batch_script_path, "w") as f:
                        f.write(batch_script)
                    os.chmod(batch_script_path, 0o755)
                    
                    st.toast(f"🎯 Generated comprehensive batch script: run_batch_vasp.sh", icon="📜")
                    
                    if calc_type == "Geometry Optimization":
                        geom_batch_script = f'''#!/bin/bash

    directory_path="{work_dir_files}"

    # Signal handler for SIGINT
    function handle_sigint() {{
        echo "Script interrupted by user, exiting."
        exit 1
    }}

    # Trap SIGINT (Control+C) and call handle_sigint
    trap handle_sigint SIGINT

    # Create directory list if it doesn't exist
    if [ ! -f "geom_directory_list.txt" ]; then
        search_dir="$directory_path"
        dir_list="geom_directory_list.txt"
        touch "$dir_list"

        for dir in "$search_dir"/*; do
            if [ -d "$dir" ] && [ -f "$dir/POSCAR" ] && [ -f "$dir/run_geom_opt.sh" ]; then
                echo "FOUND GEOM DIR: $(basename "$dir")"
                echo "$(basename "$dir")" >> "$dir_list"
            fi
        done

        # Sort the directory list
        temp_file=$(mktemp)
        sort -V "$dir_list" > "$temp_file"
        mv "$temp_file" "$dir_list"
    else
        echo "The file 'geom_directory_list.txt' already exists. Continuing with it."
    fi

    start_dir=$(pwd)
    output_file="${{directory_path}}/GEOM_BATCH_RESULTS.txt"

    # Create output file with headers
    if [ ! -f "$output_file" ]; then
        echo "Creating geometry optimization results file: $output_file"
        echo -e "#FOLDER_NAME\\tSTATUS\\tIBRION2_TIME[min]\\tIBRION1_TIME[min]\\tTOTAL_TIME[min]\\tFINAL_ENERGY[eV]" > "$output_file"
    fi

    mapfile -t dir_names < "geom_directory_list.txt"

    echo "Starting geometry optimization for ${{#dir_names[@]}} directories..."
    echo "Each directory will run: IBRION=2 followed by IBRION=1"
    echo ""

    # Loop over directories
    for dir_name in "${{dir_names[@]}}"; do
        folder_path="${{directory_path}}/$dir_name"
        if [ -d "$folder_path" ] && [ -f "$folder_path/run_geom_opt.sh" ]; then
            cd "$folder_path"
            echo ""
            echo "===================================== STARTING GEOMETRY OPTIMIZATION: $dir_name ====================================="
            echo "Path: $folder_path"
            echo "Running: ./run_geom_opt.sh"
            echo "============================================================================================================="
            
            start_time=$(date +%s)
            
            # Make sure the script is executable
            chmod +x run_geom_opt.sh
            
            # Run the geometry optimization script
            ./run_geom_opt.sh
            geom_exit_code=$?
            
            end_time=$(date +%s)
            total_duration=$(echo "scale=2; ($end_time - $start_time) / 60" | bc -l)
            
            echo "===================================== GEOMETRY OPTIMIZATION FINISHED ====================================="
            echo "Total Duration: $total_duration minutes"
            echo "======================================================================================================"
            
            # Initialize variables
            status="UNKNOWN"
            ibrion2_time="N/A"
            ibrion1_time="N/A"
            final_energy="N/A"
            
            # Try to extract timing information from output logs if available
            # This is approximate since we're running the full script
            ibrion2_time=$(echo "scale=2; $total_duration * 0.3" | bc -l)  # Rough estimate
            ibrion1_time=$(echo "scale=2; $total_duration * 0.7" | bc -l)  # Rough estimate
            
            # Check final results
            if [ $geom_exit_code -eq 0 ]; then
                if [ -f "OSZICAR" ]; then
                    final_energy=$(grep "F=" "OSZICAR" | tail -1 | awk '{{print $3}}' 2>/dev/null || echo "N/A")
                    if [ -f "CONTCAR" ]; then
                        status="COMPLETED"
                    else
                        status="INCOMPLETE"
                    fi
                else
                    status="NO_OUTPUT"
                fi
            else
                status="FAILED"
            fi
            
            cd "$start_dir" || exit
            
            # Write results to output file
            echo -e "$dir_name\\t$status\\t$ibrion2_time\\t$ibrion1_time\\t$total_duration\\t$final_energy" >> "$output_file"
            
            # Remove completed directory from list
            sed -i "/^$dir_name$/d" "geom_directory_list.txt"
            
            echo "Results for $dir_name written to $output_file"
            echo ""
        else
            echo "WARNING: Directory $folder_path does not exist or missing run_geom_opt.sh"
            cd "$start_dir" || exit
        fi
    done

    # Cleanup
    rm -f "geom_directory_list.txt"
    echo ""
    echo "=============================================="
    echo "ALL GEOMETRY OPTIMIZATIONS FINISHED!"
    echo "Results saved in: $output_file"
    echo "=============================================="

    # Generate summary statistics
    echo ""
    echo "SUMMARY:"
    total_dirs=$(wc -l < "$output_file")
    total_dirs=$((total_dirs - 1))  # Subtract header line
    completed=$(grep -c "COMPLETED" "$output_file" || echo "0")
    failed=$(grep -c -E "FAILED|NO_OUTPUT|INCOMPLETE" "$output_file" || echo "0")

    echo "Total directories processed: $total_dirs"
    echo "Successfully completed: $completed"
    echo "Failed or incomplete: $failed"

    # Calculate total time
    total_time=$(awk 'NR>1 {{if($5!="N/A") sum+=$5}} END {{printf "%.2f", sum}}' "$output_file")
    echo "Total computation time: $total_time minutes"
    '''
                        
                        geom_batch_script_path = os.path.join(work_dir_files, "run_batch_geom_opt.sh")
                        with open(geom_batch_script_path, "w") as f:
                            f.write(geom_batch_script)
                        os.chmod(geom_batch_script_path, 0o755)
                        
                        st.toast(f"🔧 Generated geometry optimization batch script: run_batch_geom_opt.sh", icon="⚙️")
                
                if subdirs_processed == 0 and subdirs_failed == 0:
                    st.toast("⚠️ No subdirectories with POSCAR files found", icon="📂")
                else:
                    st.toast(f"🎉 Processed {subdirs_processed} subdirectories", icon="✅")
                    if subdirs_failed > 0:
                        st.toast(f"⚠️ {subdirs_failed} subdirectories failed", icon="❌")
                        
            except Exception as e:
                st.toast(f"❌ Error during propagation: {str(e)}", icon="❌")
        
        if 'file_generator' in st.session_state:
            st.subheader("Download Files")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.download_button(
                    "📥 POSCAR",
                    st.session_state.file_generator['poscar'],
                    "POSCAR",
                    mime="text/plain", type='primary'
                )
            
            with col2:
                st.download_button(
                    "📥 POTCAR",
                    st.session_state.file_generator['potcar'],
                    "POTCAR",
                    mime="text/plain", type='primary'
                )
            
            with col3:
                st.download_button(
                    "📥 KPOINTS",
                    st.session_state.file_generator['kpoints'],
                    "KPOINTS",
                    mime="text/plain", type='primary'
                )
            
            with col4:
                st.download_button(
                    "📥 INCAR",
                    st.session_state.file_generator['incar'],
                    "INCAR",
                    mime="text/plain", type='primary'
                )
            
            if calc_type == "Geometry Optimization" and 'bash_script' in st.session_state.file_generator:
                with col5:
                    st.download_button(
                        "📥 run_geom.sh",
                        st.session_state.file_generator['bash_script'],
                        "run_geom_opt.sh",
                        mime="text/plain", type='primary'
                    )
    with tab_ss2:
        if calc_type == "Geometry Optimization":
            st.subheader("Run Geometry Optimization")
            
            work_dir_geom = st.text_input("Working Directory for Optimization", value=st.session_state.work_dir)
            
            if 'geom_running' not in st.session_state:
                st.session_state.geom_running = False
            if 'geom_stop_event' not in st.session_state:
                st.session_state.geom_stop_event = threading.Event()
            if 'geom_log_queue' not in st.session_state:
                st.session_state.geom_log_queue = queue.Queue()
            if 'geom_log_messages' not in st.session_state:
                st.session_state.geom_log_messages = []
            if 'energy_data' not in st.session_state:
                st.session_state.energy_data = {'steps': [], 'energies': [], 'scf_per_ionic': []}
            if 'lattice_data' not in st.session_state:
                st.session_state.lattice_data = []
            if 'max_nsw' not in st.session_state:
                st.session_state.max_nsw = 50
            if 'num_atoms' not in st.session_state:
                st.session_state.num_atoms = 1
            if 'ibrion_value' not in st.session_state:
                st.session_state.ibrion_value = 2
            if 'current_ionic_step' not in st.session_state:
                st.session_state.current_ionic_step = 0
            if 'current_scf_step' not in st.session_state:
                st.session_state.current_scf_step = 0
            if 'max_nelm' not in st.session_state:
                st.session_state.max_nelm = 60
            
            rerun_needed = not st.session_state.geom_log_queue.empty()
            while not st.session_state.geom_log_queue.empty():
                message = st.session_state.geom_log_queue.get()
                
                if isinstance(message, dict):
                    if message.get('type') == 'energy_data':
                        st.session_state.energy_data = {
                            'steps': message['steps'],
                            'energies': message['energies'],
                            'scf_per_ionic': message.get('scf_per_ionic', [])
                        }
                        if 'current_ionic' in message:
                            st.session_state.current_ionic_step = message['current_ionic']
                        if 'current_scf' in message:
                            st.session_state.current_scf_step = message['current_scf']
                    elif message.get('type') == 'lattice_data':
                        lattice_data = message['data']
                        
                        for i, data in enumerate(lattice_data):
                            if i == 0:
                                data['rel_a'] = 0.0
                                data['rel_b'] = 0.0
                                data['rel_c'] = 0.0
                                data['rel_volume'] = 0.0
                            else:
                                prev_data = lattice_data[i-1]
                                
                                data['rel_a'] = ((data['a'] - prev_data['a']) / prev_data['a']) * 100 if prev_data['a'] != 0 else 0.0
                                data['rel_b'] = ((data['b'] - prev_data['b']) / prev_data['b']) * 100 if prev_data['b'] != 0 else 0.0
                                data['rel_c'] = ((data['c'] - prev_data['c']) / prev_data['c']) * 100 if prev_data['c'] != 0 else 0.0
                                data['rel_volume'] = ((data['volume'] - prev_data['volume']) / prev_data['volume']) * 100 if prev_data['volume'] != 0 else 0.0
                        
                        st.session_state.lattice_data = lattice_data
                elif message == "GEOM_THREAD_FINISHED":
                    st.session_state.geom_running = False
                else:
                    st.session_state.geom_log_messages.append(str(message))
            
            col_run1, col_run2, col_run3 = st.columns(3)
            
            with col_run1:
                if not st.session_state.geom_running:
                    if st.button("🚀 Start Geometry Optimization", type="primary"):
                        if 'file_generator' not in st.session_state:
                            st.toast("Please generate files first!", icon="⚠️")
                        else:
                            os.makedirs(work_dir_geom, exist_ok=True)
                            
                            cleanup_files = ["OSZICAR", "XDATCAR", "OUTCAR", "vasprun.xml"]

                            cleaned_files = []
                            for cleanup_file in cleanup_files:
                                cleanup_path = os.path.join(work_dir_geom, cleanup_file)
                                if os.path.exists(cleanup_path):
                                    os.remove(cleanup_path)
                                    cleaned_files.append(cleanup_file)
                            
                            if cleaned_files:
                                st.toast(f"🧹 Cleaned previous files: {', '.join(cleaned_files)}", icon="🗑️")
                            
                            with open(os.path.join(work_dir_geom, "POSCAR"), "w") as f:
                                f.write(st.session_state.file_generator['poscar'])
                            with open(os.path.join(work_dir_geom, "POTCAR"), "w") as f:
                                f.write(st.session_state.file_generator['potcar'])
                            with open(os.path.join(work_dir_geom, "KPOINTS"), "w") as f:
                                f.write(st.session_state.file_generator['kpoints'])
                            
                            st.session_state.geom_running = True
                            st.session_state.geom_log_messages = []
                            st.session_state.energy_data = {'steps': [], 'energies': [], 'scf_per_ionic': []}
                            st.session_state.lattice_data = []
                            st.session_state.max_nsw = parse_incar_nsw(st.session_state.file_generator['incar'])
                            st.session_state.max_nelm = parse_incar_nelm(st.session_state.file_generator['incar'])
                            st.session_state.num_atoms = structure.num_sites
                            st.session_state.geom_stop_event.clear()
                            
                            thread = threading.Thread(
                                target=run_geometry_optimization,
                                args=(work_dir_geom, st.session_state.vasp_command, 
                                    st.session_state.file_generator['incar'],
                                    st.session_state.geom_log_queue, st.session_state.geom_stop_event)
                            )
                            thread.start()
                            st.rerun()
                else:
                    st.success("Geometry optimization is running...")
            
            with col_run2:
                if st.session_state.geom_running:
                    if st.button("🛑 Stop Optimization", type="secondary"):
                        st.session_state.geom_stop_event.set()
                        st.toast("Stop signal sent...", icon="⏹️")
            
            with col_run3:
                if st.button("🔄 Update from Files", type = 'tertiary'):
                    try:
                        oszicar_path = os.path.join(work_dir_geom, "OSZICAR")
                        xdatcar_path = os.path.join(work_dir_geom, "XDATCAR")
                        incar_path = os.path.join(work_dir_geom, "INCAR")
                        
                        if os.path.exists(incar_path):
                            with open(incar_path, 'r') as f:
                                incar_content = f.read()
                            st.session_state.max_nsw = parse_incar_nsw(incar_content)
                            st.session_state.max_nelm = parse_incar_nelm(incar_content)
                            st.session_state.ibrion_value = parse_incar_ibrion(incar_content)
                            
                            ibrion_descriptions = {
                                -1: "No ionic relaxation",
                                0: "Molecular dynamics",
                                1: "Quasi-Newton (RMM-DIIS)",
                                2: "Conjugate gradient",
                                3: "Damped molecular dynamics"
                            }
                            ibrion_desc = ibrion_descriptions.get(st.session_state.ibrion_value, f"IBRION={st.session_state.ibrion_value}")
                            
                            st.toast(f"📋 INCAR: NSW={st.session_state.max_nsw}, NELM={st.session_state.max_nelm}, IBRION={st.session_state.ibrion_value} ({ibrion_desc})", icon="⚙️")
                        else:
                            st.session_state.max_nsw = 50
                            st.session_state.max_nelm = 60
                            st.session_state.ibrion_value = 2
                            st.toast("⚠️ INCAR not found, using defaults: NSW=50, NELM=60, IBRION=2", icon="📋")
                        
                        if os.path.exists(oszicar_path):
                            with open(oszicar_path, 'r') as f:
                                oszicar_content = f.read()
                            ionic_steps, energies, scf_per_ionic, current_ionic, current_scf = parse_oszicar_geom(oszicar_content)
                            if ionic_steps and energies:
                                st.session_state.energy_data = {
                                    'steps': ionic_steps,
                                    'energies': energies,
                                    'scf_per_ionic': scf_per_ionic
                                }
                                st.session_state.current_ionic_step = current_ionic
                                st.session_state.current_scf_step = current_scf
                                st.toast(f"✅ Updated energy data: {len(ionic_steps)} ionic steps", icon="📊")
                        
                        if os.path.exists(xdatcar_path):
                            with open(xdatcar_path, 'r') as f:
                                xdatcar_content = f.read()
                            lattice_data = parse_xdatcar(xdatcar_content)
                            if lattice_data:
                                for i, data in enumerate(lattice_data):
                                    if i == 0:
                                        data['rel_a'] = 0.0
                                        data['rel_b'] = 0.0
                                        data['rel_c'] = 0.0
                                        data['rel_volume'] = 0.0
                                    else:
                                        prev_data = lattice_data[i-1]
                                        data['rel_a'] = ((data['a'] - prev_data['a']) / prev_data['a']) * 100 if prev_data['a'] != 0 else 0.0
                                        data['rel_b'] = ((data['b'] - prev_data['b']) / prev_data['b']) * 100 if prev_data['b'] != 0 else 0.0
                                        data['rel_c'] = ((data['c'] - prev_data['c']) / prev_data['c']) * 100 if prev_data['c'] != 0 else 0.0
                                        data['rel_volume'] = ((data['volume'] - prev_data['volume']) / prev_data['volume']) * 100 if prev_data['volume'] != 0 else 0.0
                                
                                st.session_state.lattice_data = lattice_data
                                st.toast(f"✅ Updated lattice data: {len(lattice_data)} configurations", icon="🔬")
                        
                        if not os.path.exists(oszicar_path) and not os.path.exists(xdatcar_path):
                            st.toast("⚠️ No OSZICAR or XDATCAR files found in working directory", icon="📁")
                        
                    except Exception as e:
                        st.toast(f"❌ Error reading files: {str(e)}", icon="❌")
            
            if st.session_state.energy_data['steps'] or st.session_state.lattice_data:
                st.subheader("Real-time Results")
                
                col_energy, col_lattice, col_scf = st.columns(3)
                
                with col_energy:
                    if st.session_state.energy_data['steps']:
                        steps = st.session_state.energy_data['steps']
                        energies = st.session_state.energy_data['energies']
                        
                        energy_diffs_per_atom = []
                        for i in range(len(energies)):
                            if i == 0:
                                energy_diffs_per_atom.append(0.0)
                            else:
                                diff = (energies[i] - energies[i-1]) / st.session_state.num_atoms
                                energy_diffs_per_atom.append(diff)
                        
                        fig_energy = go.Figure()
                        fig_energy.add_trace(go.Scatter(
                            x=steps,
                            y=energy_diffs_per_atom,
                            mode='lines+markers',
                            name='ΔE per atom',
                            line=dict(width=3),
                            marker=dict(size=8),
                            hovertemplate='<b>Step:</b> %{x}<br><b>ΔE per atom:</b> %{y:.6f} eV/atom<extra></extra>'
                        ))
                        fig_energy.update_layout(
                            title=dict(text="Energy Change per Atom", font=dict(size=18)),
                            xaxis=dict(
                                title="Ionic Step",
                                title_font=dict(size=16, color='black'),
                                tickfont=dict(size=14, color='black')
                            ),
                            yaxis=dict(
                                title="ΔE per atom (eV/atom)",
                                title_font=dict(size=16, color='black'),
                                tickfont=dict(size=14, color='black')
                            ),
                            legend=dict(font=dict(size=12)),
                            height=350,
                            margin=dict(l=40, r=40, t=50, b=40),
                            hoverlabel=dict(
                                bgcolor="white",
                                bordercolor="black",
                                font_size=14,
                                font_family="Arial"
                            )
                        )
                        st.plotly_chart(fig_energy, use_container_width=True)
                
                with col_lattice:
                    if st.session_state.lattice_data and len(st.session_state.lattice_data) > 0:
                        df_lattice = pd.DataFrame(st.session_state.lattice_data)
                        
                        if 'rel_a' in df_lattice.columns:
                            fig_lattice = go.Figure()
                            fig_lattice.add_trace(go.Scatter(
                                x=df_lattice['config'],
                                y=df_lattice['rel_a'],
                                mode='lines+markers',
                                name='Δa (%)',
                                line=dict(width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>Step:</b> %{x}<br><b>Δa:</b> %{y:.4f}%<extra></extra>'
                            ))
                            fig_lattice.add_trace(go.Scatter(
                                x=df_lattice['config'],
                                y=df_lattice['rel_b'],
                                mode='lines+markers',
                                name='Δb (%)',
                                line=dict(width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>Step:</b> %{x}<br><b>Δb:</b> %{y:.4f}%<extra></extra>'
                            ))
                            fig_lattice.add_trace(go.Scatter(
                                x=df_lattice['config'],
                                y=df_lattice['rel_c'],
                                mode='lines+markers',
                                name='Δc (%)',
                                line=dict(width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>Step:</b> %{x}<br><b>Δc:</b> %{y:.4f}%<extra></extra>'
                            ))
                            fig_lattice.update_layout(
                                title=dict(text="Lattice Parameters Change", font=dict(size=18)),
                                xaxis=dict(
                                    title="Ionic Step",
                                    title_font=dict(size=16, color='black'),
                                    tickfont=dict(size=14, color='black')
                                ),
                                yaxis=dict(
                                    title="Change from Previous Step (%)",
                                    title_font=dict(size=16, color='black'),
                                    tickfont=dict(size=14, color='black')
                                ),
                                legend=dict(font=dict(size=12)),
                                height=350,
                                margin=dict(l=40, r=40, t=50, b=40),
                                hoverlabel=dict(
                                    bgcolor="white",
                                    bordercolor="black",
                                    font_size=14,
                                    font_family="Arial"
                                )
                            )
                            st.plotly_chart(fig_lattice, use_container_width=True)
                
                with col_scf:
                    if st.session_state.energy_data['steps'] and 'scf_per_ionic' in st.session_state.energy_data:
                        steps = st.session_state.energy_data['steps']
                        scf_counts = st.session_state.energy_data['scf_per_ionic']
                        
                        if scf_counts and len(scf_counts) == len(steps):
                            fig_scf = go.Figure()
                            fig_scf.add_trace(go.Bar(
                                x=steps,
                                y=scf_counts,
                                name='SCF Steps',
                                marker_color='orange',
                                hovertemplate='<b>Ionic Step:</b> %{x}<br><b>SCF Steps:</b> %{y}<extra></extra>'
                            ))
                            fig_scf.update_layout(
                                title=dict(text="SCF Steps per Ionic Step", font=dict(size=18)),
                                xaxis=dict(
                                    title="Ionic Step",
                                    title_font=dict(size=16, color='black'),
                                    tickfont=dict(size=14, color='black')
                                ),
                                yaxis=dict(
                                    title="SCF Steps",
                                    title_font=dict(size=16, color='black'),
                                    tickfont=dict(size=14, color='black')
                                ),
                                height=350,
                                margin=dict(l=40, r=40, t=50, b=40),
                                hoverlabel=dict(
                                    bgcolor="white",
                                    bordercolor="black",
                                    font_size=14,
                                    font_family="Arial"
                                )
                            )
                            st.plotly_chart(fig_scf, use_container_width=True)
                
                if st.session_state.lattice_data and len(st.session_state.lattice_data) > 0:
                    df_lattice = pd.DataFrame(st.session_state.lattice_data)
                    if 'rel_volume' in df_lattice.columns:
                        fig_volume = go.Figure()
                        fig_volume.add_trace(go.Scatter(
                            x=df_lattice['config'],
                            y=df_lattice['rel_volume'],
                            mode='lines+markers',
                            name='ΔVolume (%)',
                            line=dict(width=4, color='purple'),
                            marker=dict(size=10, color='purple'),
                            hovertemplate='<b>Step:</b> %{x}<br><b>ΔVolume:</b> %{y:.4f}%<extra></extra>'
                        ))
                        
            
            st.subheader("Current Status (last finished step) - read from OSZICAR and XDATCAR")
            if st.session_state.energy_data['steps'] or st.session_state.lattice_data:
                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                
                current_step = 0
                energy_change_per_atom = 0.0
                lattice_changes = {'a': 0.0, 'b': 0.0, 'c': 0.0}
                
                if st.session_state.energy_data['steps']:
                    current_step = st.session_state.energy_data['steps'][-1]
                    if len(st.session_state.energy_data['energies']) > 1:
                        current_energy = st.session_state.energy_data['energies'][-1]
                        prev_energy = st.session_state.energy_data['energies'][-2]
                        energy_change_per_atom = (current_energy - prev_energy) / st.session_state.num_atoms
                
                if st.session_state.lattice_data and 'rel_a' in pd.DataFrame(st.session_state.lattice_data).columns:
                    df_lattice = pd.DataFrame(st.session_state.lattice_data)
                    if len(df_lattice) > 0:
                        lattice_changes['a'] = df_lattice['rel_a'].iloc[-1]
                        lattice_changes['b'] = df_lattice['rel_b'].iloc[-1]
                        lattice_changes['c'] = df_lattice['rel_c'].iloc[-1]
                
                with col1:
                    st.metric(
                        label="Ionic Step",
                        value=f"{current_step}/{st.session_state.max_nsw}"
                    )
                
                with col2:
                    st.metric(
                        label="Current SCF",
                        value=f"{st.session_state.current_scf_step}/{st.session_state.max_nelm}"
                    )
                
                with col3:
                    ibrion_descriptions = {
                        -1: "No relax",
                        0: "MD",
                        1: "Quasi-Newton",
                        2: "CG",
                        3: "Damped MD"
                    }
                    ibrion_short = ibrion_descriptions.get(st.session_state.ibrion_value, f"IBRION={st.session_state.ibrion_value}")
                    st.metric(
                        label="Method",
                        value=f"IBRION={st.session_state.ibrion_value}",
                        help=ibrion_short
                    )
                
                with col4:
                    st.metric(
                        label="ΔE per atom (eV/atom)",
                        value=f"{energy_change_per_atom:.6f}"
                    )
                
                with col5:
                    st.metric(
                        label="Δa (%)",
                        value=f"{lattice_changes['a']:.4f}%"
                    )
                
                with col6:
                    st.metric(
                        label="Δb (%)",
                        value=f"{lattice_changes['b']:.4f}%"
                    )
                
                with col7:
                    st.metric(
                        label="Δc (%)",
                        value=f"{lattice_changes['c']:.4f}%"
                    )
                
                st.markdown("**Current Values:**")
                col1_val, col2_val, col3_val, col4_val, col5_val = st.columns(5)
                
                with col1_val:
                    if st.session_state.energy_data['energies']:
                        last_energy = st.session_state.energy_data['energies'][-1]
                        st.metric(
                            label="Total Energy (eV)",
                            value=f"{last_energy:.6f}"
                        )
                    else:
                        st.metric(
                            label="Total Energy (eV)",
                            value="N/A"
                        )
                
                with col2_val:
                    if st.session_state.lattice_data:
                        df_lattice = pd.DataFrame(st.session_state.lattice_data)
                        if len(df_lattice) > 0:
                            last_a = df_lattice['a'].iloc[-1]
                            st.metric(
                                label="a (Å)",
                                value=f"{last_a:.6f}"
                            )
                        else:
                            st.metric(
                                label="a (Å)",
                                value="N/A"
                            )
                    else:
                        st.metric(
                            label="a (Å)",
                            value="N/A"
                        )
                
                with col3_val:
                    if st.session_state.lattice_data:
                        df_lattice = pd.DataFrame(st.session_state.lattice_data)
                        if len(df_lattice) > 0:
                            last_b = df_lattice['b'].iloc[-1]
                            st.metric(
                                label="b (Å)",
                                value=f"{last_b:.6f}"
                            )
                        else:
                            st.metric(
                                label="b (Å)",
                                value="N/A"
                            )
                    else:
                        st.metric(
                            label="b (Å)",
                            value="N/A"
                        )
                
                with col4_val:
                    if st.session_state.lattice_data:
                        df_lattice = pd.DataFrame(st.session_state.lattice_data)
                        if len(df_lattice) > 0:
                            last_c = df_lattice['c'].iloc[-1]
                            st.metric(
                                label="c (Å)",
                                value=f"{last_c:.6f}"
                            )
                        else:
                            st.metric(
                                label="c (Å)",
                                value="N/A"
                            )
                    else:
                        st.metric(
                            label="c (Å)",
                            value="N/A"
                        )
                
                with col5_val:
                    if st.session_state.lattice_data:
                        df_lattice = pd.DataFrame(st.session_state.lattice_data)
                        if len(df_lattice) > 0:
                            last_volume = df_lattice['volume'].iloc[-1]
                            st.metric(
                                label="Volume (Ų)",
                                value=f"{last_volume:.4f}"
                            )
                        else:
                            st.metric(
                                label="Volume (Ų)",
                                value="N/A"
                            )
                    else:
                        st.metric(
                            label="Volume (Ų)",
                            value="N/A"
                        )
            else:
                st.info("Waiting for calculation data...")
            
            st.subheader("Calculation Log")
            log_text = "\n".join(st.session_state.geom_log_messages)
            st.text_area("Log Output", log_text, height=300, key="geom_log_display")
            
            if st.session_state.geom_running or rerun_needed:
                time.sleep(2)
                st.rerun()