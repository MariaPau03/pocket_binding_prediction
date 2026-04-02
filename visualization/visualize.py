import os           
import subprocess
import colorsys
import shutil

def visualize_clusters(pdb_id, top_n=1, rotate=False, results_dir="results"):

    # ── Updated folder paths ──────────────────────────────────────────────
    pdbs_dir  = os.path.join(results_dir, "pdbs")
    logs_dir  = os.path.join(results_dir, "logs")
    cmds_dir  = os.path.join(results_dir, "cmd_scripts")
    shots_dir = os.path.join(results_dir, "screenshots")

    for folder in [pdbs_dir, logs_dir, cmds_dir, shots_dir]:
        os.makedirs(folder, exist_ok=True)

    # ── Read log ──────────────────────────────────────────────────────────
    log_path = os.path.join(logs_dir, f"{pdb_id}_results.log")  # ← updated
    all_clusters = []
    with open(log_path, "r") as f:
        for line in f:
            if line.startswith("Cluster"):
                parts = line.split(":")
                label = int(parts[0].split()[1])
                score = float(parts[1].split("=")[1])
                all_clusters.append((label, score))

    sorted_clusters = sorted(all_clusters, key=lambda x: x[1], reverse=True)
    if top_n is not None:
        sorted_clusters = sorted_clusters[:int(top_n)]

    # ── Find original PDB ─────────────────────────────────────────────────
    original_input_path = None
    for directory in [results_dir, "data", "data/subset_holo4k", "."]:
        test_path = os.path.join(directory, f"{pdb_id}.pdb")
        if os.path.exists(test_path):
            original_input_path = test_path
            break

    if original_input_path is None:
        raise FileNotFoundError(f"{pdb_id}.pdb not found")

    # ── Cluster files now in pdbs/ ────────────────────────────────────────
    cluster_files = [
        os.path.join(pdbs_dir, f"{pdb_id}cluster_{label}.pdb")  # ← updated
        for label, _ in sorted_clusters
    ]

    # ── Write .cmd script into cmd_scripts/ ───────────────────────────────
    chimera_script_path = os.path.join(cmds_dir, f"visualize_{pdb_id}.cxc")  # ← updated
    with open(chimera_script_path, "w") as f:
        f.write(f'open "{os.path.abspath(original_input_path)}"\n')
        f.write("preset 'publication 2'\n")

        for i, cluster_file in enumerate(cluster_files):
            if not os.path.exists(cluster_file):
                print(f"  [warning] Skipping missing cluster file: {cluster_file}")
                continue

            hue = (i * 0.07) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)

            residues_seen = set()
            with open(cluster_file, "r") as cf:
                for line in cf:
                    if line.startswith("ATOM"):
                        resnum = line[22:26].strip()
                        chain  = line[21].strip()
                        res_id = f"{chain}:{resnum}"   # ChimeraX format

                        if res_id not in residues_seen:
                            f.write(f"surface /{res_id} transparency 50\n")   # ← surface + transparency in one line
                            f.write(f"color /{res_id} {r:.2f},{g:.2f},{b:.2f}\n")
                            residues_seen.add(res_id)

        if rotate:
            f.write("cofr\n")
            f.write("reset\n")
            f.write("turn x 90\n")
            f.write("turn y 180\n")

        f.write("view\n")
<<<<<<< Updated upstream
        
=======
        #f.write("exit\n")
>>>>>>> Stashed changes

        # ── Screenshot saved into screenshots/ ────────────────────────────
        screenshot_path = os.path.abspath(
            os.path.join(shots_dir, f"{pdb_id}_clusters.png")  # ← updated
        )
<<<<<<< Updated upstream
        # f.write(f'save "{screenshot_path}" format png\n')
=======
        f.write(f'save "{screenshot_path}" format png\n')
>>>>>>> Stashed changes
        f.write("exit\n")

    # ── Run ChimeraX headless ─────────────────────────────────────────────

<<<<<<< Updated upstream
=======
    subprocess.run([
       "/usr/bin/chimerax",
        "--nogui",          # <- nogui  no window opens, it just executes the commands and saves the PNG automatically. You won't see anything on screen but the screenshot appears in results/screenshots/ when it's done.
        "--offscreen",      # <- enables rendering without a display
        "--script", chimera_script_path
     ])

>>>>>>> Stashed changes
    try:
        subprocess.run([
            "/usr/bin/chimerax",
            "--nogui",
            "--offscreen",
            "--script", chimera_script_path
        ], timeout=30)
    except Exception as e:
        print(f"  [warning] ChimeraX visualization skipped: {e}")

    # try:
    #     subprocess.run([
    #         "/Applications/ChimeraX-1.11.1.app/Contents/MacOS/ChimeraX",
    #         "--nogui",
    #         "--offscreen",    # ← add this
    #         "--script", chimera_script_path
    #     ], timeout=60)        # ← increase timeout for rendering
    # except Exception as e:
    #     print(f"  [warning] ChimeraX visualization skipped: {e}")