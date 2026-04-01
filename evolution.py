import numpy as np
import os

def mock_pssm_generator(fasta_file, output_dir): #id desde el fasta
    struct_id = os.path.splitext(os.path.basename(fasta_file))[0]
    output_path = os.path.join(output_dir, f"{struct_id}.pssm")
    
    # Read the FASTA file and extract the sequence
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
        # Clean the file: ignore the line starting with ">" and join all sequence lines together
        seq = "".join([l.strip() for l in lines if not l.startswith(">")])
    
    # Create the output file (CSV format)
    with open(output_path, "w") as f:
        # Write the header: Index, Amino Acid, Conservation Score
        f.write("Residue_Index,Residue,Conservation_Score\n")
        
        # For each amino acid in the sequence:
        for i, aa in enumerate(seq):
            # Generate a random conservation score between 0 and 1 (simulating a PSSM score)
            fake_score = np.random.random() 
            f.write(f"{i+1},{aa},{fake_score:.4f}\n")

if __name__ == "__main__":
    # All your execution logic stays here
    fasta_folder = "data/fastas"
    pssm_folder = "data/pssms"
    os.makedirs(pssm_folder, exist_ok=True)

    if os.path.exists(fasta_folder):
        for file in os.listdir(fasta_folder):
            if file.endswith(".fasta"):
                input_path = os.path.join(fasta_folder, file)
                mock_pssm_generator(input_path, pssm_folder)
                print(f"PSSM generated for: {file}")
    else:
        print(f"Directory {fasta_folder} not found!")