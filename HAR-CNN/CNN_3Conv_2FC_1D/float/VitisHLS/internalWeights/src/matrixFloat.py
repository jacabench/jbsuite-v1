import struct
import os

# --- CONFIGURATION ---
# Map your binary filenames to the C++ variable names you want
FILES_TO_CONVERT = {
    "W1.bin":    "W1",
    "B1.bin":    "B1",
    "W2.bin":    "W2",
    "B2.bin":    "B2",
    "W3.bin":    "W3",
    "B3.bin":    "B3",
    "W_fc1.bin": "W_fc1",
    "B_fc1.bin": "B_fc1",
    "W_fc2.bin": "W_fc2",
    "B_fc2.bin": "B_fc2"
}

HEADER_NAME = "weights.h"

def read_bin_floats(filepath):
    """Reads a binary file of 32-bit floats."""
    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found. Skipping.")
        return []

    with open(filepath, 'rb') as f:
        content = f.read()
        # Calculate how many floats are in the file (4 bytes per float)
        num_floats = len(content) // 4
        # Unpack binary data into a tuple of floats
        return struct.unpack('f' * num_floats, content)

def write_header(file_map, output_header):
    with open(output_header, 'w') as f:
        f.write("#ifndef WEIGHTS_H\n")
        f.write("#define WEIGHTS_H\n\n")
        f.write('#include "conv_tb1.h" // Ensure type_t is defined here\n\n')

        for bin_file, var_name in file_map.items():
            data = read_bin_floats(bin_file)
            if not data:
                continue

            print(f"Processing {bin_file} -> {var_name} [{len(data)} elements]...")

            # Write the array declaration
            # formatting as const type_t ensures it goes into BRAM/ROM in HLS
            f.write(f"const type_t {var_name}[{len(data)}] = {{\n")

            # Write data in comma-separated chunks
            for i, val in enumerate(data):
                f.write(f"{val: .8f}") # Write float with high precision
                if i < len(data) - 1:
                    f.write(", ")
                if (i + 1) % 10 == 0: # Newline every 10 elements for readability
                    f.write("\n    ")

            f.write("\n};\n\n")

        f.write("#endif // WEIGHTS_H\n")
    print(f"\nSUCCESS: Generated {output_header}")

if __name__ == "__main__":
    write_header(FILES_TO_CONVERT, HEADER_NAME)