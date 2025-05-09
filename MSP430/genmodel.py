import re
import numpy as np


def generate_c_array(weights, shape, var_name, indent=4):
    """Generate C code for a properly nested multi-dimensional array initialization."""
    # Reshape the flat weights
    reshaped = np.array(weights, dtype=np.int8).reshape(shape)

    # Function to recursively build the nested representation
    def format_array(arr, depth=0):
        if depth == len(shape) - 1:
            return "{" + ", ".join([str(x) for x in arr]) + "}"

        lines = [format_array(subarr, depth + 1) for subarr in arr]
        sep = ",\n" + " " * (indent * (depth + 1))
        return "{\n" + " " * (indent * (depth + 1)) + sep.join(lines) + "\n" + " " * (indent * depth) + "}"

    # Generate the C code
    return format_array(reshaped, 0)


def parse_weights_file(filename):
    """Parse the quantweights.txt file and extract all weights"""
    with open(filename, 'r') as f:
        content = f.read()

    # Define patterns to extract each section
    patterns = {
        "weight0_1": r"model->weight0_1 weights:\s*([-\d\s,]+)",
        "weight2_3": r"model->weight2_3 weights:\s*([-\d\s,]+)",
        "weight4_5": r"model->weight4_5 weights:\s*([-\d\s,]+)",
        "weight5_6": r"model->weight5_6 weights:\s*([-\d\s,]+)",
        "bias0_1": r"model->bias0_1 weights:\s*([-\d\s,]+)",
        "bias2_3": r"model->bias2_3 weights:\s*([-\d\s,]+)",
        "bias4_5": r"model->bias4_5 weights:\s*([-\d\s,]+)",
        "bias5_6": r"model->bias5_6 weights:\s*([-\d\s,]+)",
        "w0_1s": r"model->w0_1s:\s*([-\d\.]+)",
        "w2_3s": r"model->w2_3s:\s*([-\d\.]+)",
        "w4_5s": r"model->w4_5s:\s*([-\d\.]+)",
        "w5_6s": r"model->w5_6s:\s*([-\d\.]+)",
        "b0_1s": r"model->b0_1s:\s*([-\d\.]+)",
        "b2_3s": r"model->b2_3s:\s*([-\d\.]+)",
        "b4_5s": r"model->b4_5s:\s*([-\d\.]+)",
        "b5_6s": r"model->b5_6s:\s*([-\d\.]+)"
    }

    extracted_data = {}

    # Extract each section
    for name, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            if name.startswith("w") and name.endswith("s"):  # Float scaling factors
                extracted_data[name] = float(match.group(1))
            elif name.startswith("b") and name.endswith("s"):  # Float scaling factors
                extracted_data[name] = float(match.group(1))
            else:  # Integer arrays
                # Extract all numbers
                numbers_str = match.group(1)
                # Clean up and convert to integers
                numbers = re.findall(r'-?\d+', numbers_str)
                extracted_data[name] = [int(n) for n in numbers]
        else:
            print(f"Warning: Could not find pattern for {name}")

    return extracted_data


def generate_lenet5_model(data, shapes, constants):
    """Generate the complete struct initialization code"""
    # Add header with constants defined
    code = "/* Auto-generated LeNet5 model weights */\n\n"
    code += "#ifndef LENET5_MODEL_H\n"
    code += "#define LENET5_MODEL_H\n\n"

    # Include necessary headers
    code += "#include <stdint.h>\n\n"

    # Define constants if they aren't already defined
    code += "/* Model architecture constants */\n"
    for name, value in constants.items():
        code += f"#ifndef {name}\n"
        code += f"#define {name} {value}\n"
        code += f"#endif\n"
    code += "\n"

    # Define the struct type
    code += "typedef struct LeNet5Quantized {\n"
    code += "    int8_t weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];\n"
    code += "    int8_t weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];\n"
    code += "    int8_t weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];\n"
    code += "    int8_t weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];\n"
    code += "    int8_t bias0_1[LAYER1];\n"
    code += "    int8_t bias2_3[LAYER3];\n"
    code += "    int8_t bias4_5[LAYER5];\n"
    code += "    int8_t bias5_6[OUTPUT];\n"
    code += "    float w0_1s;\n"
    code += "    float w2_3s;\n"
    code += "    float w4_5s;\n"
    code += "    float w5_6s;\n"
    code += "    float b0_1s;\n"
    code += "    float b2_3s;\n"
    code += "    float b4_5s;\n"
    code += "    float b5_6s;\n"
    code += "} LeNet5Quantized;\n\n"

    # Start the persistent model declaration
    code += "/* Model weights stored in FRAM */\n"
    code += "#pragma PERSISTENT(lenet5_model)\n"
    code += "const LeNet5Quantized lenet5_model = {\n"

    # Add each array initialization
    for name, shape in shapes.items():
        if name in data and isinstance(data[name], list):
            code += f"    /* {name} */\n"
            code += f"    {generate_c_array(data[name], shape, name)},\n\n"

    # Add float values
    for name in ["w0_1s", "w2_3s", "w4_5s", "w5_6s", "b0_1s", "b2_3s", "b4_5s", "b5_6s"]:
        if name in data:
            code += f"    /* {name} */\n"
            code += f"    {data[name]}f,\n\n"

    # Close the struct
    code = code.rstrip(",\n\n") + "\n};\n\n"

    # Close the header guard
    code += "#endif /* LENET5_MODEL_H */\n"

    return code


def main():
    # Define constants from your header file
    LENGTH_KERNEL = 5
    LENGTH_FEATURE0 = 32
    LENGTH_FEATURE1 = (LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
    LENGTH_FEATURE2 = (LENGTH_FEATURE1 >> 1)
    LENGTH_FEATURE3 = (LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
    LENGTH_FEATURE4 = (LENGTH_FEATURE3 >> 1)
    LENGTH_FEATURE5 = (LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

    constants = {
        "INPUT": 1,
        "LAYER1": 6,
        "LAYER2": 6,
        "LAYER3": 16,
        "LAYER4": 16,
        "LAYER5": 120,
        "OUTPUT": 10,
        "LENGTH_KERNEL": LENGTH_KERNEL,
        "LENGTH_FEATURE0": LENGTH_FEATURE0,
        "LENGTH_FEATURE1": LENGTH_FEATURE1,
        "LENGTH_FEATURE2": LENGTH_FEATURE2,
        "LENGTH_FEATURE3": LENGTH_FEATURE3,
        "LENGTH_FEATURE4": LENGTH_FEATURE4,
        "LENGTH_FEATURE5": LENGTH_FEATURE5,
    }

    # Define shapes for each array in the struct
    shapes = {
        "weight0_1": (constants["INPUT"], constants["LAYER1"], constants["LENGTH_KERNEL"], constants["LENGTH_KERNEL"]),
        "weight2_3": (constants["LAYER2"], constants["LAYER3"], constants["LENGTH_KERNEL"], constants["LENGTH_KERNEL"]),
        "weight4_5": (constants["LAYER4"], constants["LAYER5"], constants["LENGTH_KERNEL"], constants["LENGTH_KERNEL"]),
        "weight5_6": (constants["LAYER5"] * constants["LENGTH_FEATURE5"] * constants["LENGTH_FEATURE5"], constants["OUTPUT"]),
        "bias0_1": (constants["LAYER1"],),
        "bias2_3": (constants["LAYER3"],),
        "bias4_5": (constants["LAYER5"],),
        "bias5_6": (constants["OUTPUT"],)
    }

    # Parse the weights file
    data = parse_weights_file("quantweights.txt")

    # Generate the struct initialization
    code = generate_lenet5_model(data, shapes, constants)

    # Write to a header file
    with open("lenet5_model.h", "w") as f:
        f.write(code)

    print("Successfully generated lenet5_model.h")


if __name__ == "__main__":
    main()
