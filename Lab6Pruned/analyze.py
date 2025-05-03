def analyze_weights(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Split by layer headers
    layers = content.split('model->')

    total_zeros = 0
    total_nonzeros = 0
    layer_stats = []

    for layer in layers[1:]:  # Skip first empty split
        if not layer.strip():
            continue

        lines = layer.strip().split('\n')
        layer_name = lines[0].split()[0]  # Extract layer name like "weight0_1"

        # Extract all numbers from the layer
        numbers = []
        for line in lines[1:]:  # Skip header line
            # Split by whitespace and convert to float
            values = line.strip().split()
            for val in values:
                try:
                    numbers.append(float(val))
                except ValueError:
                    continue  # Skip non-numeric values

        # Count zeros and non-zeros
        zeros = sum(1 for n in numbers if n == 0.0)
        nonzeros = sum(1 for n in numbers if n != 0.0)

        # Calculate statistics
        total_values = len(numbers)
        sparsity = (zeros / total_values * 100) if total_values > 0 else 0

        layer_stats.append({
            'name': layer_name,
            'zeros': zeros,
            'nonzeros': nonzeros,
            'total': total_values,
            'sparsity': sparsity
        })

        total_zeros += zeros
        total_nonzeros += nonzeros

    # Print results
    print("Layer-by-layer analysis:")
    print("-" * 60)
    for stat in layer_stats:
        print(f"Layer: {stat['name']}")
        print(f"  Zeros: {stat['zeros']}")
        print(f"  Non-zeros: {stat['nonzeros']}")
        print(f"  Total values: {stat['total']}")
        print(f"  Sparsity: {stat['sparsity']:.2f}%")
        print("-" * 60)

    # Overall summary
    total_values = total_zeros + total_nonzeros
    overall_sparsity = (total_zeros / total_values *
                        100) if total_values > 0 else 0

    print("\nOVERALL SUMMARY:")
    print("=" * 60)
    print(f"Total zeros: {total_zeros}")
    print(f"Total non-zeros: {total_nonzeros}")
    print(f"Total values: {total_values}")
    print(f"Overall sparsity: {overall_sparsity:.2f}%")
    print(f"Number of layers: {len(layer_stats)}")

    # Additional statistics
    if layer_stats:
        avg_sparsity = sum(stat['sparsity']
                           for stat in layer_stats) / len(layer_stats)
        max_sparsity_layer = max(layer_stats, key=lambda x: x['sparsity'])
        min_sparsity_layer = min(layer_stats, key=lambda x: x['sparsity'])

        print(f"\nAverage layer sparsity: {avg_sparsity:.2f}%")
        print(f"Most sparse layer: {max_sparsity_layer['name']} ({
              max_sparsity_layer['sparsity']:.2f}%)")
        print(f"Least sparse layer: {min_sparsity_layer['name']} ({
              min_sparsity_layer['sparsity']:.2f}%)")


if __name__ == "__main__":
    analyze_weights("test.txt")
