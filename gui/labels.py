def get_label_mapping(output_size):
    # MNIST (0-9)
    if output_size == 10:
        return {i: str(i) for i in range(10)}

    # EMNIST Letters (1-26)
    # The 'letters' split has 27 classes. Index 0 is unused/background, 1=A, 2=B, etc.
    elif output_size == 27:
        mapping = {0: "N/A"}
        for i in range(1, 27):
            # ASCII: 65 is 'A', so index 1 -> 65, index 2 -> 66
            mapping[i] = chr(64 + i)
        return mapping

    # EMNIST Balanced (0-9, A-Z, a-b...) - 47 classes
    # Mapping based on EMNIST paper/standard
    elif output_size == 47:
        # 0-9
        mapping = {i: str(i) for i in range(10)}
        # 10-35: A-Z
        for i in range(26):
            mapping[10 + i] = chr(65 + i)
        # 36-46: a, b, d, e, f, g, h, n, q, r, t (some lowercase merge with upper)
        # The specific mapping for balanced is complex, usually provided by a file.
        # For now, we handle the most common case or generic fallback.
        pass

    # Fallback: just return index as string
    return {i: str(i) for i in range(output_size)}
