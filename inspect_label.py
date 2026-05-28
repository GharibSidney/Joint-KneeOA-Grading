import math
LABEL_FILE     = "/data/net/datasets/MOST/MOST_labels.csv"            # e.g. "/data/.../most_labels.csv"<

def process_kl(value):
    if value is None or value == "":
        return -999
    
    try:
        return max(0, math.floor(float(value) + 0.5))
    except (ValueError, TypeError):
        return -999
    
def load_labels(label_file=LABEL_FILE):
    """
    Returns a dict keyed by (subject_id, image_id) →
        { 'kl_L': int, 'kl_R': int, 'aux_L': list, 'aux_R': list }

    Extend this function once you have the MOST label file.
    Currently returns an empty dict so all patients get kl_grade = -999
    and are saved without labels (or skipped, depending on your preference).
    """
    if label_file is None:
        return {}

    import pandas as pd
    df = pd.read_csv(label_file)
    labels = {}

    for _, row in df.iterrows():
        key = str(row["MOSTID"])

        labels[key] = {
            "kl_L": process_kl(row.get("V0XLKL")),
            "kl_R": process_kl(row.get("V0XRKL")),
        }
    return labels

if __name__ == "__main__":
    labels = load_labels()
    # print(labels["MOSTID"])
    for label in labels:
        print(label)
        print(labels[label])
        break
