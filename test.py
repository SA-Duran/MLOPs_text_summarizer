from datasets import load_dataset, load_from_disk



ds = load_from_disk("artifacts/data_transformation/samsum_dataset")
print(ds)  # expect DatasetDict with train/validation
print("train columns:", ds["train"].column_names[:20])

# peek first row
row = ds["train"][0]
for k, v in row.items():
    print(k, type(v), (len(v) if isinstance(v, list) else v))
