import json

splits = json.load(open("splits.json"))
sets = {name: set(pid.split("_")[0] for pid in pids) for name, pids in splits.items()}

for patient in sets["train"]:
    if patient in sets["val"]:
        print(f"LEAK: {patient} is in train AND val")

for patient in sets["train"]:
    if patient in sets["test"]:
        print(f"LEAK: {patient} is in train AND test")


# for patient in sets["val"]:
#     if patient in sets["test"]:
#         print(f"LEAK: {patient} is in val AND test")