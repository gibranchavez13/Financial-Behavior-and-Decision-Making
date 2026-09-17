import json, sys
nb = json.load(open(r"C:\ifi\5\Comportamiento\proyecto_1\simulador_sesgos_conductuales.ipynb", encoding="utf-8"))
lo, hi = int(sys.argv[1]), int(sys.argv[2])
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code" or not (lo <= i <= hi) or c["source"][0].startswith("%%writefile"):
        continue
    print(f"===== cell {i}: {c['source'][0][:70].rstrip()}")
    for o in c.get("outputs", []):
        if o["output_type"] == "stream":
            print("".join(o["text"])[:3000])
        elif o["output_type"] in ("execute_result", "display_data"):
            t = "".join(o["data"].get("text/plain", ""))
            print(t[:4500])
        elif o["output_type"] == "error":
            print("ERROR", o["ename"], o["evalue"])
