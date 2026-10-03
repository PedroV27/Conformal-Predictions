"""
Ejecuta el notebook de principio a fin y guarda las salidas dentro del mismo archivo.
Uso:  python scripts/ejecutar_notebook.py conformal_prediction_ejemplo.ipynb
(Equivale a "Run All" en Jupyter; útil para dejar el notebook ya ejecutado antes de subirlo a GitHub.)
"""
import json
import sys
import time
from pathlib import Path

from jupyter_client.manager import start_new_kernel

path = Path(sys.argv[1]).resolve()
nb = json.loads(path.read_text(encoding="utf-8"))
km, kc = start_new_kernel(kernel_name="python3", cwd=str(path.parent))
kc.execute("%matplotlib inline", store_history=False)
kc.get_shell_msg(timeout=60)
t0 = time.time()
count = 0
ok = True
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    count += 1
    msg_id = kc.execute("".join(cell["source"]))
    outputs = []
    while True:
        msg = kc.get_iopub_msg(timeout=900)
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        t, c = msg["msg_type"], msg["content"]
        if t == "status" and c["execution_state"] == "idle":
            break
        if t == "stream":
            if outputs and outputs[-1].get("output_type") == "stream" and outputs[-1]["name"] == c["name"]:
                outputs[-1]["text"] += c["text"]
            else:
                outputs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
        elif t in ("execute_result", "display_data"):
            o = {"output_type": t, "data": c["data"], "metadata": c.get("metadata", {})}
            if t == "execute_result":
                o["execution_count"] = count
            outputs.append(o)
        elif t == "error":
            outputs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"], "traceback": c["traceback"]})
            ok = False
            print("ERROR en celda", count, c["ename"], c["evalue"])
    for o in outputs:
        if o["output_type"] == "stream":
            o["text"] = o["text"].splitlines(keepends=True)
    cell["outputs"] = outputs
    cell["execution_count"] = count
    if not ok:
        break
kc.stop_channels()
km.shutdown_kernel(now=True)
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{'OK' if ok else 'FALLO'}: {count} celdas de codigo en {time.time() - t0:.1f} s")
