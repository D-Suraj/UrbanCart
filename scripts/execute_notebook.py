"""Execute the notebook in a fresh local Jupyter kernel and retain its outputs."""
import json
import sys
from pathlib import Path

from jupyter_client import KernelManager


def execute(path, selected_ids=None, bootstrap=None):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv[0] = sys.executable
    manager.start_kernel(cwd=str(path.parent))
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=30)
        if bootstrap:
            client.execute(bootstrap, allow_stdin=False, stop_on_error=True)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            if selected_ids is not None and cell["id"] not in selected_ids:
                continue
            cell["outputs"] = []
            msg_id = client.execute("".join(cell["source"]), allow_stdin=False, stop_on_error=True)
            while True:
                message = client.get_iopub_msg(timeout=120)
                if message["parent_header"].get("msg_id") != msg_id:
                    continue
                kind, data = message["msg_type"], message["content"]
                if kind == "status" and data["execution_state"] == "idle":
                    break
                if kind == "execute_input":
                    cell["execution_count"] = data["execution_count"]
                elif kind == "error":
                    raise RuntimeError(f"{cell['id']}: {data['ename']}: {data['evalue']}")
                elif kind == "stream":
                    cell["outputs"].append({"output_type": kind, "name": data["name"], "text": data["text"]})
                elif kind in {"display_data", "execute_result"}:
                    output = {"output_type": kind, "data": data["data"], "metadata": data["metadata"]}
                    if kind == "execute_result":
                        output["execution_count"] = data["execution_count"]
                    cell["outputs"].append(output)
            print(f"{cell['id']}: {len(cell['outputs'])} outputs", flush=True)
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    execute(Path(__file__).resolve().parents[1] / "UrbanCart_Submission.ipynb")
