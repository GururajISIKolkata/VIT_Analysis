import json
import os


def get_root_dir():
    return os.path.dirname(os.path.dirname(__file__))


def get_results_dir():
    root = get_root_dir()
    results_dir = os.path.join(root, "Results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_results(data, filename):
    results_dir = get_results_dir()

    path = os.path.join(results_dir, f"{filename}.json")

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_plot(filename, fig=None, dpi=300):
    results_dir = get_results_dir()
    path = os.path.join(results_dir, f"{filename}.png")

    if fig is None:
        # Lazy import avoids forcing matplotlib dependency when not saving plots.
        import matplotlib.pyplot as plt
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    return path


def load_results(filename):
    root = get_root_dir()
    path = os.path.join(root, "Results", f"{filename}.json")

    with open(path, "r") as f:
        return json.load(f)