"""Virtual subject sources: SDF images and graphs generated in memory (e.g. synthetic perturbations), addressed by
'<prefix>:<id>' strings in the SDF / Graph columns. Nothing is written to disk; every process regenerates them."""
import nibabel as nib

PROVIDERS = {}  # prefix -> callable(path) -> (nibabel image, graph dict as graph_coords.load_graph)


def register(prefix, provider):
    PROVIDERS[prefix] = provider


def is_virtual(path):
    return isinstance(path, str) and ':' in path and path.split(':', 1)[0] in PROVIDERS


def load_image(path):
    return PROVIDERS[path.split(':', 1)[0]](path)[0] if is_virtual(path) else nib.load(path)


def virtual_graph(path):
    graph = PROVIDERS[path.split(':', 1)[0]](path)[1]
    return {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in graph.items()}
