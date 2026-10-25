from pathlib import Path
from mjlab.rl.runner import MjlabOnPolicyRunner


class AMPOnPolicyRunner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
        super().save(path, infos)
        self.export_policy_to_onnx(str(Path(path).parent), "policy.onnx")
