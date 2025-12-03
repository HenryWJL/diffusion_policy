from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class RobosuiteImageRunner(BaseImageRunner):

  def __init__(self, *args, **kwargs):
    super().__init__(output_dir=None)

  def run(self):
    return dict()
