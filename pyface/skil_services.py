
from skil import Skil, get_service_by_id

class SkilService:

    def __init__(self):

        self.skil_server = Skil(
            host          = 'localhost',
            port          = 9008,
            user_id       = 'admin',
            password      = 'Skymind'
        )

        self.experiment_id   = 'vgg-experiment-01'
        self.model_id        = 'vgg-model-01'
        self.deployment_name = 'vgg-deployment'

        deployments = self.skil_server.api.deployments()
        deployment = next(deployment for deployment in deployments if deployment.name == self.deployment_name)
        self.deployment_id = deployment.id

        self.service = get_service_by_id(
            self.skil_server,
            self.experiment_id,
            self.model_id,
            self.deployment_id
        )