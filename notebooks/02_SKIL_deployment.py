from skil import Skil, WorkSpace, Experiment, Model, Deployment

skil_server = Skil(
    host          = 'localhost',
    port          = 9008,
    user_id       = 'admin',
    password      = 'Skymind'
)

work_space = WorkSpace(
    skil          = skil_server, 
    name          = 'vgg-workspace'
)

experiment = Experiment(
    work_space    = work_space, 
    name          = 'vgg-experiment', 
    experiment_id = 'vgg-experiment-01'
)

model = Model(
    model         = 'vgg_face_descriptor.h5', 
    name          = 'vgg-model',
    model_id      = 'vgg-model-01',
    experiment    = experiment
)

deployment = Deployment(
    skil          = skil_server,
    name          = 'vgg-deployment'
)

service = model.deploy(deployment)