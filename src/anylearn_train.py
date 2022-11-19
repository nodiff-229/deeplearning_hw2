import logging

logging.basicConfig(level=logging.INFO)
from anylearn.applications.quickstart import quick_train

from anylearn.config import init_sdk

from anylearn.interfaces.resource import SyncResourceUploader

init_sdk('https://anylearn.nelbds.cn/', "nodiff", "bleacher1032")

task, _, _, _ = quick_train(algorithm_name="hw2_model", algorithm_dir="../src",
                            entrypoint="python main.py",
                            output="checkpoints",
                            algorithm_force_update=True,

                            dataset_id="DSET83a377904cfd979991a4431b8877", dataset_hyperparam_name='data_dir',
                            hyperparams={
                                'save_dir': 'checkpoints',
                            },
                            mirror_name="QUICKSTART_PYTORCH1.9.0_CUDA11",
                            resource_uploader=SyncResourceUploader(),
                            resource_request=[{'DL2022-2':
                                                   {'RTX-3090-shared': 1,
                                                    'CPU': 4,
                                                    'Memory': 32,}
                                               }],
                            )
print(task)