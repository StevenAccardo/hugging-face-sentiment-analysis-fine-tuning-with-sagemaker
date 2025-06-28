from sagemaker.huggingface import HuggingFace
from sagemaker import Session

sess = Session()
role = 'arn:aws:iam::215840227122:role/service-role/AmazonSageMaker-ExecutionRole-20250626T132824'

estimator = HuggingFace(
    entry_point='train.py',
    source_dir='.',
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    role=role,
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39'
)

estimator.fit()