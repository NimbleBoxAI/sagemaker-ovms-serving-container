# <img alt="SageMaker" src="branding/icon/sagemaker-banner.png" height="100">

# SageMaker OpenVINO Model Server Container

SageMaker OVMS Serving Container is an a open source project that builds
docker images for running OpenVINO Model Server on
[Amazon SageMaker](https://aws.amazon.com/documentation/sagemaker/).

Supported versions of OVMS: ``2021.2.1``

ECR repositories for SageMaker built OVMS Serving Container:

- `'sagemaker-ovms-serving'` for all versions in the following AWS accounts:
  - `"553900043809"` in `"us-east-1"`;

This documentation covers building and testing these docker images.

For information about using TensorFlow Serving on SageMaker, see:
[Deploying to TensorFlow Serving Endpoints](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving.html)
in the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) documentation.

For notebook examples, see: [Amazon SageMaker Examples](https://github.com/awslabs/amazon-sagemaker-examples).

## Table of Contents

1. [Getting Started](#getting-started)
2. [Building your image](#building-your-image)
3. [Running the tests](#running-the-tests)
4. [Pre/Post-Processing](#pre/post-processing)
5. [Deploying a TensorFlow Serving Model](#deploying-a-tensorflow-serving-model)
6. [Deploying to Multi-Model Endpoint](#deploying-to-multi-model-endpoint)

## Getting Started

### Prerequisites

Make sure you have installed all of the following prerequisites on your
development machine:

- [Docker](https://www.docker.com/)
- [AWS CLI](https://aws.amazon.com/cli/)

For testing, you will also need:

- [Python 3.6](https://www.python.org/)
- [tox](https://tox.readthedocs.io/en/latest/)
- [npm](https://npmjs.org/)
- [jshint](https://jshint.com/about/)

To test GPU images locally, you will also need:

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

**Note:** Some of the build and tests scripts interact with resources in your AWS account. Be sure to
set your default AWS credentials and region using `aws configure` before using these scripts.

## Building your image

Amazon SageMaker uses Docker containers to run all training jobs and inference endpoints.

The Docker images are built from the Dockerfiles in
[docker/](https://github.com/aws/sagemaker-tensorflow-serving-container/tree/master/docker).

The Dockerfiles are grouped based on the version of TensorFlow Serving they support. Each supported
processor type (e.g. "cpu", "gpu", "ei") has a different Dockerfile in each group.

To build an image, run the `./scripts/build.sh` script:

```bash
./scripts/build.sh --version 2021.2.1 --arch cpu
```


If your are testing locally, building the image is enough. But if you want to your updated image
in SageMaker, you need to publish it to an ECR repository in your account. The
`./scripts/publish.sh` script makes that easy:

```bash
./scripts/publish.sh --version 2021.2.1 --arch cpu
```

Note: this will publish to ECR in your default region. Use the `--region` argument to
specify a different region.

### Creating a SageMaker Model

A SageMaker Model contains references to a `model.tar.gz` file in S3 containing serialized model data, and a Docker image used to serve predictions with that model.

You must package the contents in a model directory (including models, inference.py and external modules) in .tar.gz format in a file named "model.tar.gz" and upload it to S3. If you're on a Unix-based operating system, you can create a "model.tar.gz" using the `tar` utility:

```
tar -czvf model.tar.gz 12345 code
```

where "12345" is your TensorFlow serving model version which contains your SavedModel.

After uploading your `model.tar.gz` to an S3 URI, such as `s3://your-bucket/your-models/model.tar.gz`, create a [SageMaker Model](https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateModel.html) which will be used to generate inferences. Set `PrimaryContainer.ModelDataUrl` to the S3 URI where you uploaded the `model.tar.gz`, and set `PrimaryContainer.Image` to an image following this format:

```
520713654638.dkr.ecr.{REGION}.amazonaws.com/sagemaker-tensorflow-serving:{SAGEMAKER_TENSORFLOW_SERVING_VERSION}-{cpu|gpu}
```

```
763104351884.dkr.ecr.{REGION}.amazonaws.com/tensorflow-inference:{TENSORFLOW_INFERENCE_VERSION}-{cpu|gpu}

The code examples below show how to create a SageMaker Model from a `model.tar.gz` containing a OpenVINO IR model using the AWS CLI (though you can use any language supported by the [AWS SDK](https://aws.amazon.com/tools/)) and the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk).

#### SageMaker Python SDK

```python
import os
import sagemaker
from sagemaker.tensorflow.serving import Model

sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::038453126632:role/service-role/AmazonSageMaker-ExecutionRole-20180718T141171'
bucket = 'am-datasets'
prefix = 'sagemaker/high-throughput-tfs-batch-transform'
s3_path = 's3://{}/{}'.format(bucket, prefix)

model_data = sagemaker_session.upload_data('model.tar.gz',
                                           bucket,
                                           os.path.join(prefix, 'model'))

# The "Model" object doesn't create a SageMaker Model until a Transform Job or Endpoint is created.
tensorflow_serving_model = Model(model_data=model_data,
                                 role=role,
                                 framework_version='1.13',
                                 sagemaker_session=sagemaker_session)

```

After creating a SageMaker Model, you can refer to the model name to create Transform Jobs and Endpoints. Code examples are given below.


#### CLI
```bash
TRANSFORM_JOB_NAME="tfs-transform-job"
TRANSFORM_S3_INPUT="s3://my-sagemaker-input-bucket/sagemaker-transform-input-data/"
TRANSFORM_S3_OUTPUT="s3://my-sagemaker-output-bucket/sagemaker-transform-output-data/"

TRANSFORM_INPUT_DATA_SOURCE={S3DataSource={S3DataType="S3Prefix",S3Uri=$TRANSFORM_S3_INPUT}}
CONTENT_TYPE="application/x-image"

INSTANCE_TYPE="ml.p2.xlarge"
INSTANCE_COUNT=2

MAX_PAYLOAD_IN_MB=1
MAX_CONCURRENT_TRANSFORMS=16

aws sagemaker create-transform-job \
    --model-name $MODEL_NAME \
    --transform-input DataSource=$TRANSFORM_INPUT_DATA_SOURCE,ContentType=$CONTENT_TYPE \
    --transform-output S3OutputPath=$TRANSFORM_S3_OUTPUT \
    --transform-resources InstanceType=$INSTANCE_TYPE,InstanceCount=$INSTANCE_COUNT \
    --max-payload-in-mb $MAX_PAYLOAD_IN_MB \
    --max-concurrent-transforms $MAX_CONCURRENT_TRANSFORMS \
    --transform-job-name $JOB_NAME
```

#### SageMaker Python SDK

```python
output_path = 's3://my-sagemaker-output-bucket/sagemaker-transform-output-data/'
tensorflow_serving_transformer = tensorflow_serving_model.transformer(
                                     framework_version = '1.12',
                                     instance_count=2,
                                     instance_type='ml.p2.xlarge',
                                     max_concurrent_transforms=16,
                                     max_payload=1,
                                     output_path=output_path)

input_path = 's3://my-sagemaker-input-bucket/sagemaker-transform-input-data/'
tensorflow_serving_transformer.transform(input_path, content_type='application/x-image')
```

### Creating an Endpoint

A SageMaker Endpoint hosts your TensorFlow Serving model for real-time inference. The [InvokeEndpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/API_runtime_InvokeEndpoint.html) API
is used to send data for predictions to your TensorFlow Serving model.

#### AWS CLI

```bash
ENDPOINT_CONFIG_NAME="my-endpoint-config"
VARIANT_NAME="TFS"
INITIAL_INSTANCE_COUNT=1
INSTANCE_TYPE="ml.p2.xlarge"
aws sagemaker create-endpoint-config \
    --endpoint-config-name $ENDPOINT_CONFIG_NAME \
    --production-variants VariantName=$VARIANT_NAME,ModelName=$MODEL_NAME,InitialInstanceCount=$INITIAL_INSTANCE_COUNT,InstanceType=$INSTANCE_TYPE

ENDPOINT_NAME="my-tfs-endpoint"
aws sagemaker create-endpoint \
    --endpoint-name $ENDPOINT_NAME \
    --endpoint-config-name $ENDPOINT_CONFIG_NAME

BODY="fileb://myfile.jpeg"
CONTENT_TYPE='application/x-image'
OUTFILE="response.json"
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name $ENDPOINT_NAME \
    --content-type=$CONTENT_TYPE \
    --body $BODY \
    $OUTFILE
```

## License

This library is licensed under the Apache 2.0 License.

