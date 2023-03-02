import logging
import boto3
import botocore
from botocore.exceptions import ClientError


def create_presigned_url(bucket_name, object_name, expiration=600):
    # Generate a presigned URL for the S3 object
    s3_client = boto3.client('s3',
                             region_name="fr-par",
                             config=boto3.session.Config(
                                 signature_version='s3v4',
                                 #  addressing_style="virtual"
                             ),
                             aws_access_key_id="SCW8R5G497CM3A5D2WG5",
                             aws_secret_access_key="6929b364-4146-4eb6-afe2-14cab4c8b611",
                             endpoint_url='https://s3.fr-par.scw.cloud',
                             )
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except Exception as e:
        print(e)
        logging.error(e)
        return "Error"

    # The response contains the presigned URL
    print(response)
    return response
