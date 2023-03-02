

from os import environ


class Config(object):
    aws_access_key_id = environ.get('aws_access_key_id')
    aws_secret_access_key = environ.get('aws_secret_access_key')
    DEBUG = False
    DEVELOPMENT = False


class DevConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class ProdConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


config = {
    'dev': DevConfig,
    'prod': ProdConfig,
    'default': DevConfig,
}
