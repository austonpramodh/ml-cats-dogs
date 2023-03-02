
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from os import environ
import random
import string
from flask import jsonify, request

from backend import create_app
from .bucket import create_presigned_url
from .cat_dog import routes as car_dog_routes

# Flask constructor takes the name of
# current module (__name__) as argument.
app = create_app(environ.get('FLASK_CONFIG'))

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.


@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    url = create_presigned_url("austons-ml-bucket", "test.key")
    print(url)
    return 'Hello World'


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)


@app.route('/signed-obj-url/<bucket_name>/<obj_name>', methods=['GET'])
def ReturnJSON(bucket_name, obj_name):
    if (request.method == 'GET'):

        url = create_presigned_url(bucket_name, obj_name)
        data = {
            "success": True,
            "message": "Feteched a unique url with signature",
            "data": {
                "url": url,
                "config": app.config.get("DEBUG")
            }
        }

        return jsonify(data)


# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
