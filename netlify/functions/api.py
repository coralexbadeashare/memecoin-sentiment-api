import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mangum import Mangum
from main import app

_handler = Mangum(app, lifespan="off")

def handler(event, context):
    return _handler(event, context)
