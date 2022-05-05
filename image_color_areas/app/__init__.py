__version__ = '0.1.0'

from flask import Flask
from image_color_areas.config import Config

app=Flask(__name__,
			static_url_path='', 
            static_folder='static',
            template_folder='templates',
)
app.config.from_object(Config)



from image_color_areas.app import routes
