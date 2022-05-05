from flask_wtf import FlaskForm
from wtforms import SubmitField, FileField, DecimalField, IntegerRangeField, SelectField, BooleanField
from wtforms.widgets import ListWidget, CheckboxInput
from wtforms.validators import DataRequired

class ImageColorizingForm(FlaskForm):


	image = FileField(
						"Upload image: ",
						validators=[DataRequired()],
						)

	# grades:int,fast=False,palet=palet
	# threshold:int=5,contrast:float = 1.0,noise_level:float=0.2,noise_mag:float=0.025

	grades = IntegerRangeField(
								"Grades: ",
								validators=[DataRequired()],
								default=16,
								)
						
	fast = BooleanField(
						"Fast: ",
						validators=[DataRequired()],
						default=True,
						)


	palet = 0

	threshold = DecimalField(
							"Threshold: ",
							validators=[DataRequired()],
							default=25,
							)

	contrast = DecimalField(
							"Contrast: ",
							validators=[DataRequired()],
							default=1.0,
							)

	noise_level = DecimalField(
							"Noise level: ",
							validators=[DataRequired()],
							default=0.2,
							)

	noise_mag = DecimalField(
							"Noise mag: ",
							validators=[DataRequired()],
							default=0.025,
							)







	submit = SubmitField("Submit")
