'''
Created on Apr 17, 2021

@author: immanueltrummer
'''
import re

def scale(value, factor):
    """ Scales given parameter value. """
    # Return original value for scaling factor 1
    if factor == 1:
        return value
    # Is it numerical parameter, followed by unit?
    if re.match(r'\d+[a-zA-Z]*', value):
        # Separate number from unit, scale number
        digits = re.sub(r'(\d+)([a-zA-Z]*)', r'\g<1>', value)
        units = re.sub(r'(\d+)([a-zA-Z]*)', r'\g<1>', value)
        scaled = int(digits) * factor
        return str(scaled) + units
    else:
        # No scaling possible
        return value