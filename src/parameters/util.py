'''
Created on Apr 17, 2021

@author: immanueltrummer
'''
import re

def is_numerical(value):
    """ Returns true iff value is number, optionally followed by unit. """
    return True if re.match(r'\d+%$', str(value)) else False

def decompose_val(value: str):
    """ Decomposes parameter value into float value and unit. 
    
    Args:
        value: value string containing digits, optionally followed by unit or %
    
    Returns:
        Tuple containing float value and unit (string)
    """
    if re.match(r'\d+%$', value):
        raw_float_val = float(re.sub(r'(\d+)%', r'\g<1>', value))
        float_val = raw_float_val/100.0
        val_unit = ''
    else:
        val_regex = r'(\d+)(\w+)'
        float_val = float(re.sub(val_regex, r'\g<1>', value))
        val_unit = re.sub(val_regex, r'\g<2>', value)
    return float_val, val_unit
#
# def scale(value, factor):
    # """ Scales given parameter value. """
    # # Return original value for scaling factor 1
    # if factor == 1:
        # return value
    # # Is it numerical parameter, followed by unit?
    # if re.match(r'\d+[a-zA-Z]*', value):
        # # Separate number from unit, scale number
        # digits = re.sub(r'(\d+)([a-zA-Z]*)', r'\g<1>', value)
        # units = re.sub(r'(\d+)([a-zA-Z]*)', r'\g<1>', value)
        # scaled = int(digits) * factor
        # return str(scaled) + units
    # else:
        # # No scaling possible
        # return value