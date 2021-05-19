'''
Created on Apr 17, 2021

@author: immanueltrummer
'''
import configparser
import re

param_reg = r'[a-z_]+_[a-z]+'
value_reg = r'(\d+(\.\d+)?)(%|\w*)'

def is_numerical(value):
    """ Returns true iff value is number, optionally followed by unit. """
    return True if re.match(value_reg + r'$', str(value)) else False

def read_numerical(file_path):
    """ Reads all numerical parameters from a configuration file. 
    
    Args:
        file_path: path to configuration file to read
    
    Returns:
        List of numerical parameters read from file
    """
    config = configparser.ConfigParser()
    with open(file_path) as stream:
        # Make sure we integrate parameters into section
        config.read_string("[dummysection]\n" + stream.read())
    num_params = []
    for section in config.sections():
        for param in config[section]:
            value = config[section][param].split()[0]
            if is_numerical(value):
                num_params.append(param)
    return num_params

def decompose_val(value: str):
    """ Decomposes parameter value into float value and unit. 
    
    Args:
        value: value string containing digits, optionally followed by unit or %
    
    Returns:
        Tuple containing float value and unit (string)
    """
    str_value = str(value)
    float_val = float(re.sub(value_reg, r'\g<1>', str_value))
    unit = re.sub(value_reg, r'\g<3>', str_value)
    if unit == '%':
        return float_val / 100.0, ''
    else:
        return float_val, unit

def convert_to_bytes(value):
    """ Try converting value with unit into byte value. 
    
    Args:
        value: a number followed by a unit
        
    Returns:
        byte size if successful or None
    """
    if is_numerical(value):
        number, unit = decompose_val(value)
        low_unit = unit.lower()
        if len(low_unit) == 0:
            return number
        elif 'g' in low_unit:
            return number * 1000000000
        elif 'm' in low_unit:
            return number * 1000000
        elif 'k' in low_unit:
            return number * 1000
        else:
            return None
    else:
        return None