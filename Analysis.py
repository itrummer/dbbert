'''
Created on Nov 11, 2020

@author: immanueltrummer
'''
""" Analyze impact of tuning parameters. """

import Configurations

# Prepare tuning and perform NLP analysis
print("About to create tuning config")
tuning_config = Configurations.TuningConfig(
    '/etc/postgresql/10/main/postgresql.conf')

# Iterate over all parameters and try different settings
f = open("tpchAnalysis.txt", "w")
for lineID, param in tuning_config.idToTunable.items():
    for factor in (0.2, 5, 1):
        tuning_config.set_scale(lineID, factor)
        print(f'Trying with factor {factor} for {param.name}')
        """
        error, millis = tuning_config.evaluateConfig(
            'tpccs2', 'postgres', 'postgres')
        """
        error, millis = tuning_config.tpch_eval()
        print(f'Trying with factor {factor}')
        f.write(f'{param.name}\t{factor}\t{error}\t{millis}\n')
        f.flush()
f.close()
