'''
Created on Nov 16, 2020

@author: immanueltrummer
'''
import os
import time

class MySQLeval:
    """ Benchmarks MySQL database """
    def __init__(self, config):
        self.config = config
        
    def change_config(self):
        """ Change configuration via restart, return error flag """
        # Stop MySQL server
        r_code = os.system('sudo /etc/init.d/mysql stop')
        print(f'stopped server with return {r_code}', 
              flush=True)
        # Remove old temporary configuration file
        r_code = os.system('sudo rm tmpConfig')
        print(f'removed configuration file with return {r_code}')
        # Write and copy new configuration file 
        with open('tmpConfig', 'w') as config_file:
            self.config.write(config_file)
        r_code = os.system('sudo cp --remove-destination ' \
                           'tmpConfig /etc/mysql/my.cnf')
        print(f'wrote configuration file with code {r_code}', 
              flush=True)
        # Delete old log files which may prevent server start
        r_code = os.system('sudo rm /var/lib/mysql/ib_logfile*')
        print(f'removed old log files with code {r_code}',
              flush=True)
        # Start server with new configuration
        r_code = os.system('sudo /etc/init.d/mysql start')
        print(f'started server with return {r_code}', 
              flush=True)
        time.sleep(5)
        return r_code != 0
        
    def tpch_eval(self):
        """ Evaluate current configuration with TPC-H """
        # Change configuration and iterate over TPC-H queries
        error = False
        times = [-1 for _ in range(22)]
        try:
            error |= self.change_config()
            print(f"Error status is {error}")
            if not error:
                print("About to run benchmark", flush=True)
                error = True
                for q in range(1, 23):
                    start_ms = time.time() * 1000.0
                    r_code = os.system(
                        f'sudo time mysql tpchs1 ' \
                        f'< queries/ms/{q}.sql')
                    print(f"Executed with code {r_code}", 
                          flush=True)
                    end_ms = time.time() * 1000.0
                    times[q-1] = end_ms - start_ms
                error = False
        except Exception as e:
            error = True
            print(f'Exception: {e}')
        return error, times