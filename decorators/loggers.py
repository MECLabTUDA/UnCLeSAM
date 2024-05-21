import logging, os, time
from functools import wraps

def timeit(func):
    """
    Decorator to print the runtime of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'{func.__name__} took {end - start:.6f} seconds to complete.')
        return result
    return wrapper

""" Example Use case:   
@timeit
def process_data():
    time.sleep(1)
process_data()
# process_data took 1.000012 seconds to complete
"""
def log_prints(file_path):
    def logger_func(func):
        @wraps(func)
        def wrapper_logger(*args, **kwargs):
            os.makedirs(os.path.join(os.sep, *file_path.split(os.sep)[:-1]), exist_ok=True)
            print("Prints will be logged at: {}.".format(file_path))

            logging.basicConfig(filename = file_path, format = '%(asctime)s %(message)s') 
            logging.captureWarnings(True)
            logger = logging.getLogger() 
            logger.setLevel(logging.DEBUG)
            
            try:
                value = func(*args, **kwargs)
            except Exception as err:
                print(err)
                logger.error("Error: " + str(err))
                value = None

            if value is not None: 
                print(' - '.join(value), flush=True)
                logger.debug(' - '.join(value))
            return value
        return wrapper_logger
    return logger_func 


""" Example Use case:
@log_prints(log_file)
def process_data():
    print("XYZ")
process_data()
# prints "XYZ" in terminal and in log_file.
"""