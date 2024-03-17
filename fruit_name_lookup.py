'''
This module contains a function that takes in the fruits dataframe
and returns a dictionary that maps the fruit_label to the fruit_name.
'''

def fruit_name_lookup(fruits):
    '''
    parameters: fruits (DataFrame) - the fruits dataset
    returns: dict - a dictionary that maps the fruit_label to the fruit_name
    '''
    lookup_fruit_name = dict(zip(fruits['fruit_label'], fruits['fruit_name']))
    return lookup_fruit_name
