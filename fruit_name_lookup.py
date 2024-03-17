def fruit_name_lookup(fruits):
    '''
    This function takes in the fruits dataframe and returns a dictionary that maps the fruit_label to the fruit_name.
    parameters: fruits (DataFrame) - the fruits dataset
    returns: dict - a dictionary that maps the fruit_label to the fruit_name
    '''
    lookup_fruit_name = dict(zip(fruits['fruit_label'], fruits['fruit_name']))
    return lookup_fruit_name