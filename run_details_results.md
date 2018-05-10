## run1:
    lr: 1e-3
    Arch: Densenet
    models: saved in models1 folder
    ''' py
        def __init__(self, growth_rate=4, block_config=(1, 2, 3),
             num_init_features=8, bn_size=4, drop_rate=0.2, out_number=10):
    '''
    Results:
        Best Avg. accuracy: (66.666 + 57.15 + 57.15 + 58.334 + 58.334 + 66.666) /6.0 = 60.723 <== Validation Data Without Tencrops


## run2:
    lr: 1e-4
    Arch: Densenet
    models: saved in models2 folder
    ''' py
        def __init__(self, growth_rate=2, block_config=(1, 2, 3),
             num_init_features=8, bn_size=2, drop_rate=0.2, out_number=10):
    '''
    Results:
        Best Avg. accuracy: (59.723 + 60.70 + 64.281 + 58.334 + 58.333 + 62.5) /6.0 =  60.645 <== Validation Data Without Tencrops

  
## run3: 
    Arch: 3 layered CNN 
    lr: 1e-4
    models: saved in models3 folder
    ''' py
        def __init__(self, growth_rate=1, block_config=(1, 2, 3),
             num_init_features=10, bn_size=1, drop_rate=0.2, out_number=10):
    '''
    Results:
        Best Avg. accuracy: (61.111 + 58.991 + 58.932 + 55.55 + 58.33 + 62.5) /6.0 = 59.462 <== Validation Data Without Tencrops


## run4:
    scanner1 and scanner23: to increase datapoints...
    lr: 1e-4
    models: saved in models4 folder
    ''' py
        def __init__(self, growth_rate=2, block_config=(1, 2, 3),
             num_init_features=8, bn_size=2, drop_rate=0.2, out_number=10):
    '''
    Results:
        Best Avg. accuracy: (66.7 + 61.483) / 2.0 = 64.091 <== Validation Data Without Tencrops

