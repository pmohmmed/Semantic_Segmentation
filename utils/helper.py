

def message(msg, c=' ', l=50):
    bef = aft = int((l - len(msg)) / 2 ) * c
    if len(msg) % 2:
        bef += ' '

    print()
    print('--' + bef + msg + aft + '--')
          

