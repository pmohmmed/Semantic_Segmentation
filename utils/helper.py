

def stg_msg(msg='', c=' ', l=50):
    bef = aft = int((l - len(msg)) / 2 ) * c
    if len(msg) % 2:
        aft += c

    print('\n-- ' + bef + msg + aft + ' --')


def save_msg(msg, c='|', l=2):
    for _ in range(l):
        print(c)
    print(f'-> {msg}')
